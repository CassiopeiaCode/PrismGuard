use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Map, Value};

use crate::format::RequestFormat;

pub fn maybe_transform_sse(
    raw: &[u8],
    upstream_format: Option<RequestFormat>,
    client_format: Option<RequestFormat>,
) -> Option<Vec<u8>> {
    let from_format = upstream_format?;
    let to_format = client_format?;
    if from_format == to_format {
        return None;
    }

    let mut transcoder = StreamTranscoder::new(from_format, to_format, None);
    let mut out = transcoder.feed_chunk(raw);
    out.extend(transcoder.flush());
    Some(out)
}

#[derive(Debug, Clone)]
struct SseFrame {
    event: Option<String>,
    data: String,
}

#[derive(Debug, Clone)]
enum InternalEvent {
    Start {
        meta: Map<String, Value>,
    },
    TextDelta {
        text: String,
    },
    ToolCallStart {
        call_id: String,
        name: String,
    },
    ToolCallArgsDelta {
        call_id: String,
        name: String,
        delta: String,
    },
    Final {
        finish_reason: Option<String>,
        usage: Option<Value>,
    },
    Done,
}

trait InternalSink: Send {
    fn on_start(&mut self, meta: &Map<String, Value>) -> Vec<Vec<u8>>;
    fn on_text_delta(&mut self, text: &str) -> Vec<Vec<u8>>;
    fn on_tool_call_start(&mut self, call_id: &str, name: &str) -> Vec<Vec<u8>>;
    fn on_tool_call_args_delta(&mut self, call_id: &str, name: &str, delta: &str) -> Vec<Vec<u8>>;
    fn on_final(&mut self, finish_reason: Option<&str>, usage: Option<&Value>) -> Vec<Vec<u8>>;
    fn on_done(&mut self) -> Vec<Vec<u8>>;
}

pub struct StreamTranscoder {
    from_format: RequestFormat,
    sink: Box<dyn InternalSink>,
    meta: Map<String, Value>,
    started: bool,
    seen_tool_calls: HashMap<String, Map<String, Value>>,
    pending: Vec<u8>,
}

impl StreamTranscoder {
    pub fn new(
        from_format: RequestFormat,
        to_format: RequestFormat,
        estimated_prompt_tokens: Option<i64>,
    ) -> Self {
        let mut meta = Map::new();
        if let Some(tokens) = estimated_prompt_tokens.filter(|tokens| *tokens > 0) {
            meta.insert(
                "usage".to_string(),
                json!({
                    "prompt_tokens": tokens,
                    "completion_tokens": 0,
                    "total_tokens": tokens
                }),
            );
        }
        Self {
            from_format,
            sink: create_sink(to_format),
            meta,
            started: false,
            seen_tool_calls: HashMap::new(),
            pending: Vec::new(),
        }
    }

    pub fn feed_chunk(&mut self, raw: &[u8]) -> Vec<u8> {
        self.pending.extend_from_slice(raw);
        let mut out = Vec::new();
        for frame in take_complete_sse_frames(&mut self.pending) {
            for event in self.decode_frame(frame) {
                out.extend(self.emit_internal_event(event));
            }
        }
        out.concat()
    }

    pub fn flush(&mut self) -> Vec<u8> {
        let mut out = Vec::new();
        for frame in take_trailing_sse_frames(&mut self.pending) {
            for event in self.decode_frame(frame) {
                out.extend(self.emit_internal_event(event));
            }
        }
        out.concat()
    }

    fn decode_frame(&mut self, frame: SseFrame) -> Vec<InternalEvent> {
        let data = frame.data.trim();
        if data.is_empty() {
            return Vec::new();
        }
        if data == "[DONE]" {
            return vec![InternalEvent::Done];
        }

        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };
        let Some(obj) = payload.as_object() else {
            return Vec::new();
        };
        let event_name = frame
            .event
            .as_deref()
            .or_else(|| obj.get("type").and_then(Value::as_str))
            .or_else(|| obj.get("event").and_then(Value::as_str));
        decode_to_internal(
            self.from_format,
            event_name,
            obj,
            &mut self.meta,
            &mut self.seen_tool_calls,
        )
    }

    fn emit_internal_event(&mut self, event: InternalEvent) -> Vec<Vec<u8>> {
        match event {
            InternalEvent::Start { meta } => {
                self.started = true;
                self.sink.on_start(&meta)
            }
            InternalEvent::TextDelta { text } => {
                let mut out = Vec::new();
                if !self.started {
                    self.started = true;
                    out.extend(self.sink.on_start(&self.meta));
                }
                out.extend(self.sink.on_text_delta(&text));
                out
            }
            InternalEvent::ToolCallStart { call_id, name } => {
                let mut out = Vec::new();
                if !self.started {
                    self.started = true;
                    out.extend(self.sink.on_start(&self.meta));
                }
                out.extend(self.sink.on_tool_call_start(&call_id, &name));
                out
            }
            InternalEvent::ToolCallArgsDelta {
                call_id,
                name,
                delta,
            } => {
                let mut out = Vec::new();
                if !self.started {
                    self.started = true;
                    out.extend(self.sink.on_start(&self.meta));
                }
                out.extend(self.sink.on_tool_call_args_delta(&call_id, &name, &delta));
                out
            }
            InternalEvent::Final {
                finish_reason,
                usage,
            } => {
                let mut out = Vec::new();
                if !self.started {
                    self.started = true;
                    out.extend(self.sink.on_start(&self.meta));
                }
                out.extend(self.sink.on_final(finish_reason.as_deref(), usage.as_ref()));
                out
            }
            InternalEvent::Done => self.sink.on_done(),
        }
    }
}

fn create_sink(format: RequestFormat) -> Box<dyn InternalSink> {
    match format {
        RequestFormat::OpenAiChat => Box::new(OpenAiChatSink::default()),
        RequestFormat::OpenAiResponses => Box::new(OpenAiResponsesSink::default()),
        RequestFormat::ClaudeChat => Box::new(ClaudeSink::default()),
        RequestFormat::GeminiChat => Box::new(GeminiSink::default()),
    }
}

#[derive(Default)]
struct OpenAiChatSink {
    response_id: String,
    model: String,
    created: i64,
    role_sent: bool,
    done_sent: bool,
    tool_call_indexes: HashMap<String, usize>,
    next_tool_call_index: usize,
}

impl InternalSink for OpenAiChatSink {
    fn on_start(&mut self, meta: &Map<String, Value>) -> Vec<Vec<u8>> {
        if self.response_id.is_empty() {
            self.response_id = meta_string(meta, "id");
        }
        if self.model.is_empty() {
            self.model = meta_string(meta, "model");
        }
        if self.created == 0 {
            self.created = meta
                .get("created")
                .or_else(|| meta.get("created_at"))
                .and_then(Value::as_i64)
                .unwrap_or_else(now_timestamp);
        }
        if self.role_sent {
            return Vec::new();
        }
        self.role_sent = true;
        vec![encode_json_sse(&json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": Value::Null
            }]
        }))]
    }

    fn on_text_delta(&mut self, text: &str) -> Vec<Vec<u8>> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse(&json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": Value::Null
            }]
        }))]
    }

    fn on_tool_call_start(&mut self, call_id: &str, name: &str) -> Vec<Vec<u8>> {
        if call_id.is_empty() && name.is_empty() {
            return Vec::new();
        }
        let call_index = self.tool_call_index(call_id);
        vec![encode_json_sse(&json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": call_index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""}
                    }]
                },
                "finish_reason": Value::Null
            }]
        }))]
    }

    fn on_tool_call_args_delta(&mut self, call_id: &str, name: &str, delta: &str) -> Vec<Vec<u8>> {
        if delta.is_empty() {
            return Vec::new();
        }
        let call_index = self.tool_call_index(call_id);
        vec![encode_json_sse(&json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": call_index,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": delta}
                    }]
                },
                "finish_reason": Value::Null
            }]
        }))]
    }

    fn on_final(&mut self, finish_reason: Option<&str>, usage: Option<&Value>) -> Vec<Vec<u8>> {
        let mut chunk = json!({
            "id": self.response_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        if let Some(usage) = usage.cloned() {
            chunk["usage"] = usage;
        }
        vec![encode_json_sse(&chunk)]
    }

    fn on_done(&mut self) -> Vec<Vec<u8>> {
        if self.done_sent {
            return Vec::new();
        }
        self.done_sent = true;
        vec![encode_sse("[DONE]", None)]
    }
}

impl OpenAiChatSink {
    fn tool_call_index(&mut self, call_id: &str) -> usize {
        if call_id.is_empty() {
            return 0;
        }
        if let Some(index) = self.tool_call_indexes.get(call_id).copied() {
            return index;
        }
        let index = self.next_tool_call_index;
        self.next_tool_call_index += 1;
        self.tool_call_indexes.insert(call_id.to_string(), index);
        index
    }
}

#[derive(Default)]
struct OpenAiResponsesSink {
    response_id: String,
    model: String,
    created_at: i64,
    started: bool,
}

impl InternalSink for OpenAiResponsesSink {
    fn on_start(&mut self, meta: &Map<String, Value>) -> Vec<Vec<u8>> {
        if self.response_id.is_empty() {
            self.response_id = meta_string(meta, "id");
        }
        if self.model.is_empty() {
            self.model = meta_string(meta, "model");
        }
        if self.created_at == 0 {
            self.created_at = meta
                .get("created_at")
                .or_else(|| meta.get("created"))
                .and_then(Value::as_i64)
                .unwrap_or_else(now_timestamp);
        }
        if self.started {
            return Vec::new();
        }
        self.started = true;
        let response = json!({
            "id": self.response_id,
            "model": self.model,
            "created_at": self.created_at,
            "status": "in_progress"
        });
        vec![
            encode_json_sse_with_event(
                &json!({"type": "response.created", "response": response}),
                "response.created",
            ),
            encode_json_sse_with_event(
                &json!({"type": "response.in_progress", "response": {
                    "id": self.response_id,
                    "model": self.model,
                    "created_at": self.created_at,
                    "status": "in_progress"
                }}),
                "response.in_progress",
            ),
        ]
    }

    fn on_text_delta(&mut self, text: &str) -> Vec<Vec<u8>> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse_with_event(
            &json!({"type": "response.output_text.delta", "delta": text}),
            "response.output_text.delta",
        )]
    }

    fn on_tool_call_start(&mut self, call_id: &str, name: &str) -> Vec<Vec<u8>> {
        if call_id.is_empty() && name.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse_with_event(
            &json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": call_id,
                    "call_id": call_id,
                    "type": "function_call",
                    "name": name
                }
            }),
            "response.output_item.added",
        )]
    }

    fn on_tool_call_args_delta(&mut self, call_id: &str, _name: &str, delta: &str) -> Vec<Vec<u8>> {
        if delta.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse_with_event(
            &json!({
                "type": "response.function_call_arguments.delta",
                "call_id": call_id,
                "item_id": call_id,
                "delta": delta
            }),
            "response.function_call_arguments.delta",
        )]
    }

    fn on_final(&mut self, finish_reason: Option<&str>, usage: Option<&Value>) -> Vec<Vec<u8>> {
        let status = match finish_reason {
            Some("length") => "incomplete",
            Some("error") => "failed",
            _ => "completed",
        };
        let mut response = json!({
            "id": self.response_id,
            "model": self.model,
            "created_at": self.created_at,
            "status": status
        });
        if let Some(usage) = usage.and_then(chat_usage_to_responses_usage) {
            response["usage"] = usage;
        }
        let event = match status {
            "incomplete" => "response.incomplete",
            "failed" => "response.failed",
            _ => "response.completed",
        };
        vec![encode_json_sse_with_event(
            &json!({"type": event, "response": response}),
            event,
        )]
    }

    fn on_done(&mut self) -> Vec<Vec<u8>> {
        vec![encode_sse("[DONE]", None)]
    }
}

#[derive(Default)]
struct ClaudeSink {
    id: String,
    model: String,
    started: bool,
}

impl InternalSink for ClaudeSink {
    fn on_start(&mut self, meta: &Map<String, Value>) -> Vec<Vec<u8>> {
        if self.id.is_empty() {
            self.id = meta_string(meta, "id");
        }
        if self.model.is_empty() {
            self.model = meta_string(meta, "model");
        }
        if self.started {
            return Vec::new();
        }
        self.started = true;
        let usage = meta
            .get("usage")
            .and_then(chat_usage_to_claude_stream_usage)
            .unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0}));
        vec![encode_json_sse_with_event(
            &json!({
                "type": "message_start",
                "message": {
                    "id": self.id,
                    "model": self.model,
                    "role": "assistant",
                    "content": [],
                    "usage": usage
                }
            }),
            "message_start",
        )]
    }

    fn on_text_delta(&mut self, text: &str) -> Vec<Vec<u8>> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse_with_event(
            &json!({
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": text
                }
            }),
            "content_block_delta",
        )]
    }

    fn on_tool_call_start(&mut self, call_id: &str, name: &str) -> Vec<Vec<u8>> {
        vec![encode_json_sse_with_event(
            &json!({
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {}
                }
            }),
            "content_block_start",
        )]
    }

    fn on_tool_call_args_delta(&mut self, _call_id: &str, _name: &str, delta: &str) -> Vec<Vec<u8>> {
        if delta.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse_with_event(
            &json!({
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": delta
                }
            }),
            "content_block_delta",
        )]
    }

    fn on_final(&mut self, finish_reason: Option<&str>, usage: Option<&Value>) -> Vec<Vec<u8>> {
        let stop_reason = match finish_reason {
            Some("length") => "max_tokens",
            Some("error") => "error",
            _ => "end_turn",
        };
        let usage_obj = usage
            .and_then(chat_usage_to_claude_stream_usage)
            .unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0}));
        vec![
            encode_json_sse_with_event(
                &json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": Value::Null},
                    "usage": usage_obj
                }),
                "message_delta",
            ),
            encode_json_sse_with_event(&json!({"type": "message_stop"}), "message_stop"),
        ]
    }

    fn on_done(&mut self) -> Vec<Vec<u8>> {
        vec![encode_sse("[DONE]", None)]
    }
}

#[derive(Default)]
struct GeminiSink {
    id: String,
    model: String,
}

impl InternalSink for GeminiSink {
    fn on_start(&mut self, meta: &Map<String, Value>) -> Vec<Vec<u8>> {
        if self.id.is_empty() {
            self.id = meta_string(meta, "id");
        }
        if self.model.is_empty() {
            self.model = meta_string(meta, "model");
        }
        Vec::new()
    }

    fn on_text_delta(&mut self, text: &str) -> Vec<Vec<u8>> {
        if text.is_empty() {
            return Vec::new();
        }
        vec![encode_json_sse(&json!({
            "responseId": self.id,
            "modelVersion": self.model,
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": text}]
                }
            }]
        }))]
    }

    fn on_tool_call_start(&mut self, _call_id: &str, _name: &str) -> Vec<Vec<u8>> {
        Vec::new()
    }

    fn on_tool_call_args_delta(&mut self, call_id: &str, name: &str, delta: &str) -> Vec<Vec<u8>> {
        let args = serde_json::from_str::<Value>(delta).unwrap_or_else(|_| json!({}));
        vec![encode_json_sse(&json!({
            "responseId": self.id,
            "modelVersion": self.model,
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "id": call_id,
                            "name": name,
                            "args": args
                        }
                    }]
                }
            }]
        }))]
    }

    fn on_final(&mut self, finish_reason: Option<&str>, _usage: Option<&Value>) -> Vec<Vec<u8>> {
        let finish_reason = match finish_reason {
            Some("length") => "MAX_TOKENS",
            Some("error") => "ERROR",
            _ => "STOP",
        };
        vec![encode_json_sse(&json!({
            "responseId": self.id,
            "modelVersion": self.model,
            "candidates": [{
                "finishReason": finish_reason
            }]
        }))]
    }

    fn on_done(&mut self) -> Vec<Vec<u8>> {
        vec![encode_sse("[DONE]", None)]
    }
}

fn parse_sse_frames(text: &str) -> Vec<SseFrame> {
    let mut frames = Vec::new();
    for raw in text.split("\n\n") {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut event = None;
        let mut data_lines = Vec::new();
        for line in raw.lines() {
            if let Some(value) = line.strip_prefix("event:") {
                let value = value.trim();
                if !value.is_empty() {
                    event = Some(value.to_string());
                }
            } else if let Some(value) = line.strip_prefix("data:") {
                data_lines.push(value.trim_start().to_string());
            }
        }
        if data_lines.is_empty() && event.is_none() {
            continue;
        }
        frames.push(SseFrame {
            event,
            data: data_lines.join("\n"),
        });
    }
    frames
}

fn take_complete_sse_frames(pending: &mut Vec<u8>) -> Vec<SseFrame> {
    let mut frames = Vec::new();

    while let Some(end) = find_sse_frame_end(pending) {
        let frame_bytes = pending.drain(..end).collect::<Vec<_>>();
        let text = String::from_utf8_lossy(&frame_bytes);
        frames.extend(parse_sse_frames(&text));
    }

    frames
}

fn take_trailing_sse_frames(pending: &mut Vec<u8>) -> Vec<SseFrame> {
    if pending.is_empty() {
        return Vec::new();
    }

    let frame_bytes = std::mem::take(pending);
    let text = String::from_utf8_lossy(&frame_bytes);
    parse_sse_frames(&text)
}

fn find_sse_frame_end(buffer: &[u8]) -> Option<usize> {
    let mut idx = 0;
    while idx < buffer.len() {
        if idx + 1 < buffer.len() && buffer[idx] == b'\n' && buffer[idx + 1] == b'\n' {
            return Some(idx + 2);
        }
        if idx + 3 < buffer.len()
            && buffer[idx] == b'\r'
            && buffer[idx + 1] == b'\n'
            && buffer[idx + 2] == b'\r'
            && buffer[idx + 3] == b'\n'
        {
            return Some(idx + 4);
        }
        idx += 1;
    }
    None
}

fn decode_to_internal(
    from_format: RequestFormat,
    event_name: Option<&str>,
    event: &Map<String, Value>,
    meta: &mut Map<String, Value>,
    seen_tool_calls: &mut HashMap<String, Map<String, Value>>,
) -> Vec<InternalEvent> {
    match from_format {
        RequestFormat::OpenAiChat => decode_openai_chat(event, meta, seen_tool_calls),
        RequestFormat::OpenAiResponses => decode_openai_responses(event_name, event, meta, seen_tool_calls),
        RequestFormat::ClaudeChat => decode_claude(event_name, event, meta, seen_tool_calls),
        RequestFormat::GeminiChat => decode_gemini(event, meta, seen_tool_calls),
    }
}

fn decode_openai_chat(
    event: &Map<String, Value>,
    meta: &mut Map<String, Value>,
    seen_tool_calls: &mut HashMap<String, Map<String, Value>>,
) -> Vec<InternalEvent> {
    let Some(choice) = event
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };

    if let Some(id) = event.get("id").cloned() {
        meta.insert("id".to_string(), id);
    }
    if let Some(model) = event.get("model").cloned() {
        meta.insert("model".to_string(), model);
    }
    if let Some(created) = event.get("created").cloned() {
        meta.insert("created".to_string(), created);
    } else {
        meta.entry("created".to_string())
            .or_insert_with(|| json!(now_timestamp()));
    }
    if let Some(usage) = event.get("usage").cloned() {
        meta.insert("usage".to_string(), usage);
    }

    let mut out = vec![InternalEvent::Start { meta: meta.clone() }];
    let delta = choice.get("delta").and_then(Value::as_object);
    if let Some(text) = delta.and_then(|delta| delta.get("content")).and_then(Value::as_str) {
        if !text.is_empty() {
            out.push(InternalEvent::TextDelta {
                text: text.to_string(),
            });
        }
    }

    if let Some(tool_calls) = delta.and_then(|delta| delta.get("tool_calls")).and_then(Value::as_array) {
        for tool_call in tool_calls.iter().filter_map(Value::as_object) {
            let mut call_id = tool_call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let idx = tool_call.get("index").and_then(Value::as_u64);
            if call_id.is_empty() {
                if let Some(idx) = idx {
                    call_id = seen_tool_calls
                        .get("__openai_chat_tool_index_map__")
                        .and_then(|index_map| index_map.get(&idx.to_string()))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                }
            } else if let Some(idx) = idx {
                seen_tool_calls
                    .entry("__openai_chat_tool_index_map__".to_string())
                    .or_default()
                    .insert(idx.to_string(), Value::String(call_id.clone()));
            }
            let function = tool_call
                .get("function")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let entry = seen_tool_calls.entry(call_id.clone()).or_default();
            let known_name = entry
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let final_name = if !name.is_empty() { name.clone() } else { known_name };
            if !call_id.is_empty() && entry.get("started").is_none() {
                entry.insert("started".to_string(), Value::Bool(true));
                if !final_name.is_empty() {
                    out.push(InternalEvent::ToolCallStart {
                        call_id: call_id.clone(),
                        name: final_name.clone(),
                    });
                }
            }
            if !name.is_empty() {
                entry.insert("name".to_string(), Value::String(name.clone()));
            }
            if !args.is_empty() && !call_id.is_empty() {
                out.push(InternalEvent::ToolCallArgsDelta {
                    call_id,
                    name: final_name,
                    delta: args,
                });
            }
        }
    }

    if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
        out.push(InternalEvent::Final {
            finish_reason: Some(finish_reason.to_string()),
            usage: event.get("usage").cloned(),
        });
        out.push(InternalEvent::Done);
    }
    out
}

fn decode_openai_responses(
    event_name: Option<&str>,
    event: &Map<String, Value>,
    meta: &mut Map<String, Value>,
    seen_tool_calls: &mut HashMap<String, Map<String, Value>>,
) -> Vec<InternalEvent> {
    let etype = event_name
        .or_else(|| event.get("type").and_then(Value::as_str))
        .unwrap_or_default();
    let mut out = Vec::new();
    let resp = event.get("response").and_then(Value::as_object);

    match etype {
        "response.created" | "response.in_progress" => {
            if let Some(resp) = resp {
                if let Some(id) = resp.get("id").cloned() {
                    meta.insert("id".to_string(), id);
                }
                if let Some(model) = resp.get("model").cloned() {
                    meta.insert("model".to_string(), model);
                }
                if let Some(created_at) = resp
                    .get("created_at")
                    .cloned()
                    .or_else(|| resp.get("created").cloned())
                {
                    meta.insert("created_at".to_string(), created_at);
                }
            }
            out.push(InternalEvent::Start { meta: meta.clone() });
        }
        "response.output_text.delta" => {
            if let Some(delta) = event
                .get("delta")
                .or_else(|| event.get("text"))
                .and_then(Value::as_str)
            {
                if !delta.is_empty() {
                    out.push(InternalEvent::TextDelta {
                        text: delta.to_string(),
                    });
                }
            }
        }
        "response.output_item.added" => {
            let Some(item) = event.get("item").and_then(Value::as_object) else {
                return out;
            };
            if item.get("type").and_then(Value::as_str) != Some("function_call") {
                return out;
            }
            let item_id = item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let call_id = item
                .get("call_id")
                .or_else(|| item.get("id"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if !item_id.is_empty() && !call_id.is_empty() {
                seen_tool_calls
                    .entry("__responses_item_id_map__".to_string())
                    .or_default()
                    .insert(item_id.clone(), Value::String(call_id.clone()));
            }
            let entry = seen_tool_calls.entry(call_id.clone()).or_default();
            if !name.is_empty() {
                entry.insert("name".to_string(), Value::String(name.clone()));
            }
            if entry.get("started").is_none() && !call_id.is_empty() && !name.is_empty() {
                entry.insert("started".to_string(), Value::Bool(true));
                out.push(InternalEvent::ToolCallStart {
                    call_id: call_id.clone(),
                    name: name.clone(),
                });
            }
            let pending_call = seen_tool_calls
                .get("__responses_pending_args__")
                .and_then(|map| map.get(&call_id))
                .cloned();
            if let Some(Value::Array(pending)) = pending_call {
                for delta in pending.iter().filter_map(Value::as_str) {
                    out.push(InternalEvent::ToolCallArgsDelta {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        delta: delta.to_string(),
                    });
                }
            }
            if let Some(map) = seen_tool_calls.get_mut("__responses_pending_args__") {
                map.remove(&call_id);
            }
            let pending_item = seen_tool_calls
                .get("__responses_pending_args_by_item__")
                .and_then(|map| map.get(&item_id))
                .cloned();
            if let Some(Value::Array(pending)) = pending_item {
                for delta in pending.iter().filter_map(Value::as_str) {
                    out.push(InternalEvent::ToolCallArgsDelta {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        delta: delta.to_string(),
                    });
                }
            }
            if let Some(map) = seen_tool_calls.get_mut("__responses_pending_args_by_item__") {
                map.remove(&item_id);
            }
        }
        "response.function_call_arguments.delta"
        | "response.function_call_arguments.done"
        | "response.function_call.delta" => {
            let item_id = event
                .get("item_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let mut call_id = event
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            if call_id.is_empty() {
                call_id = seen_tool_calls
                    .get("__responses_item_id_map__")
                    .and_then(|map| map.get(&item_id))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
            }
            let delta = event
                .get("delta")
                .or_else(|| event.get("arguments"))
                .map(stringify_json_value)
                .unwrap_or_default();
            if delta.is_empty() {
                return out;
            }
            if call_id.is_empty() {
                let pending_by_item = seen_tool_calls
                    .entry("__responses_pending_args_by_item__".to_string())
                    .or_default();
                push_string_array_value(pending_by_item, &item_id, delta);
                return out;
            }
            let name = event
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| {
                    seen_tool_calls
                        .get(&call_id)
                        .and_then(|entry| entry.get("name"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .unwrap_or_default();
            let started = seen_tool_calls
                .get(&call_id)
                .and_then(|entry| entry.get("started"))
                .is_some();
            if !started && name.is_empty() {
                let pending_by_call = seen_tool_calls
                    .entry("__responses_pending_args__".to_string())
                    .or_default();
                push_string_array_value(pending_by_call, &call_id, delta);
                return out;
            }
            let entry = seen_tool_calls.entry(call_id.clone()).or_default();
            if !name.is_empty() {
                entry.insert("name".to_string(), Value::String(name.clone()));
            }
            if entry.get("started").is_none() {
                entry.insert("started".to_string(), Value::Bool(true));
                out.push(InternalEvent::ToolCallStart {
                    call_id: call_id.clone(),
                    name: name.clone(),
                });
            }
            out.push(InternalEvent::ToolCallArgsDelta {
                call_id,
                name,
                delta,
            });
        }
        "response.completed" | "response.failed" | "response.incomplete" | "error" => {
            let finish_reason = match etype {
                "response.failed" | "error" => Some("error".to_string()),
                "response.incomplete" => Some("length".to_string()),
                _ => Some("stop".to_string()),
            };
            let usage = resp.and_then(|resp| resp.get("usage")).and_then(normalize_stream_usage);
            out.push(InternalEvent::Final {
                finish_reason,
                usage,
            });
            out.push(InternalEvent::Done);
        }
        _ => {}
    }
    out
}

fn decode_claude(
    event_name: Option<&str>,
    event: &Map<String, Value>,
    meta: &mut Map<String, Value>,
    seen_tool_calls: &mut HashMap<String, Map<String, Value>>,
) -> Vec<InternalEvent> {
    let dtype = event_name
        .or_else(|| event.get("type").and_then(Value::as_str))
        .unwrap_or_default();
    let mut out = Vec::new();
    match dtype {
        "message_start" => {
            if let Some(message) = event.get("message").and_then(Value::as_object) {
                if let Some(id) = message.get("id").cloned() {
                    meta.insert("id".to_string(), id);
                }
                if let Some(model) = message.get("model").cloned() {
                    meta.insert("model".to_string(), model);
                }
            }
            out.push(InternalEvent::Start { meta: meta.clone() });
        }
        "content_block_start" => {
            let Some(block) = event.get("content_block").and_then(Value::as_object) else {
                return out;
            };
            if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                let call_id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let entry = seen_tool_calls.entry(call_id.clone()).or_default();
                if entry.get("started").is_none() && !call_id.is_empty() {
                    entry.insert("started".to_string(), Value::Bool(true));
                    entry.insert("name".to_string(), Value::String(name.clone()));
                    out.push(InternalEvent::ToolCallStart {
                        call_id: call_id.clone(),
                        name: name.clone(),
                    });
                }
                if let Some(input) = block.get("input") {
                    let delta = stringify_json_value(input);
                    if delta != "{}" && !delta.is_empty() {
                        out.push(InternalEvent::ToolCallArgsDelta {
                            call_id,
                            name,
                            delta,
                        });
                    }
                }
            }
        }
        "content_block_delta" => {
            let Some(delta) = event.get("delta").and_then(Value::as_object) else {
                return out;
            };
            match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => {
                    if let Some(text) = delta.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            out.push(InternalEvent::TextDelta {
                                text: text.to_string(),
                            });
                        }
                    }
                }
                Some("input_json_delta") => {
                    if let Some(partial) = delta.get("partial_json").and_then(Value::as_str) {
                        if let Some((call_id, entry)) = seen_tool_calls
                            .iter()
                            .find(|(key, _)| !key.starts_with("__"))
                        {
                            let name = entry
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string();
                            out.push(InternalEvent::ToolCallArgsDelta {
                                call_id: call_id.clone(),
                                name,
                                delta: partial.to_string(),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        "message_delta" => {
            let finish_reason = event
                .get("delta")
                .and_then(Value::as_object)
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str)
                .map(|reason| match reason {
                    "max_tokens" => "length".to_string(),
                    "error" => "error".to_string(),
                    _ => "stop".to_string(),
                });
            let usage = event.get("usage").cloned();
            out.push(InternalEvent::Final {
                finish_reason,
                usage,
            });
        }
        "message_stop" => out.push(InternalEvent::Done),
        _ => {}
    }
    out
}

fn decode_gemini(
    event: &Map<String, Value>,
    meta: &mut Map<String, Value>,
    seen_tool_calls: &mut HashMap<String, Map<String, Value>>,
) -> Vec<InternalEvent> {
    let Some(candidate) = event
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };

    meta.entry("id".to_string())
        .or_insert_with(|| event.get("responseId").cloned().unwrap_or_else(|| json!("gemini-stream")));
    meta.entry("model".to_string()).or_insert_with(|| {
        event.get("modelVersion")
            .cloned()
            .unwrap_or_else(|| json!("gemini"))
    });

    let mut out = vec![InternalEvent::Start { meta: meta.clone() }];
    if let Some(content) = candidate.get("content").and_then(Value::as_object) {
        if let Some(parts) = content.get("parts").and_then(Value::as_array) {
            for part in parts.iter().filter_map(Value::as_object) {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        out.push(InternalEvent::TextDelta {
                            text: text.to_string(),
                        });
                    }
                } else if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                    let call_id = function_call
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("gemini_call")
                        .to_string();
                    let name = function_call
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let entry = seen_tool_calls.entry(call_id.clone()).or_default();
                    if entry.get("started").is_none() {
                        entry.insert("started".to_string(), Value::Bool(true));
                        entry.insert("name".to_string(), Value::String(name.clone()));
                        out.push(InternalEvent::ToolCallStart {
                            call_id: call_id.clone(),
                            name: name.clone(),
                        });
                    }
                    let args = function_call.get("args").cloned().unwrap_or_else(|| json!({}));
                    out.push(InternalEvent::ToolCallArgsDelta {
                        call_id,
                        name,
                        delta: stringify_json_value(&args),
                    });
                }
            }
        }
    }
    if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
        let finish_reason = match finish_reason {
            "MAX_TOKENS" => "length",
            "ERROR" => "error",
            _ => "stop",
        };
        out.push(InternalEvent::Final {
            finish_reason: Some(finish_reason.to_string()),
            usage: None,
        });
        out.push(InternalEvent::Done);
    }
    out
}

fn normalize_stream_usage(usage: &Value) -> Option<Value> {
    let usage = usage.as_object()?;
    Some(json!({
        "input_tokens": usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0)),
        "output_tokens": usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0)),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
    }))
}

fn chat_usage_to_responses_usage(usage: &Value) -> Option<Value> {
    let usage = usage.as_object()?;
    Some(json!({
        "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or_else(|| usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0))),
        "output_tokens": usage.get("completion_tokens").cloned().unwrap_or_else(|| usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0))),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
    }))
}

fn chat_usage_to_claude_stream_usage(usage: &Value) -> Option<Value> {
    let usage = usage.as_object()?;
    let prompt_details = usage.get("prompt_tokens_details").and_then(Value::as_object);
    Some(json!({
        "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or_else(|| usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0))),
        "output_tokens": usage.get("completion_tokens").cloned().unwrap_or_else(|| usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0))),
        "cache_creation_input_tokens": prompt_details.and_then(|details| details.get("cached_creation_tokens").cloned()).unwrap_or_else(|| json!(0)),
        "cache_read_input_tokens": prompt_details.and_then(|details| details.get("cached_tokens").cloned()).unwrap_or_else(|| json!(0)),
    }))
}

fn push_string_array_value(map: &mut Map<String, Value>, key: &str, value: String) {
    if key.is_empty() {
        return;
    }
    let entry = map
        .entry(key.to_string())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Some(items) = entry.as_array_mut() {
        items.push(Value::String(value));
    }
}

fn stringify_json_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Object(_) | Value::Array(_) => serde_json::to_string(value).unwrap_or_default(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn meta_string(meta: &Map<String, Value>, key: &str) -> String {
    meta.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn now_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn encode_json_sse(payload: &Value) -> Vec<u8> {
    encode_sse(&payload.to_string(), None)
}

fn encode_json_sse_with_event(payload: &Value, event: &str) -> Vec<u8> {
    encode_sse(&payload.to_string(), Some(event))
}

fn encode_sse(data: &str, event: Option<&str>) -> Vec<u8> {
    let mut out = String::new();
    if let Some(event) = event {
        out.push_str("event: ");
        out.push_str(event);
        out.push('\n');
    }
    out.push_str("data: ");
    out.push_str(data);
    out.push_str("\n\n");
    out.into_bytes()
}
