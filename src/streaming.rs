use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::format::RequestFormat;

pub fn maybe_transform_sse(
    raw: &[u8],
    upstream_format: Option<RequestFormat>,
    client_format: Option<RequestFormat>,
) -> Option<Vec<u8>> {
    match (upstream_format, client_format) {
        (Some(RequestFormat::OpenAiResponses), Some(RequestFormat::OpenAiChat)) => {
            Some(transform_openai_responses_to_openai_chat(raw))
        }
        _ => None,
    }
}

fn transform_openai_responses_to_openai_chat(raw: &[u8]) -> Vec<u8> {
    let text = String::from_utf8_lossy(raw);
    let mut response_id = String::new();
    let mut model = String::new();
    let mut created = 0_i64;
    let mut out = String::new();
    let mut started = false;
    let mut finished = false;
    let mut item_to_call_id = HashMap::<String, String>::new();
    let mut call_names = HashMap::<String, String>::new();
    let mut call_indexes = HashMap::<String, usize>::new();
    let mut started_tool_calls = HashSet::<String>::new();
    let mut pending_tool_args = HashMap::<String, Vec<String>>::new();

    for frame in text.split("\n\n") {
        let trimmed = frame.trim();
        if trimmed.is_empty() {
            continue;
        }

        let data = trimmed
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap_or("");

        if data == "[DONE]" {
            if finished {
                continue;
            }
            out.push_str("data: [DONE]\n\n");
            continue;
        }

        let Ok(payload) = serde_json::from_str::<Value>(data) else {
            continue;
        };

        match payload.get("type").and_then(Value::as_str) {
            Some("response.created") => {
                if let Some(response) = payload.get("response").and_then(Value::as_object) {
                    response_id = response
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    model = response
                        .get("model")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    created = response
                        .get("created_at")
                        .or_else(|| response.get("created"))
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                }
                ensure_created(&mut created);
                emit_chat_start_chunk(
                    &mut out,
                    &mut started,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                );
            }
            Some("response.in_progress") => {
                if let Some(response) = payload.get("response").and_then(Value::as_object) {
                    if response_id.is_empty() {
                        response_id = response
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                    }
                    if model.is_empty() {
                        model = response
                            .get("model")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                    }
                    if created == 0 {
                        created = response
                            .get("created_at")
                            .or_else(|| response.get("created"))
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                    }
                }
                ensure_created(&mut created);
                emit_chat_start_chunk(
                    &mut out,
                    &mut started,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                );
            }
            Some("response.output_text.delta") => {
                let delta = payload
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if delta.is_empty() {
                    continue;
                }
                ensure_created(&mut created);
                let chunk = json!({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": delta
                        },
                        "finish_reason": Value::Null
                    }]
                });
                out.push_str("data: ");
                out.push_str(&chunk.to_string());
                out.push_str("\n\n");
            }
            Some("response.output_item.added") => {
                let Some(item) = payload.get("item").and_then(Value::as_object) else {
                    continue;
                };
                if item.get("type").and_then(Value::as_str) != Some("function_call") {
                    continue;
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
                let call_index = payload
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0);
                let name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();

                if !item_id.is_empty() && !call_id.is_empty() {
                    item_to_call_id.insert(item_id.clone(), call_id.clone());
                }
                if !call_id.is_empty() && !name.is_empty() {
                    call_names.insert(call_id.clone(), name.clone());
                    call_indexes.insert(call_id.clone(), call_index);
                }

                ensure_created(&mut created);
                emit_chat_start_chunk(
                    &mut out,
                    &mut started,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                );

                if !call_id.is_empty() && !name.is_empty() && started_tool_calls.insert(call_id.clone()) {
                    emit_tool_call_chunk(
                        &mut out,
                        response_id.as_str(),
                        model.as_str(),
                        created,
                        call_indexes.get(&call_id).copied().unwrap_or(0),
                        call_id.as_str(),
                        name.as_str(),
                        "",
                    );
                }

                let pending = pending_tool_args
                    .remove(&item_id)
                    .or_else(|| pending_tool_args.remove(&call_id));
                if let Some(pending) = pending {
                    for delta in pending {
                        emit_tool_call_chunk(
                            &mut out,
                            response_id.as_str(),
                            model.as_str(),
                            created,
                            call_indexes.get(&call_id).copied().unwrap_or(0),
                            call_id.as_str(),
                            name.as_str(),
                            delta.as_str(),
                        );
                    }
                }
            }
            Some("response.function_call_arguments.delta")
            | Some("response.function_call_arguments.done")
            | Some("response.function_call.delta") => {
                let item_id = payload
                    .get("item_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let mut call_id = payload
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if call_id.is_empty() {
                    call_id = item_to_call_id.get(item_id).cloned().unwrap_or_default();
                }
                let delta = payload
                    .get("delta")
                    .or_else(|| payload.get("arguments"))
                    .map(|value| match value {
                        Value::String(text) => text.clone(),
                        Value::Object(_) | Value::Array(_) => {
                            serde_json::to_string(value).unwrap_or_else(|_| String::new())
                        }
                        _ => String::new(),
                    })
                    .unwrap_or_default();
                if delta.is_empty() {
                    continue;
                }

                let name = payload
                    .get("name")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| call_names.get(&call_id).cloned())
                    .unwrap_or_default();
                let call_index = payload
                    .get("output_index")
                    .and_then(Value::as_u64)
                    .and_then(|value| usize::try_from(value).ok())
                    .or_else(|| call_indexes.get(&call_id).copied())
                    .unwrap_or(0);
                if !call_id.is_empty() {
                    call_indexes.entry(call_id.clone()).or_insert(call_index);
                }

                ensure_created(&mut created);
                emit_chat_start_chunk(
                    &mut out,
                    &mut started,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                );

                if call_id.is_empty() {
                    if !item_id.is_empty() {
                        pending_tool_args
                            .entry(item_id.to_string())
                            .or_default()
                            .push(delta);
                    }
                    continue;
                }

                if !name.is_empty() && started_tool_calls.insert(call_id.clone()) {
                    emit_tool_call_chunk(
                        &mut out,
                        response_id.as_str(),
                        model.as_str(),
                        created,
                        call_index,
                        call_id.as_str(),
                        name.as_str(),
                        "",
                    );
                }

                if name.is_empty() {
                    pending_tool_args.entry(call_id).or_default().push(delta);
                    continue;
                }

                emit_tool_call_chunk(
                    &mut out,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                    call_index,
                    call_id.as_str(),
                    name.as_str(),
                    delta.as_str(),
                );
            }
            Some("response.completed") | Some("response.failed") | Some("response.incomplete") | Some("error") => {
                let finish_reason = match payload.get("type").and_then(Value::as_str) {
                    Some("error") | Some("response.failed") => "error",
                    Some("response.incomplete") => "length",
                    _ => "stop",
                };
                ensure_created(&mut created);
                emit_chat_start_chunk(
                    &mut out,
                    &mut started,
                    response_id.as_str(),
                    model.as_str(),
                    created,
                );
                let mut chunk = json!({
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish_reason
                    }]
                });
                if let Some(usage) = payload
                    .get("response")
                    .and_then(|response| response.get("usage"))
                    .and_then(normalize_stream_usage)
                {
                    chunk["usage"] = usage;
                }
                out.push_str("data: ");
                out.push_str(&chunk.to_string());
                out.push_str("\n\n");
                out.push_str("data: [DONE]\n\n");
                finished = true;
            }
            _ => {}
        }
    }

    out.into_bytes()
}

fn ensure_created(created: &mut i64) {
    if *created != 0 {
        return;
    }

    *created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0);
}

fn normalize_stream_usage(usage: &Value) -> Option<Value> {
    let usage = usage.as_object()?;
    Some(json!({
        "input_tokens": usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0)),
        "output_tokens": usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0)),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
    }))
}

fn emit_chat_start_chunk(
    out: &mut String,
    started: &mut bool,
    response_id: &str,
    model: &str,
    created: i64,
) {
    if *started {
        return;
    }

    *started = true;
    let chunk = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "role": "assistant"
            },
            "finish_reason": Value::Null
        }]
    });
    out.push_str("data: ");
    out.push_str(&chunk.to_string());
    out.push_str("\n\n");
}

fn emit_tool_call_chunk(
    out: &mut String,
    response_id: &str,
    model: &str,
    created: i64,
    call_index: usize,
    call_id: &str,
    name: &str,
    arguments: &str,
) {
    let chunk = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": call_index,
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments
                    }
                }]
            },
            "finish_reason": Value::Null
        }]
    });
    out.push_str("data: ");
    out.push_str(&chunk.to_string());
    out.push_str("\n\n");
}
