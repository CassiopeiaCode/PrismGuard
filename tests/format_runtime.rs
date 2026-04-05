#![allow(dead_code)]

use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestFormat {
    OpenAiChat,
    ClaudeChat,
    OpenAiResponses,
    GeminiChat,
}

impl RequestFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiChat => "openai_chat",
            Self::ClaudeChat => "claude_chat",
            Self::OpenAiResponses => "openai_responses",
            Self::GeminiChat => "gemini_chat",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "openai_chat" => Some(Self::OpenAiChat),
            "claude_chat" => Some(Self::ClaudeChat),
            "openai_responses" => Some(Self::OpenAiResponses),
            "gemini_chat" => Some(Self::GeminiChat),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RequestPlan {
    pub source_format: Option<RequestFormat>,
    pub target_format: Option<RequestFormat>,
    pub body: Value,
    pub path: String,
    pub stream: bool,
}

#[derive(Debug)]
pub enum RequestProcessError {
    StrictParse(String),
    Transform(String),
}

#[derive(Debug, Clone)]
struct InternalRequest {
    messages: Vec<InternalMessage>,
    model: String,
    stream: bool,
    tools: Vec<InternalTool>,
    tool_choice: Option<Value>,
    extra: Map<String, Value>,
}

#[derive(Debug, Clone)]
struct InternalMessage {
    role: String,
    content: Vec<InternalContentBlock>,
}

#[derive(Debug, Clone)]
enum InternalContentBlock {
    Text(String),
    ToolCall {
        id: String,
        name: String,
        arguments: Value,
    },
    ToolResult {
        call_id: String,
        name: Option<String>,
        output: Value,
    },
    ImageUrl {
        url: String,
        detail: Option<String>,
    },
}

#[derive(Debug, Clone)]
struct InternalTool {
    name: String,
    description: Option<String>,
    input_schema: Value,
}

pub fn process_request(
    config: &Value,
    path: &str,
    headers: &[(String, String)],
    body: Value,
) -> std::result::Result<RequestPlan, RequestProcessError> {
    let mut plan = RequestPlan {
        source_format: None,
        target_format: None,
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        body,
        path: path.to_string(),
    };

    let transform_cfg = config
        .get("format_transform")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let transform_enabled = transform_cfg
        .get("enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    if !transform_enabled {
        return Ok(plan);
    }

    let strict_parse = transform_cfg
        .get("strict_parse")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let disable_tools = transform_cfg
        .get("disable_tools")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let from_cfg = transform_cfg.get("from");
    let candidates = configured_candidates(from_cfg);
    let detectable = detect_formats_from_candidates(&candidates, path, headers, &plan.body);
    let mut parse_errors = Vec::new();
    let mut parsed = None;

    for source in detectable.iter().copied() {
        match parse_request(source, &plan.body, path) {
            Ok(internal) => {
                parsed = Some((source, internal));
                break;
            }
            Err(error) => parse_errors.push(format!("{}: {error}", source.as_str())),
        }
    }

    if parsed.is_none() {
        if strict_parse {
            let excluded = all_request_formats()
                .into_iter()
                .filter(|format| !candidates.contains(format))
                .collect::<Vec<_>>();
            let detectable_excluded =
                detect_formats_from_candidates(&excluded, path, headers, &plan.body);

            let mut message = if !detectable_excluded.is_empty() {
                let expected = expected_formats_label(from_cfg, &candidates);
                let detected = detectable_excluded
                    .iter()
                    .map(|format| format!("'{}'", format.as_str()))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Format mismatch: Request appears to be in format [{detected}], but only [{expected}] is allowed."
                )
            } else {
                let expected = expected_formats_label(from_cfg, &candidates);
                format!(
                    "unable to detect request format. Expected format: {expected}. Please verify your request body structure matches the expected format."
                )
            };

            if !parse_errors.is_empty() {
                message.push_str(&format!(" Parse errors: {}", parse_errors.join("; ")));
            }

            return Err(RequestProcessError::StrictParse(message));
        }
        return Ok(plan);
    }

    let (source, internal) = parsed.expect("checked is_some");
    let internal = if disable_tools {
        strip_tools(internal)
    } else {
        internal
    };
    let target = transform_cfg
        .get("to")
        .and_then(Value::as_str)
        .and_then(RequestFormat::from_name)
        .unwrap_or(source);
    plan.stream = internal.stream;
    plan.source_format = Some(source);
    plan.target_format = Some(target);

    if target != source || disable_tools {
        plan.body = emit_request(target, &internal)
            .map_err(|error| RequestProcessError::Transform(format!("Format transform error: {error}")))?;
        if target != source {
            plan.path = rewrite_path(path, target_path(target, &internal));
        }
    }

    Ok(plan)
}

fn detect_format(from_cfg: Option<&Value>, path: &str, headers: &[(String, String)], body: &Value) -> Option<RequestFormat> {
    detect_formats_from_candidates(&configured_candidates(from_cfg), path, headers, body)
        .into_iter()
        .next()
}

fn configured_candidates(from_cfg: Option<&Value>) -> Vec<RequestFormat> {
    if let Some(cfg) = from_cfg {
        match cfg {
            Value::String(name) if name != "auto" => RequestFormat::from_name(name).into_iter().collect(),
            Value::Array(values) => values
                .iter()
                .filter_map(Value::as_str)
                .filter_map(RequestFormat::from_name)
                .collect(),
            _ => default_detection_order(),
        }
    } else {
        default_detection_order()
    }
}

fn detect_formats_from_candidates(
    candidates: &[RequestFormat],
    path: &str,
    headers: &[(String, String)],
    body: &Value,
) -> Vec<RequestFormat> {
    let Some(body) = body.as_object() else {
        return Vec::new();
    };

    candidates
        .iter()
        .copied()
        .filter(|format| can_parse(*format, path, headers, body))
        .collect()
}

fn all_request_formats() -> Vec<RequestFormat> {
    vec![
        RequestFormat::OpenAiChat,
        RequestFormat::ClaudeChat,
        RequestFormat::OpenAiResponses,
        RequestFormat::GeminiChat,
    ]
}

fn expected_formats_label(from_cfg: Option<&Value>, candidates: &[RequestFormat]) -> String {
    match from_cfg {
        Some(Value::String(name)) if name != "auto" => format!("'{name}'"),
        _ => format!(
            "[{}]",
            candidates
                .iter()
                .map(|format| format!("'{}'", format.as_str()))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn default_detection_order() -> Vec<RequestFormat> {
    vec![
        RequestFormat::GeminiChat,
        RequestFormat::OpenAiChat,
        RequestFormat::ClaudeChat,
        RequestFormat::OpenAiResponses,
    ]
}

fn can_parse(format: RequestFormat, path: &str, headers: &[(String, String)], body: &Map<String, Value>) -> bool {
    match format {
        RequestFormat::GeminiChat => can_parse_gemini_chat(path, body),
        RequestFormat::OpenAiChat => can_parse_openai_chat(path, body),
        RequestFormat::ClaudeChat => can_parse_claude_chat(path, headers, body),
        RequestFormat::OpenAiResponses => can_parse_openai_responses(path, body),
    }
}

fn can_parse_openai_chat(path: &str, body: &Map<String, Value>) -> bool {
    if let Some(Value::Array(contents)) = body.get("contents") {
        if contents
            .first()
            .and_then(Value::as_object)
            .and_then(|item| item.get("parts"))
            .is_some()
        {
            return false;
        }
    }
    if body.contains_key("prompt") && !body.contains_key("messages") {
        return false;
    }
    if body.contains_key("system") || body.contains_key("anthropic_version") {
        return false;
    }
    if body.get("thinking").and_then(Value::as_object).is_some() {
        return false;
    }
    if let Some(Value::Array(tools)) = body.get("tools") {
        if tools.iter().any(|tool| tool.get("input_schema").is_some()) {
            return false;
        }
    }
    if path.contains("/chat/completions") {
        return true;
    }
    body.get("messages")
        .and_then(Value::as_array)
        .and_then(|messages| messages.first())
        .and_then(Value::as_object)
        .map(|msg| msg.contains_key("role"))
        .unwrap_or(false)
}

fn can_parse_claude_chat(path: &str, headers: &[(String, String)], body: &Map<String, Value>) -> bool {
    if let Some(Value::Array(contents)) = body.get("contents") {
        if contents
            .first()
            .and_then(Value::as_object)
            .and_then(|item| item.get("parts"))
            .is_some()
        {
            return false;
        }
    }
    if let Some(Value::Array(messages)) = body.get("messages") {
        for msg in messages {
            if let Some(msg) = msg.as_object() {
                if msg.get("role").and_then(Value::as_str) == Some("tool") {
                    return false;
                }
                if msg
                    .get("content")
                    .and_then(Value::as_array)
                    .map(|parts| parts.iter().any(|part| part.get("type").and_then(Value::as_str) == Some("image_url")))
                    .unwrap_or(false)
                {
                    return false;
                }
            }
        }
    }
    if path.contains("/messages") {
        return true;
    }
    if headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case("anthropic-version"))
    {
        return true;
    }
    if body.contains_key("anthropic_version") {
        return true;
    }
    if matches!(body.get("messages"), Some(Value::Array(_)) | Some(Value::String(_))) {
        return true;
    }
    body.get("prompt").and_then(Value::as_str).is_some()
}

fn can_parse_openai_responses(path: &str, body: &Map<String, Value>) -> bool {
    if path.contains("/responses") {
        return true;
    }
    if body.contains_key("input") && body.contains_key("model") {
        return true;
    }
    body.get("object").and_then(Value::as_str) == Some("response") && body.contains_key("output")
}

fn can_parse_gemini_chat(path: &str, body: &Map<String, Value>) -> bool {
    if path.contains("generativelanguage.googleapis.com")
        || path.contains("generateContent")
        || path.contains("streamGenerateContent")
        || path.contains("aistudio.google.com")
        || path.contains("/v1beta/models/")
    {
        return true;
    }
    let Some(Value::Array(contents)) = body.get("contents") else {
        return false;
    };
    let Some(first) = contents.first().and_then(Value::as_object) else {
        return false;
    };
    if !matches!(first.get("parts"), Some(Value::Array(_))) {
        return false;
    }
    match first.get("role").and_then(Value::as_str) {
        Some("model") => true,
        Some("user") => body.contains_key("generationConfig") || body.contains_key("safetySettings"),
        _ => false,
    }
}

fn parse_request(format: RequestFormat, body: &Value, path: &str) -> Result<InternalRequest> {
    let body = body.as_object().ok_or_else(|| anyhow!("request body must be an object"))?;
    match format {
        RequestFormat::OpenAiChat => parse_openai_chat(body),
        RequestFormat::ClaudeChat => parse_claude_chat(body),
        RequestFormat::OpenAiResponses => parse_openai_responses(body),
        RequestFormat::GeminiChat => parse_gemini_chat(body, path),
    }
}

fn emit_request(format: RequestFormat, req: &InternalRequest) -> Result<Value> {
    Ok(match format {
        RequestFormat::OpenAiChat => emit_openai_chat(req),
        RequestFormat::ClaudeChat => emit_claude_chat(req),
        RequestFormat::OpenAiResponses => emit_openai_responses(req),
        RequestFormat::GeminiChat => emit_gemini_chat(req),
    })
}

fn parse_openai_chat(body: &Map<String, Value>) -> Result<InternalRequest> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(parse_openai_chat_message)
        .collect::<Result<Vec<_>>>()?;
    let tools = parse_openai_tools(body.get("tools"));
    Ok(InternalRequest {
        messages,
        model: body.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools,
        tool_choice: body.get("tool_choice").cloned(),
        extra: filter_keys(body, &["messages", "model", "stream", "tools", "tool_choice"]),
    })
}

fn parse_openai_chat_message(msg: &Map<String, Value>) -> Result<InternalMessage> {
    let mut content = Vec::new();
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
    if role != "tool" {
        match msg.get("content") {
            Some(Value::String(text)) => {
                if !text.is_empty() {
                    content.push(InternalContentBlock::Text(text.clone()));
                }
            }
            Some(Value::Array(parts)) => {
                for part in parts.iter().filter_map(Value::as_object) {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            content.push(InternalContentBlock::Text(
                                part.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
                            ));
                        }
                        Some("image_url") => {
                            if let Some(image) = part.get("image_url").and_then(Value::as_object) {
                                if let Some(url) = image.get("url").and_then(Value::as_str) {
                                    content.push(InternalContentBlock::ImageUrl {
                                        url: url.to_string(),
                                        detail: image.get("detail").and_then(Value::as_str).map(ToString::to_string),
                                    });
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    if role == "tool" {
        content.push(InternalContentBlock::ToolResult {
            call_id: msg.get("tool_call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
            name: msg.get("name").and_then(Value::as_str).map(ToString::to_string),
            output: msg.get("content").cloned().unwrap_or(Value::String(String::new())),
        });
    }

    if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
        for tool_call in tool_calls.iter().filter_map(Value::as_object) {
            let function = tool_call.get("function").and_then(Value::as_object).cloned().unwrap_or_default();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str(raw).ok())
                .unwrap_or_else(|| function.get("arguments").cloned().unwrap_or_else(|| json!({})));
            content.push(InternalContentBlock::ToolCall {
                id: tool_call.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: function.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                arguments,
            });
        }
    }

    if content.is_empty() {
        content.push(InternalContentBlock::Text(String::new()));
    }

    Ok(InternalMessage {
        role: role.to_string(),
        content,
    })
}

fn parse_claude_chat(body: &Map<String, Value>) -> Result<InternalRequest> {
    if body.get("prompt").and_then(Value::as_str).is_some() && !body.contains_key("messages") {
        return parse_claude_code(body);
    }

    let mut messages = Vec::new();
    if let Some(system) = body.get("system") {
        let system_text = match system {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(Value::as_object)
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        if !system_text.is_empty() {
            messages.push(InternalMessage {
                role: "system".to_string(),
                content: vec![InternalContentBlock::Text(system_text)],
            });
        }
    }

    let raw_messages = match body.get("messages") {
        Some(Value::Array(messages)) => messages.clone(),
        Some(Value::String(raw)) => serde_json::from_str(raw).unwrap_or_default(),
        _ => Vec::new(),
    };
    for msg in raw_messages.iter().filter_map(Value::as_object) {
        let mut content = Vec::new();
        match msg.get("content") {
            Some(Value::String(text)) => content.push(InternalContentBlock::Text(text.clone())),
            Some(Value::Object(part)) => parse_claude_parts(std::slice::from_ref(part), &mut content),
            Some(Value::Array(parts)) => parse_claude_parts(
                &parts.iter().filter_map(Value::as_object).cloned().collect::<Vec<_>>(),
                &mut content,
            ),
            _ => {}
        }
        if content.is_empty() {
            content.push(InternalContentBlock::Text(String::new()));
        }
        let role = match msg.get("role").and_then(Value::as_str) {
            Some("user") => "user",
            _ => "assistant",
        };
        messages.push(InternalMessage {
            role: role.to_string(),
            content,
        });
    }

    Ok(InternalRequest {
        messages,
        model: body.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools: parse_claude_tools(body.get("tools")),
        tool_choice: body.get("tool_choice").cloned(),
        extra: filter_keys(body, &["system", "messages", "model", "stream", "tools", "tool_choice"]),
    })
}

fn parse_claude_parts(parts: &[Map<String, Value>], out: &mut Vec<InternalContentBlock>) {
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => out.push(InternalContentBlock::Text(
                part.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
            )),
            Some("thinking") => out.push(InternalContentBlock::Text(
                part.get("thinking").and_then(Value::as_str).unwrap_or_default().to_string(),
            )),
            Some("tool_use") => out.push(InternalContentBlock::ToolCall {
                id: part.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: part.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                arguments: part
                    .get("input")
                    .filter(|value| value.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            }),
            Some("tool_result") => {
                let output = match part.get("content") {
                    Some(Value::Array(items)) => Value::String(
                        items
                            .iter()
                            .filter_map(Value::as_object)
                            .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
                            .filter_map(|item| item.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("\n"),
                    ),
                    Some(value) => value.clone(),
                    None => Value::String(String::new()),
                };
                out.push(InternalContentBlock::ToolResult {
                    call_id: part.get("tool_use_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                    name: None,
                    output,
                });
            }
            _ => {}
        }
    }
}

fn parse_claude_code(body: &Map<String, Value>) -> Result<InternalRequest> {
    let mut messages = Vec::new();
    let options = body.get("options").and_then(Value::as_object).cloned().unwrap_or_default();
    if let Some(system_prompt) = options.get("systemPrompt").and_then(Value::as_str) {
        messages.push(InternalMessage {
            role: "system".to_string(),
            content: vec![InternalContentBlock::Text(system_prompt.to_string())],
        });
    }
    if let Some(prompt) = body.get("prompt").and_then(Value::as_str) {
        messages.push(InternalMessage {
            role: "user".to_string(),
            content: vec![InternalContentBlock::Text(prompt.to_string())],
        });
    }
    Ok(InternalRequest {
        messages,
        model: options.get("model").and_then(Value::as_str).unwrap_or("claude-sonnet-4-5").to_string(),
        stream: false,
        tools: Vec::new(),
        tool_choice: options.get("tool_choice").cloned(),
        extra: filter_keys(&options, &["model", "systemPrompt", "mcpServers", "tool_choice"]),
    })
}

fn parse_openai_responses(body: &Map<String, Value>) -> Result<InternalRequest> {
    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        if !instructions.trim().is_empty() {
            messages.push(InternalMessage {
                role: "system".to_string(),
                content: vec![InternalContentBlock::Text(instructions.trim().to_string())],
            });
        }
    }

    match body.get("input") {
        Some(Value::String(text)) => messages.push(InternalMessage {
            role: "user".to_string(),
            content: vec![InternalContentBlock::Text(text.clone())],
        }),
        Some(Value::Object(item)) => {
            if let Some(message) = parse_responses_input_item(&Value::Object(item.clone()))? {
                messages.push(message);
            }
        }
        Some(Value::Array(items)) => {
            for item in items {
                if let Some(message) = parse_responses_input_item(item)? {
                    messages.push(message);
                }
            }
        }
        _ => {}
    }

    let mut extra = filter_keys(
        body,
        &[
            "model",
            "instructions",
            "input",
            "tools",
            "tool_choice",
            "stream",
            "response_format",
            "metadata",
            "max_output_tokens",
            "temperature",
            "top_p",
            "reasoning",
            "user",
            "parallel_tool_calls",
            "store",
            "service_tier",
        ],
    );
    for key in [
        "response_format",
        "metadata",
        "max_output_tokens",
        "temperature",
        "top_p",
        "reasoning",
        "parallel_tool_calls",
        "store",
        "service_tier",
        "user",
    ] {
        if let Some(value) = body.get(key) {
            extra.entry(key.to_string()).or_insert_with(|| value.clone());
        }
    }

    Ok(InternalRequest {
        messages,
        model: body.get("model").and_then(Value::as_str).unwrap_or_default().to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools: parse_responses_tools(body.get("tools")),
        tool_choice: body.get("tool_choice").cloned(),
        extra,
    })
}

fn parse_responses_input_item(item: &Value) -> Result<Option<InternalMessage>> {
    let Some(item) = item.as_object() else {
        return Ok(None);
    };
    let item_type = item.get("type").and_then(Value::as_str).unwrap_or("message");
    if item.get("type").and_then(Value::as_str) == Some("reasoning") {
        let text = item
            .get("summary")
            .and_then(Value::as_array)
            .map(|summary| {
                summary
                    .iter()
                    .filter_map(Value::as_object)
                    .filter_map(|entry| entry.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        return Ok(Some(InternalMessage {
            role: "assistant".to_string(),
            content: vec![InternalContentBlock::Text(text)],
        }));
    }
    let content = match item_type {
        "message" | "input_text" | "output_text" | "" => {
            responses_content_blocks(item.get("content"), item.get("text").and_then(Value::as_str))
        }
        "function_call_output" | "tool_result" => {
            vec![InternalContentBlock::ToolResult {
                call_id: item.get("call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: item.get("name").and_then(Value::as_str).map(ToString::to_string),
                output: Value::String(responses_output_text(
                    item.get("output"),
                    item.get("text").and_then(Value::as_str),
                )),
            }]
        }
        _ => responses_content_blocks(None, None)
            .into_iter()
            .chain({
                let mut content = Vec::new();
                push_responses_content_block(&mut content, item);
                content
            })
            .collect(),
    };
    if content.is_empty() {
        return Ok(None);
    }
    let role = item
        .get("role")
        .and_then(Value::as_str)
        .unwrap_or_else(|| item.get("type").and_then(Value::as_str).unwrap_or("user"));
    Ok(Some(InternalMessage {
        role: match role {
            "assistant" | "model" => "assistant",
            "function_call" => "assistant",
            "system" => "system",
            "function_call_output" | "tool" => "tool",
            _ => "user",
        }
        .to_string(),
        content,
    }))
}

fn responses_content_blocks(content: Option<&Value>, default_text: Option<&str>) -> Vec<InternalContentBlock> {
    let mut blocks = Vec::new();
    match content {
        Some(Value::String(text)) => blocks.push(InternalContentBlock::Text(text.clone())),
        Some(Value::Object(content_obj)) => {
            if let Some(Value::Array(parts)) = content_obj.get("items") {
                for part in parts.iter().filter_map(Value::as_object) {
                    push_responses_content_block(&mut blocks, part);
                }
            } else {
                push_responses_content_block(&mut blocks, content_obj);
            }
        }
        Some(Value::Array(parts)) => {
            for part in parts.iter().filter_map(Value::as_object) {
                push_responses_content_block(&mut blocks, part);
            }
        }
        _ => {}
    }
    if blocks.is_empty() {
        if let Some(text) = default_text {
            blocks.push(InternalContentBlock::Text(text.to_string()));
        }
    }
    blocks
}

fn responses_output_text(output: Option<&Value>, default_text: Option<&str>) -> String {
    let blocks = responses_content_blocks(output, default_text);
    blocks
        .into_iter()
        .filter_map(|block| match block {
            InternalContentBlock::Text(text) => Some(text),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn push_responses_content_block(content: &mut Vec<InternalContentBlock>, part: &Map<String, Value>) {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") | Some("text") => {
            content.push(InternalContentBlock::Text(
                part.get("text").and_then(Value::as_str).unwrap_or_default().to_string(),
            ));
        }
        Some("input_image") | Some("image_url") => {
            let image_url = part.get("image_url");
            let url = image_url
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| part.get("url").and_then(Value::as_str).map(ToString::to_string))
                .or_else(|| {
                    image_url
                        .and_then(Value::as_object)
                        .and_then(|image| image.get("url"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
                .or_else(|| {
                    part.get("image")
                        .and_then(Value::as_object)
                        .and_then(|image| image.get("url"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                });
            let detail = part
                .get("detail")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| {
                    image_url
                        .and_then(Value::as_object)
                        .and_then(|image| image.get("detail"))
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                });
            if let Some(url) = url {
                content.push(InternalContentBlock::ImageUrl { url, detail });
            }
        }
        Some("function_call") => {
            let arguments = part
                .get("arguments")
                .map(|raw| match raw {
                    Value::String(text) => serde_json::from_str(text).unwrap_or_else(|_| json!({})),
                    Value::Object(_) | Value::Array(_) => raw.clone(),
                    _ => json!({}),
                })
                .unwrap_or_else(|| json!({}));
            content.push(InternalContentBlock::ToolCall {
                id: part.get("call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: part.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                arguments,
            });
        }
        Some("function_call_output") => {
            let output = match part.get("output") {
                Some(Value::String(text)) => Value::String(text.clone()),
                Some(Value::Array(items)) => Value::String(
                    items
                        .iter()
                        .filter_map(Value::as_object)
                        .filter(|item| {
                            matches!(
                                item.get("type").and_then(Value::as_str),
                                Some("input_text") | Some("output_text") | Some("text")
                            )
                        })
                        .filter_map(|item| item.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n"),
                ),
                Some(Value::Object(obj)) => {
                    if let Some(Value::Array(items)) = obj.get("items") {
                        Value::String(
                            items
                                .iter()
                                .filter_map(Value::as_object)
                                .filter(|item| {
                                    matches!(
                                        item.get("type").and_then(Value::as_str),
                                        Some("input_text") | Some("output_text") | Some("text")
                                    )
                                })
                                .filter_map(|item| item.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n"),
                        )
                    } else {
                        Value::String(String::new())
                    }
                }
                _ => Value::String(String::new()),
            };
            content.push(InternalContentBlock::ToolResult {
                call_id: part.get("call_id").and_then(Value::as_str).unwrap_or_default().to_string(),
                name: part.get("name").and_then(Value::as_str).map(ToString::to_string),
                output,
            });
        }
        _ => {}
    }
}

fn parse_gemini_chat(body: &Map<String, Value>, path: &str) -> Result<InternalRequest> {
    let mut messages = Vec::new();
    if let Some(system_text) = parse_gemini_system_instruction(body.get("systemInstruction")) {
        messages.push(InternalMessage {
            role: "system".to_string(),
            content: vec![InternalContentBlock::Text(system_text)],
        });
    }
    if let Some(Value::Array(contents)) = body.get("contents") {
        for content in contents.iter().filter_map(Value::as_object) {
            let role = match content.get("role").and_then(Value::as_str) {
                Some("model") => "assistant",
                Some("user") => "user",
                _ => "user",
            };
            let mut blocks = Vec::new();
            if let Some(Value::Array(parts)) = content.get("parts") {
                for part in parts.iter().filter_map(Value::as_object) {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        blocks.push(InternalContentBlock::Text(text.to_string()));
                    } else if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                        blocks.push(InternalContentBlock::ToolCall {
                            id: function_call.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                            name: function_call.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                            arguments: function_call.get("args").cloned().unwrap_or_else(|| json!({})),
                        });
                    } else if let Some(function_response) = part.get("functionResponse").and_then(Value::as_object) {
                        blocks.push(InternalContentBlock::ToolResult {
                            call_id: function_response.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
                            name: function_response.get("name").and_then(Value::as_str).map(ToString::to_string),
                            output: function_response.get("response").cloned().unwrap_or_else(|| json!({})),
                        });
                    }
                }
            }
            if blocks.is_empty() {
                blocks.push(InternalContentBlock::Text(String::new()));
            }
            messages.push(InternalMessage {
                role: role.to_string(),
                content: blocks,
            });
        }
    }
    let mut extra = filter_keys(
        body,
        &[
            "contents",
            "model",
            "tools",
            "toolConfig",
            "generationConfig",
            "safetySettings",
            "systemInstruction",
        ],
    );
    if let Some(generation_config) = body.get("generationConfig") {
        extra.insert("generationConfig".to_string(), generation_config.clone());
    }
    if let Some(safety_settings) = body.get("safetySettings") {
        extra.insert("safetySettings".to_string(), safety_settings.clone());
    }
    Ok(InternalRequest {
        messages,
        model: body.get("model").and_then(Value::as_str).unwrap_or("gemini-2.5-flash").to_string(),
        stream: path.contains("streamGenerateContent"),
        tools: parse_gemini_tools(body.get("tools")),
        tool_choice: body.get("toolConfig").cloned(),
        extra,
    })
}

fn emit_openai_chat(req: &InternalRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));
    let mut messages = Vec::new();
    for message in &req.messages {
        if message.role != "tool" {
            messages.push(openai_chat_message(message));
        }
        for block in &message.content {
            if let InternalContentBlock::ToolResult { call_id, name, output } = block {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": match output {
                        Value::Object(_) | Value::Array(_) => {
                            Value::String(serde_json::to_string(output).unwrap_or_else(|_| String::new()))
                        }
                        Value::String(text) => Value::String(text.clone()),
                        other => Value::String(other.to_string()),
                    }
                }));
            }
        }
    }
    body.insert("messages".to_string(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(req.tools.iter().map(openai_tool).collect()));
    }
    if let Some(tool_choice) = &req.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice.clone());
    }
    body.extend(req.extra.clone());
    Value::Object(body)
}

fn emit_claude_chat(req: &InternalRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));

    let mut messages = Vec::new();
    let mut system_text = Vec::new();
    for message in &req.messages {
        if message.role == "system" {
            for block in &message.content {
                if let InternalContentBlock::Text(text) = block {
                    if !text.is_empty() {
                        system_text.push(text.clone());
                    }
                }
            }
            continue;
        }
        messages.push(claude_message(message));
    }

    if !system_text.is_empty() {
        body.insert("system".to_string(), Value::String(system_text.join("\n")));
    }
    body.insert("messages".to_string(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(req.tools.iter().map(claude_tool).collect()));
    }
    if let Some(tool_choice) = &req.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice.clone());
    }
    body.extend(req.extra.clone());
    Value::Object(body)
}

fn emit_openai_responses(req: &InternalRequest) -> Value {
    let mut body = normalize_extra_for_openai_responses(&req.extra);
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));

    let mut input = Vec::new();
    let mut instructions = Vec::new();
    for message in &req.messages {
        if message.role == "system" {
            for block in &message.content {
                if let InternalContentBlock::Text(text) = block {
                    instructions.push(text.clone());
                }
            }
            continue;
        }
        input.push(responses_message(message));
    }
    if !instructions.is_empty() {
        body.insert("instructions".to_string(), Value::String(instructions.join("\n\n")));
    }
    if input.is_empty() {
        body.insert("input".to_string(), Value::String(String::new()));
    } else {
        body.insert("input".to_string(), Value::Array(input));
    }
    if !req.tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(req.tools.iter().map(responses_tool).collect()));
    }
    if let Some(tool_choice) = &req.tool_choice {
        body.insert(
            "tool_choice".to_string(),
            normalize_tool_choice_for_openai_responses(tool_choice),
        );
    }
    Value::Object(body)
}

fn emit_gemini_chat(req: &InternalRequest) -> Value {
    let mut body = Map::new();
    body.insert("model".to_string(), Value::String(req.model.clone()));
    let contents = req
        .messages
        .iter()
        .filter(|message| message.role != "system")
        .filter_map(gemini_message)
        .collect::<Vec<_>>();
    body.insert("contents".to_string(), Value::Array(contents));
    if !req.tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(vec![json!({
            "functionDeclarations": req.tools.iter().map(gemini_tool_decl).collect::<Vec<_>>()
        })]));
    }
    if let Some(tool_choice) = &req.tool_choice {
        body.insert("toolConfig".to_string(), tool_choice.clone());
    }
    body.extend(req.extra.clone());
    Value::Object(body)
}

fn target_path(format: RequestFormat, req: &InternalRequest) -> String {
    match format {
        RequestFormat::OpenAiChat => "/v1/chat/completions".to_string(),
        RequestFormat::ClaudeChat => "/v1/messages".to_string(),
        RequestFormat::OpenAiResponses => "/v1/responses".to_string(),
        RequestFormat::GeminiChat => {
            let model = if req.model.is_empty() {
                "gemini-2.5-flash"
            } else {
                req.model.as_str()
            };
            if req.stream {
                format!("/v1beta/models/{model}:streamGenerateContent")
            } else {
                format!("/v1beta/models/{model}:generateContent")
            }
        }
    }
}

fn rewrite_path(original_path: &str, transformed_path: String) -> String {
    for pattern in ["/v1/chat/completions", "/v1/messages", "/v1/responses"] {
        if let Some(idx) = original_path.find(pattern) {
            return format!("{}{}", &original_path[..idx], transformed_path);
        }
    }
    if let Some(idx) = original_path.find("/v1beta/models/") {
        return format!("{}{}", &original_path[..idx], transformed_path);
    }
    transformed_path
}

fn filter_keys(body: &Map<String, Value>, excluded: &[&str]) -> Map<String, Value> {
    body.iter()
        .filter(|(key, _)| !excluded.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

fn strip_tools(req: InternalRequest) -> InternalRequest {
    let messages = req
        .messages
        .into_iter()
        .filter_map(|message| {
            if message.role == "tool" {
                return None;
            }

            let mut content = message
                .content
                .into_iter()
                .filter(|block| {
                    !matches!(
                        block,
                        InternalContentBlock::ToolCall { .. } | InternalContentBlock::ToolResult { .. }
                    )
                })
                .collect::<Vec<_>>();

            if content.is_empty() {
                content.push(InternalContentBlock::Text(String::new()));
            }

            Some(InternalMessage {
                role: message.role,
                content,
            })
        })
        .collect();

    InternalRequest {
        messages,
        model: req.model,
        stream: req.stream,
        tools: Vec::new(),
        tool_choice: None,
        extra: req.extra,
    }
}

fn parse_openai_tools(value: Option<&Value>) -> Vec<InternalTool> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(|tool| {
            let function = tool.get("function")?.as_object()?;
            Some(InternalTool {
                name: function.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                description: function.get("description").and_then(Value::as_str).map(ToString::to_string),
                input_schema: function.get("parameters").cloned().unwrap_or_else(|| json!({})),
            })
        })
        .collect()
}

fn parse_claude_tools(value: Option<&Value>) -> Vec<InternalTool> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter_map(|tool| {
            let function = tool.get("function").and_then(Value::as_object);
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| function.and_then(|func| func.get("name").and_then(Value::as_str)))?;
            Some(InternalTool {
                name: name.to_string(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .or_else(|| function.and_then(|func| func.get("description").and_then(Value::as_str)).map(ToString::to_string)),
                input_schema: tool
                    .get("input_schema")
                    .cloned()
                    .or_else(|| function.and_then(|func| func.get("parameters").cloned()))
                    .unwrap_or_else(|| json!({})),
            })
        })
        .collect()
}

fn parse_responses_tools(value: Option<&Value>) -> Vec<InternalTool> {
    value
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("function"))
        .map(|tool| {
            let function = tool.get("function").and_then(Value::as_object);
            InternalTool {
                name: tool
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| function.and_then(|func| func.get("name").and_then(Value::as_str)))
                    .unwrap_or_default()
                    .to_string(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .or_else(|| function.and_then(|func| func.get("description").and_then(Value::as_str)).map(ToString::to_string)),
                input_schema: tool
                    .get("parameters")
                    .cloned()
                    .or_else(|| function.and_then(|func| func.get("parameters").cloned()))
                    .unwrap_or_else(|| json!({})),
            }
        })
        .collect()
}

fn parse_gemini_tools(value: Option<&Value>) -> Vec<InternalTool> {
    let mut tools = Vec::new();
    if let Some(Value::Array(tool_sets)) = value {
        for tool_set in tool_sets.iter().filter_map(Value::as_object) {
            if let Some(Value::Array(decls)) = tool_set.get("functionDeclarations") {
                for decl in decls.iter().filter_map(Value::as_object) {
                    tools.push(InternalTool {
                        name: decl.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                        description: decl.get("description").and_then(Value::as_str).map(ToString::to_string),
                        input_schema: decl.get("parameters").cloned().unwrap_or_else(|| json!({})),
                    });
                }
            }
        }
    }
    tools
}

fn openai_chat_message(message: &InternalMessage) -> Value {
    let mut msg = Map::new();
    msg.insert("role".to_string(), Value::String(message.role.clone()));
    let mut text_parts = Vec::new();
    let mut rich_parts = Vec::new();
    let mut tool_calls = Vec::new();
    for block in &message.content {
        match block {
            InternalContentBlock::Text(text) => {
                text_parts.push(text.clone());
                rich_parts.push(json!({"type":"text","text": text}));
            }
            InternalContentBlock::ImageUrl { url, detail } => {
                let mut image = Map::new();
                image.insert("url".to_string(), Value::String(url.clone()));
                if let Some(detail) = detail {
                    image.insert("detail".to_string(), Value::String(detail.clone()));
                }
                rich_parts.push(Value::Object(Map::from_iter([
                    ("type".to_string(), Value::String("image_url".to_string())),
                    ("image_url".to_string(), Value::Object(image)),
                ])));
            }
            InternalContentBlock::ToolCall { id, name, arguments } => {
                tool_calls.push(json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())
                    }
                }));
            }
            InternalContentBlock::ToolResult { .. } => {}
        }
    }
    if msg.get("content").is_none() {
        if rich_parts.iter().any(|part| part.get("type").and_then(Value::as_str) == Some("image_url")) {
            msg.insert("content".to_string(), Value::Array(rich_parts));
        } else {
            msg.insert("content".to_string(), Value::String(text_parts.join("\n")));
        }
    }
    if !tool_calls.is_empty() {
        msg.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    Value::Object(msg)
}

fn claude_message(message: &InternalMessage) -> Value {
    let parts = message
        .content
        .iter()
        .filter_map(|block| match block {
            InternalContentBlock::Text(text) => Some(json!({"type":"text","text": text})),
            InternalContentBlock::ToolCall { id, name, arguments } => Some(json!({
                "type":"tool_use",
                "id": id,
                "name": name,
                "input": arguments
            })),
            InternalContentBlock::ToolResult { call_id, output, .. } => Some(json!({
                "type":"tool_result",
                "tool_use_id": call_id,
                "content": match output {
                    Value::String(text) => Value::Array(vec![json!({"type":"text","text": text})]),
                    Value::Object(_) => Value::Array(vec![json!({
                        "type":"text",
                        "text": serde_json::to_string(output).unwrap_or_else(|_| String::new())
                    })]),
                    Value::Array(items) => Value::Array(items.clone()),
                    other => Value::Array(vec![json!({"type":"text","text": other.to_string()})]),
                }
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    json!({
        "role": if message.role == "assistant" { "assistant" } else { "user" },
        "content": parts
    })
}

fn responses_message(message: &InternalMessage) -> Value {
    let mut content = Vec::new();
    for block in &message.content {
        match block {
            InternalContentBlock::Text(text) => {
                content.push(json!({"type":"input_text","text": text}));
            }
            InternalContentBlock::ImageUrl { url, detail } => {
                let mut part = Map::new();
                part.insert("type".to_string(), Value::String("input_image".to_string()));
                part.insert("image_url".to_string(), Value::String(url.clone()));
                if let Some(detail) = detail {
                    part.insert("detail".to_string(), Value::String(detail.clone()));
                }
                content.push(Value::Object(part));
            }
            InternalContentBlock::ToolCall { id, name, arguments } => {
                return json!({
                    "type":"function_call",
                    "id": id,
                    "call_id": id,
                    "name": name,
                    "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
                    "status": "completed"
                });
            }
            InternalContentBlock::ToolResult { call_id, name, output } => {
                return json!({
                    "type":"function_call_output",
                    "call_id": call_id,
                    "name": name,
                    "output": output,
                    "status": "completed"
                });
            }
        }
    }
    json!({
        "type": "message",
        "role": if message.role == "assistant" { "assistant" } else { "user" },
        "content": content
    })
}

fn gemini_message(message: &InternalMessage) -> Option<Value> {
    let parts = message
        .content
        .iter()
        .filter_map(|block| match block {
            InternalContentBlock::Text(text) if !text.is_empty() => Some(json!({"text": text})),
            InternalContentBlock::ToolCall { id, name, arguments } => Some(json!({
                "functionCall": {
                    "id": id,
                    "name": name,
                    "args": arguments
                }
            })),
            InternalContentBlock::ToolResult { call_id, name, output } => Some(json!({
                "functionResponse": {
                    "id": call_id,
                    "name": name,
                    "response": output
                }
            })),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        return None;
    }
    Some(json!({
        "role": if message.role == "assistant" { "model" } else { "user" },
        "parts": parts
    }))
}

fn openai_tool(tool: &InternalTool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    })
}

fn claude_tool(tool: &InternalTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": normalize_claude_input_schema(&tool.input_schema)
    })
}

fn responses_tool(tool: &InternalTool) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema
    })
}

fn gemini_tool_decl(tool: &InternalTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema
    })
}

fn parse_gemini_system_instruction(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
        Some(Value::Object(instruction)) => instruction
            .get("parts")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .filter_map(Value::as_object)
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .filter(|text| !text.is_empty()),
        _ => None,
    }
}

fn normalize_claude_input_schema(schema: &Value) -> Value {
    let mut schema = match schema {
        Value::Object(map) if !map.is_empty() => map.clone(),
        _ => Map::new(),
    };

    if schema.get("type").and_then(Value::as_str) != Some("object") {
        schema.insert("type".to_string(), Value::String("object".to_string()));
    }
    if !matches!(schema.get("properties"), Some(Value::Object(_))) {
        schema.insert("properties".to_string(), Value::Object(Map::new()));
    }
    if matches!(schema.get("required"), Some(value) if !value.is_array()) {
        schema.remove("required");
    }

    Value::Object(schema)
}

fn normalize_tool_choice_for_openai_responses(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(_) => tool_choice.clone(),
        Value::Object(choice) => match choice.get("type").and_then(Value::as_str) {
            Some("function") => {
                if let Some(name) = choice
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        choice
                            .get("function")
                            .and_then(Value::as_object)
                            .and_then(|function| function.get("name").and_then(Value::as_str))
                    })
                {
                    json!({"type": "function", "name": name})
                } else {
                    tool_choice.clone()
                }
            }
            Some("tool") => choice
                .get("name")
                .and_then(Value::as_str)
                .map(|name| json!({"type": "function", "name": name}))
                .unwrap_or_else(|| tool_choice.clone()),
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

fn normalize_extra_for_openai_responses(extra: &Map<String, Value>) -> Map<String, Value> {
    if extra.is_empty() {
        return Map::new();
    }

    let mut out = extra.clone();

    if !out.contains_key("max_output_tokens") {
        let maybe_max = out
            .get("max_tokens")
            .and_then(Value::as_i64)
            .or_else(|| out.get("max_completion_tokens").and_then(Value::as_i64));
        if let Some(max_output_tokens) = maybe_max {
            out.insert("max_output_tokens".to_string(), Value::Number(max_output_tokens.into()));
        }
    }
    out.remove("max_tokens");
    out.remove("max_completion_tokens");

    let response_format = out.remove("response_format");
    if let Some(response_format) = response_format {
        if !out.contains_key("text") {
            if let Some(text) = normalize_responses_text_format(&response_format) {
                out.insert("text".to_string(), text);
            }
        }
    }

    if let Some(Value::Object(stream_options)) = out.get_mut("stream_options") {
        stream_options.remove("include_usage");
        if stream_options.is_empty() {
            out.remove("stream_options");
        }
    }

    for key in [
        "messages",
        "n",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "logit_bias",
        "logprobs",
    ] {
        out.remove(key);
    }

    out
}

fn normalize_responses_text_format(response_format: &Value) -> Option<Value> {
    let response_format = response_format.as_object()?;
    let format = match response_format.get("type").and_then(Value::as_str) {
        Some("json_schema") => {
            let schema = response_format
                .get("json_schema")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let mut format = Map::new();
            format.insert("type".to_string(), Value::String("json_schema".to_string()));
            format.insert(
                "name".to_string(),
                Value::String(
                    schema
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("response")
                        .to_string(),
                ),
            );
            format.insert(
                "schema".to_string(),
                schema.get("schema").cloned().unwrap_or_else(|| json!({})),
            );
            if let Some(description) = schema.get("description") {
                format.insert("description".to_string(), description.clone());
            }
            if let Some(strict) = schema.get("strict") {
                format.insert("strict".to_string(), strict.clone());
            }
            Value::Object(format)
        }
        Some("json_object") | Some("text") => {
            json!({"type": response_format.get("type").and_then(Value::as_str).unwrap_or_default()})
        }
        _ => return None,
    };

    Some(json!({"format": format}))
}
