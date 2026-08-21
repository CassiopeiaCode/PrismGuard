use anyhow::{anyhow, Result};
use serde_json::{json, Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestFormat {
    OpenAiChat,
    ClaudeChat,
    OpenAiResponses,
    GeminiChat,
}

const DEFAULT_CLAUDE_MAX_TOKENS: u64 = 4096;

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
    pub passthrough: bool,
    pub moderation_text: Option<String>,
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
    thinking: Option<Value>,
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
    File {
        file_id: Option<String>,
        url: Option<String>,
        data: Option<String>,
        media_type: Option<String>,
        filename: Option<String>,
    },
    Audio {
        data: String,
        format: String,
    },
}

#[derive(Debug, Clone)]
struct InternalTool {
    name: String,
    description: Option<String>,
    input_schema: Value,
    strict: Option<bool>,
}

#[derive(Debug, Clone)]
struct ClaudeThinkingPlan {
    thinking: Value,
    output_config: Option<Value>,
    enabled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NormalizedToolChoice {
    Auto,
    None,
    Required,
    Tool(String),
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
        passthrough: false,
        moderation_text: None,
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

            let diagnostics = request_format_diagnostics(&candidates, path, headers, &plan.body);
            if !diagnostics.is_empty() {
                message.push_str(&format!(" Diagnostics: {}", diagnostics.join("; ")));
            } else {
                message.push_str(&format!(
                    " Diagnostics: none of the required structural checks for {} matched the request body; JSON root keys: {}",
                    expected_formats_label(from_cfg, &candidates),
                    plan.body
                        .as_object()
                        .map(|object| {
                            object.keys().map(String::as_str).collect::<Vec<_>>().join(", ")
                        })
                        .unwrap_or_else(|| format!("<{}>", json_type_name(&plan.body)))
                ));
            }

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
    let passthrough = transform_cfg
        .get("to")
        .and_then(Value::as_str)
        .is_some_and(|target| target == "pass_through");
    let target = if passthrough {
        source
    } else {
        transform_cfg
            .get("to")
            .and_then(Value::as_str)
            .and_then(RequestFormat::from_name)
            .unwrap_or(source)
    };
    plan.stream = internal.stream;
    plan.source_format = Some(source);
    plan.target_format = Some(target);
    plan.passthrough = passthrough;
    plan.moderation_text = Some(moderation_text_from_internal_request(&internal));

    if !passthrough {
        plan.body = emit_request(target, &internal).map_err(|error| {
            RequestProcessError::Transform(format!("Format transform error: {error}"))
        })?;
        if target != source {
            plan.path = rewrite_path(path, target_path(target, &internal));
        }
    }

    Ok(plan)
}

fn moderation_text_from_internal_request(req: &InternalRequest) -> String {
    let mut texts = Vec::new();
    for message in &req.messages {
        for block in &message.content {
            match block {
                InternalContentBlock::Text(text) => push_non_empty_text(text, &mut texts),
                InternalContentBlock::ToolResult { output, .. } => {
                    collect_moderation_value_text(output, &mut texts);
                }
                InternalContentBlock::ToolCall { .. }
                | InternalContentBlock::ImageUrl { .. }
                | InternalContentBlock::File { .. }
                | InternalContentBlock::Audio { .. } => {}
            }
        }
    }
    texts.join("\n")
}

fn collect_moderation_value_text(value: &Value, texts: &mut Vec<String>) {
    match value {
        Value::String(text) => push_non_empty_text(text, texts),
        Value::Array(items) => {
            for item in items {
                collect_moderation_value_text(item, texts);
            }
        }
        _ => {}
    }
}

fn push_non_empty_text(text: &str, texts: &mut Vec<String>) {
    if !text.is_empty() {
        texts.push(text.to_string());
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn detect_format(
    from_cfg: Option<&Value>,
    path: &str,
    headers: &[(String, String)],
    body: &Value,
) -> Option<RequestFormat> {
    detect_formats_from_candidates(&configured_candidates(from_cfg), path, headers, body)
        .into_iter()
        .next()
}

fn configured_candidates(from_cfg: Option<&Value>) -> Vec<RequestFormat> {
    if let Some(cfg) = from_cfg {
        match cfg {
            Value::String(name) if name != "auto" => {
                RequestFormat::from_name(name).into_iter().collect()
            }
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

fn request_format_diagnostics(
    candidates: &[RequestFormat],
    path: &str,
    headers: &[(String, String)],
    body: &Value,
) -> Vec<String> {
    candidates
        .iter()
        .flat_map(|format| format_diagnostics(*format, path, headers, body))
        .collect()
}

fn format_diagnostics(
    format: RequestFormat,
    path: &str,
    headers: &[(String, String)],
    body: &Value,
) -> Vec<String> {
    let Some(object) = body.as_object() else {
        return vec![format!(
            "format '{}': JSON path '$' must contain an object, got {}",
            format.as_str(),
            json_type_name(body)
        )];
    };

    match format {
        RequestFormat::OpenAiChat => openai_chat_diagnostics(path, headers, object),
        RequestFormat::ClaudeChat => claude_chat_diagnostics(path, headers, object),
        RequestFormat::OpenAiResponses => openai_responses_diagnostics(path, object),
        RequestFormat::GeminiChat => gemini_chat_diagnostics(path, object),
    }
}

fn openai_chat_diagnostics(
    path: &str,
    headers: &[(String, String)],
    body: &Map<String, Value>,
) -> Vec<String> {
    let mut diagnostics = Vec::new();
    let anthropic_header = headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case("anthropic-version"));

    if path.contains("/messages") || anthropic_header || body.contains_key("anthropic_version") {
        diagnostics.push(
            "format 'openai_chat': request path or headers identify an Anthropic/Claude request"
                .to_string(),
        );
    }
    if body.contains_key("prompt") && !body.contains_key("messages") {
        diagnostics.push(
            "JSON path '$.prompt': found a prompt request, expected '$.messages' for openai_chat"
                .to_string(),
        );
    }
    if body.contains_key("system") {
        diagnostics.push(
            "JSON path '$.system': Claude-style system field is not valid for openai_chat detection"
                .to_string(),
        );
    }

    if let Some(Value::Array(contents)) = body.get("contents") {
        if contents.iter().any(|item| {
            item.as_object()
                .and_then(|item| item.get("parts"))
                .is_some()
        }) {
            diagnostics.push(
                "JSON path '$.contents': Gemini-style contents/parts detected; expected '$.messages' for openai_chat"
                    .to_string(),
            );
        }
    }

    match body.get("messages") {
        None => diagnostics.push(
            "JSON path '$.messages': missing required field; expected an array of chat messages"
                .to_string(),
        ),
        Some(Value::Array(messages)) if messages.is_empty() => diagnostics.push(
            "JSON path '$.messages': must contain at least one message".to_string(),
        ),
        Some(Value::Array(messages)) => {
            for (index, message) in messages.iter().enumerate() {
                let message_path = format!("$.messages[{index}]");
                let Some(message) = message.as_object() else {
                    diagnostics.push(format!(
                        "JSON path '{message_path}': expected an object, got {}",
                        json_type_name(message)
                    ));
                    continue;
                };
                match message.get("role") {
                    None => diagnostics.push(format!(
                        "JSON path '{message_path}.role': missing required field; expected a string"
                    )),
                    Some(value) if value.as_str().is_none() => diagnostics.push(format!(
                        "JSON path '{message_path}.role': expected a string, got {}",
                        json_type_name(value)
                    )),
                    _ => {}
                }
            }
        }
        Some(value) => diagnostics.push(format!(
            "JSON path '$.messages': expected an array, got {}",
            json_type_name(value)
        )),
    }

    if let Some(value) = body.get("stream") {
        if value.as_bool().is_none() {
            diagnostics.push(format!(
                "JSON path '$.stream': expected a boolean, got {}",
                json_type_name(value)
            ));
        }
    }
    if let Some(value) = body.get("model") {
        if value.as_str().is_none() {
            diagnostics.push(format!(
                "JSON path '$.model': expected a string, got {}",
                json_type_name(value)
            ));
        }
    }

    if let Some(Value::Array(messages)) = body.get("messages") {
        for (index, message) in messages.iter().enumerate() {
            let Some(message) = message.as_object() else {
                continue;
            };
            if let Some(Value::Array(content)) = message.get("content") {
                if content.iter().any(|part| {
                    part.as_object()
                        .is_some_and(|part| part.contains_key("cache_control"))
                }) {
                    diagnostics.push(format!(
                        "JSON path '$.messages[{index}].content': contains Claude cache_control blocks, not openai_chat content"
                    ));
                }
            }
        }
    }
    if body.get("thinking").and_then(Value::as_object).is_some()
        && !has_openai_chat_specific_signal(path, body)
    {
        diagnostics.push(
            "JSON path '$.thinking': Claude thinking configuration detected; expected OpenAI chat signals such as '$.reasoning_effort'"
                .to_string(),
        );
    }
    if let Some(Value::Array(tools)) = body.get("tools") {
        for (index, tool) in tools.iter().enumerate() {
            if tool.get("input_schema").is_some() {
                diagnostics.push(format!(
                    "JSON path '$.tools[{index}].input_schema': Claude-style tool schema detected; expected OpenAI '$.tools[{index}].function.parameters'"
                ));
            }
        }
    }

    diagnostics
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn claude_chat_diagnostics(
    path: &str,
    headers: &[(String, String)],
    body: &Map<String, Value>,
) -> Vec<String> {
    let mut diagnostics = Vec::new();
    let is_claude_endpoint = path.contains("/messages")
        || headers
            .iter()
            .any(|(key, _)| key.eq_ignore_ascii_case("anthropic-version"))
        || body.contains_key("anthropic_version");
    if !is_claude_endpoint && !body.contains_key("prompt") {
        diagnostics.push(
            "format 'claude_chat': expected an Anthropic messages endpoint, anthropic-version header, or '$.anthropic_version'"
                .to_string(),
        );
    }
    match body.get("messages") {
        Some(Value::Array(messages)) => {
            for (index, message) in messages.iter().enumerate() {
                let path = format!("$.messages[{index}]");
                let Some(message) = message.as_object() else {
                    diagnostics.push(format!(
                        "JSON path '{path}': expected an object, got {}",
                        json_type_name(message)
                    ));
                    continue;
                };
                if message.get("role").and_then(Value::as_str).is_none() {
                    diagnostics.push(format!(
                        "JSON path '{path}.role': missing or invalid; expected a string"
                    ));
                }
            }
        }
        Some(Value::String(_)) => {}
        Some(value) => diagnostics.push(format!(
            "JSON path '$.messages': expected an array or string, got {}",
            json_type_name(value)
        )),
        None if !body.contains_key("prompt") => diagnostics.push(
            "JSON path '$.messages': missing required field; expected an array of Claude messages"
                .to_string(),
        ),
        None => {}
    }
    if let Some(value) = body.get("max_tokens") {
        if value.as_u64().is_none() {
            diagnostics.push(format!(
                "JSON path '$.max_tokens': expected a non-negative integer, got {}",
                json_type_name(value)
            ));
        }
    }
    diagnostics
}

fn openai_responses_diagnostics(path: &str, body: &Map<String, Value>) -> Vec<String> {
    let mut diagnostics = Vec::new();
    if !path.contains("/responses") && !body.contains_key("input") {
        diagnostics.push(
            "format 'openai_responses': expected an '/responses' endpoint or JSON path '$.input'"
                .to_string(),
        );
    }
    if !path.contains("/responses") && body.contains_key("input") && !body.contains_key("model") {
        diagnostics.push(
            "JSON path '$.model': missing required field when using '$.input'; expected a string"
                .to_string(),
        );
    }
    if let Some(value) = body.get("model") {
        if value.as_str().is_none() {
            diagnostics.push(format!(
                "JSON path '$.model': expected a string, got {}",
                json_type_name(value)
            ));
        }
    }
    if let Some(value) = body.get("input") {
        if !matches!(value, Value::String(_) | Value::Array(_)) {
            diagnostics.push(format!(
                "JSON path '$.input': expected a string or array, got {}",
                json_type_name(value)
            ));
        }
    } else if !path.contains("/responses") {
        diagnostics.push(
            "JSON path '$.input': missing required field; expected a string or array".to_string(),
        );
    }
    diagnostics
}

fn gemini_chat_diagnostics(path: &str, body: &Map<String, Value>) -> Vec<String> {
    let mut diagnostics = Vec::new();
    let gemini_endpoint = path.contains("generateContent")
        || path.contains("streamGenerateContent")
        || path.contains("/v1beta/models/");
    if !gemini_endpoint && !body.contains_key("contents") {
        diagnostics.push(
            "format 'gemini_chat': expected a Gemini generateContent endpoint or JSON path '$.contents'"
                .to_string(),
        );
    }
    match body.get("contents") {
        Some(Value::Array(contents)) => {
            if contents.is_empty() {
                diagnostics.push("JSON path '$.contents': must contain at least one content item".to_string());
            }
            for (index, content) in contents.iter().enumerate() {
                let path = format!("$.contents[{index}]");
                let Some(content) = content.as_object() else {
                    diagnostics.push(format!(
                        "JSON path '{path}': expected an object, got {}",
                        json_type_name(content)
                    ));
                    continue;
                };
                match content.get("parts") {
                    Some(Value::Array(parts)) if !parts.is_empty() => {}
                    Some(Value::Array(_)) => diagnostics.push(format!(
                        "JSON path '{path}.parts': must contain at least one part"
                    )),
                    Some(value) => diagnostics.push(format!(
                        "JSON path '{path}.parts': expected an array, got {}",
                        json_type_name(value)
                    )),
                    None => diagnostics.push(format!(
                        "JSON path '{path}.parts': missing required field; expected an array"
                    )),
                }
            }
        }
        Some(value) => diagnostics.push(format!(
            "JSON path '$.contents': expected an array, got {}",
            json_type_name(value)
        )),
        None => diagnostics.push(
            "JSON path '$.contents': missing required field; expected an array of content items"
                .to_string(),
        ),
    }
    diagnostics
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

fn can_parse(
    format: RequestFormat,
    path: &str,
    headers: &[(String, String)],
    body: &Map<String, Value>,
) -> bool {
    match format {
        RequestFormat::GeminiChat => can_parse_gemini_chat(path, body),
        RequestFormat::OpenAiChat => can_parse_openai_chat(path, headers, body),
        RequestFormat::ClaudeChat => can_parse_claude_chat(path, headers, body),
        RequestFormat::OpenAiResponses => can_parse_openai_responses(path, body),
    }
}

fn can_parse_openai_chat(
    path: &str,
    headers: &[(String, String)],
    body: &Map<String, Value>,
) -> bool {
    if path.contains("/messages")
        || headers
            .iter()
            .any(|(key, _)| key.eq_ignore_ascii_case("anthropic-version"))
    {
        return false;
    }
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
    if !path.contains("/chat/completions") {
        if let Some(Value::Array(messages)) = body.get("messages") {
            for msg in messages.iter().filter_map(Value::as_object) {
                if let Some(Value::Array(content)) = msg.get("content") {
                    if content
                        .iter()
                        .filter_map(Value::as_object)
                        .any(|block| block.contains_key("cache_control"))
                    {
                        return false;
                    }
                }
            }
        }
    }
    if body.get("thinking").and_then(Value::as_object).is_some()
        && !has_openai_chat_specific_signal(path, body)
    {
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

fn has_openai_chat_specific_signal(path: &str, body: &Map<String, Value>) -> bool {
    if path.contains("/chat/completions") {
        return true;
    }
    if body.contains_key("max_completion_tokens")
        || body.contains_key("reasoning_effort")
        || body.contains_key("stream_options")
    {
        return true;
    }
    if body
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| {
            tools.iter().any(|tool| {
                tool.get("type").and_then(Value::as_str) == Some("function")
                    || tool.get("function").and_then(Value::as_object).is_some()
            })
        })
    {
        return true;
    }
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages
                .iter()
                .filter_map(Value::as_object)
                .any(|msg| msg.get("tool_calls").and_then(Value::as_array).is_some())
        })
}

fn can_parse_claude_chat(
    path: &str,
    headers: &[(String, String)],
    body: &Map<String, Value>,
) -> bool {
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
                    .map(|parts| {
                        parts.iter().any(|part| {
                            part.get("type").and_then(Value::as_str) == Some("image_url")
                        })
                    })
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
    if matches!(
        body.get("messages"),
        Some(Value::Array(_)) | Some(Value::String(_))
    ) {
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
        Some("user") => {
            body.contains_key("generationConfig") || body.contains_key("safetySettings")
        }
        _ => false,
    }
}

fn parse_request(format: RequestFormat, body: &Value, path: &str) -> Result<InternalRequest> {
    let body = body
        .as_object()
        .ok_or_else(|| anyhow!("request body must be an object"))?;
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
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools,
        tool_choice: body.get("tool_choice").cloned(),
        thinking: None,
        extra: filter_keys(
            body,
            &["messages", "model", "stream", "tools", "tool_choice"],
        ),
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
                                part.get("text")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default()
                                    .to_string(),
                            ));
                        }
                        Some("image_url") => {
                            if let Some(image) = part.get("image_url").and_then(Value::as_object) {
                                if let Some(url) = image.get("url").and_then(Value::as_str) {
                                    content.push(InternalContentBlock::ImageUrl {
                                        url: url.to_string(),
                                        detail: image
                                            .get("detail")
                                            .and_then(Value::as_str)
                                            .map(ToString::to_string),
                                    });
                                }
                            }
                        }
                        Some("file") => {
                            if let Some(file) = part.get("file").and_then(Value::as_object) {
                                let file_data = file
                                    .get("file_data")
                                    .and_then(Value::as_str)
                                    .filter(|value| !value.trim().is_empty())
                                    .map(parse_file_data);
                                let file_id = file
                                    .get("file_id")
                                    .and_then(Value::as_str)
                                    .filter(|value| !value.trim().is_empty())
                                    .map(ToString::to_string);
                                let file_url = file
                                    .get("file_url")
                                    .and_then(Value::as_str)
                                    .filter(|value| !value.trim().is_empty())
                                    .map(ToString::to_string);
                                if file_data.is_some() || file_id.is_some() || file_url.is_some() {
                                    let (data, media_type) = file_data
                                        .map(|(data, media_type)| (Some(data), media_type))
                                        .unwrap_or((None, None));
                                    content.push(InternalContentBlock::File {
                                        file_id,
                                        url: file_url,
                                        data,
                                        media_type,
                                        filename: file
                                            .get("filename")
                                            .and_then(Value::as_str)
                                            .map(ToString::to_string),
                                    });
                                }
                            }
                        }
                        Some("input_audio") => {
                            if let Some(audio) = part.get("input_audio").and_then(Value::as_object)
                            {
                                if let (Some(data), Some(format)) = (
                                    audio.get("data").and_then(Value::as_str),
                                    audio.get("format").and_then(Value::as_str),
                                ) {
                                    content.push(InternalContentBlock::Audio {
                                        data: data.to_string(),
                                        format: format.to_string(),
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
            call_id: msg
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            name: msg
                .get("name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            output: normalize_openai_tool_result(
                msg.get("content")
                    .cloned()
                    .unwrap_or(Value::String(String::new())),
            )?,
        });
    }

    if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
        for tool_call in tool_calls.iter().filter_map(Value::as_object) {
            let function = tool_call
                .get("function")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .and_then(|raw| serde_json::from_str(raw).ok())
                .unwrap_or_else(|| {
                    function
                        .get("arguments")
                        .cloned()
                        .unwrap_or_else(|| json!({}))
                });
            content.push(InternalContentBlock::ToolCall {
                id: tool_call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
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

fn normalize_openai_tool_result(output: Value) -> Result<Value> {
    let Value::Array(items) = output else {
        return Ok(output);
    };
    let mut normalized = Vec::with_capacity(items.len());
    for item in items {
        let item = item
            .as_object()
            .ok_or_else(|| anyhow!("OpenAI tool content blocks must be objects"))?;
        match item.get("type").and_then(Value::as_str) {
            Some("text") | Some("input_text") | Some("output_text") => normalized.push(json!({
                "type":"input_text",
                "text":item.get("text").and_then(Value::as_str).unwrap_or_default()
            })),
            Some("image_url") | Some("input_image") => {
                let image_url = item.get("image_url");
                let url = image_url
                    .and_then(Value::as_str)
                    .or_else(|| image_url.and_then(Value::as_object).and_then(|image| image.get("url")).and_then(Value::as_str))
                    .or_else(|| item.get("url").and_then(Value::as_str))
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| anyhow!("OpenAI tool image is missing image_url"))?;
                let detail = item
                    .get("detail")
                    .and_then(Value::as_str)
                    .or_else(|| image_url.and_then(Value::as_object).and_then(|image| image.get("detail")).and_then(Value::as_str))
                    .unwrap_or("auto");
                normalized.push(json!({"type":"input_image","detail":detail,"image_url":url}));
            }
            other => return Err(anyhow!("unsupported OpenAI tool content type: {other:?}")),
        }
    }
    Ok(Value::Array(normalized))
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
            Some(Value::Object(part)) => {
                parse_claude_parts(std::slice::from_ref(part), &mut content)?
            }
            Some(Value::Array(parts)) => parse_claude_parts(
                &parts
                    .iter()
                    .filter_map(Value::as_object)
                    .cloned()
                    .collect::<Vec<_>>(),
                &mut content,
            )?,
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
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools: parse_claude_tools(body.get("tools")),
        tool_choice: body.get("tool_choice").cloned(),
        thinking: body.get("thinking").cloned(),
        extra: filter_keys(
            body,
            &[
                "system",
                "messages",
                "model",
                "stream",
                "tools",
                "tool_choice",
                "thinking",
            ],
        ),
    })
}

fn parse_claude_parts(parts: &[Map<String, Value>], out: &mut Vec<InternalContentBlock>) -> Result<()> {
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => out.push(InternalContentBlock::Text(
                part.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )),
            Some("thinking") => out.push(InternalContentBlock::Text(
                part.get("thinking")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            )),
            Some("image") => {
                let source = part
                    .get("source")
                    .and_then(Value::as_object)
                    .ok_or_else(|| anyhow!("Claude image.source must be an object"))?;
                let url = match source.get("type").and_then(Value::as_str) {
                    Some("base64") => {
                        let media_type = source
                            .get("media_type")
                            .and_then(Value::as_str)
                            .filter(|value| !value.is_empty())
                            .ok_or_else(|| anyhow!("Claude base64 image is missing media_type"))?;
                        let data = source
                            .get("data")
                            .and_then(Value::as_str)
                            .filter(|value| !value.is_empty())
                            .ok_or_else(|| anyhow!("Claude base64 image is missing data"))?;
                        format!("data:{media_type};base64,{data}")
                    }
                    Some("url") => source
                        .get("url")
                        .and_then(Value::as_str)
                        .filter(|value| !value.trim().is_empty())
                        .map(ToString::to_string)
                        .ok_or_else(|| anyhow!("Claude URL image is missing url"))?,
                    other => return Err(anyhow!("unsupported Claude image source type: {other:?}")),
                };
                out.push(InternalContentBlock::ImageUrl { url, detail: None });
            }
            Some("document") => {
                let source = part
                    .get("source")
                    .and_then(Value::as_object)
                    .ok_or_else(|| anyhow!("Claude document.source must be an object"))?;
                let source_type = source.get("type").and_then(Value::as_str);
                let url = (source_type == Some("url"))
                    .then(|| source.get("url").and_then(Value::as_str))
                    .flatten()
                    .filter(|value| !value.trim().is_empty())
                    .map(ToString::to_string);
                let data = (source_type == Some("base64"))
                    .then(|| source.get("data").and_then(Value::as_str))
                    .flatten()
                    .filter(|value| !value.trim().is_empty())
                    .map(ToString::to_string);
                if url.is_some() || data.is_some() {
                    out.push(InternalContentBlock::File {
                        file_id: None,
                        url,
                        data,
                        media_type: source
                            .get("media_type")
                            .and_then(Value::as_str)
                            .map(ToString::to_string)
                            .or_else(|| {
                                (source_type == Some("base64"))
                                    .then(|| "application/pdf".to_string())
                            }),
                        filename: part
                            .get("title")
                            .or_else(|| part.get("filename"))
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                    });
                }
            }
            Some("tool_use") => out.push(InternalContentBlock::ToolCall {
                id: part
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: part
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: part
                    .get("input")
                    .filter(|value| value.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            }),
            Some("tool_result") => {
                let output = match part.get("content") {
                    Some(Value::Array(items)) => normalize_claude_tool_result(items)?,
                    Some(value) => value.clone(),
                    None => Value::String(String::new()),
                };
                out.push(InternalContentBlock::ToolResult {
                    call_id: part
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: None,
                    output,
                });
            }
            _ => {}
        }
    }
    Ok(())
}

fn normalize_claude_tool_result(items: &[Value]) -> Result<Value> {
    let has_image = items.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("image")
    });
    if !has_image {
        return Ok(Value::String(
            items
                .iter()
                .filter_map(Value::as_object)
                .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|item| item.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }

    let mut output = Vec::with_capacity(items.len());
    for item in items {
        let item = item
            .as_object()
            .ok_or_else(|| anyhow!("Claude tool_result content blocks must be objects"))?;
        match item.get("type").and_then(Value::as_str) {
            Some("text") => output.push(json!({
                "type": "input_text",
                "text": item.get("text").and_then(Value::as_str).unwrap_or_default()
            })),
            Some("image") => {
                let mut parsed = Vec::new();
                parse_claude_parts(std::slice::from_ref(item), &mut parsed)?;
                let Some(InternalContentBlock::ImageUrl { url, .. }) = parsed.into_iter().next() else {
                    return Err(anyhow!("invalid Claude tool_result image"));
                };
                output.push(json!({"type":"input_image","detail":"auto","image_url":url}));
            }
            _ => {}
        }
    }
    Ok(Value::Array(output))
}

fn parse_claude_code(body: &Map<String, Value>) -> Result<InternalRequest> {
    let mut messages = Vec::new();
    let options = body
        .get("options")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
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
        model: options
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("claude-sonnet-4-5")
            .to_string(),
        stream: false,
        tools: Vec::new(),
        tool_choice: options.get("tool_choice").cloned(),
        thinking: options.get("thinking").cloned(),
        extra: filter_keys(
            &options,
            &[
                "model",
                "systemPrompt",
                "mcpServers",
                "tool_choice",
                "thinking",
            ],
        ),
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
            extra
                .entry(key.to_string())
                .or_insert_with(|| value.clone());
        }
    }

    Ok(InternalRequest {
        messages,
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        stream: body.get("stream").and_then(Value::as_bool).unwrap_or(false),
        tools: parse_responses_tools(body.get("tools")),
        tool_choice: body.get("tool_choice").cloned(),
        thinking: None,
        extra,
    })
}

fn parse_responses_input_item(item: &Value) -> Result<Option<InternalMessage>> {
    let Some(item) = item.as_object() else {
        return Ok(None);
    };
    let item_type = item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("message");
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
        "message" | "input_text" | "output_text" | "" => responses_content_blocks(
            item.get("content"),
            item.get("text").and_then(Value::as_str),
        ),
        "function_call_output" | "tool_result" => {
            vec![InternalContentBlock::ToolResult {
                call_id: item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: item
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                output: normalize_responses_tool_output(
                    item.get("output"),
                    item.get("text").and_then(Value::as_str),
                ),
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
            "system" | "developer" => "system",
            "function_call_output" | "tool" => "tool",
            _ => "user",
        }
        .to_string(),
        content,
    }))
}

fn responses_content_blocks(
    content: Option<&Value>,
    default_text: Option<&str>,
) -> Vec<InternalContentBlock> {
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

fn normalize_responses_tool_output(output: Option<&Value>, default_text: Option<&str>) -> Value {
    if !responses_output_has_image(output) {
        return Value::String(responses_output_text(output, default_text));
    }
    match output {
        Some(Value::Array(items)) => normalize_responses_output_items(items),
        Some(Value::Object(object)) => object
            .get("items")
            .and_then(Value::as_array)
            .map(|items| normalize_responses_output_items(items))
            .unwrap_or_else(|| Value::Object(object.clone())),
        Some(value) => value.clone(),
        None => Value::String(default_text.unwrap_or_default().to_string()),
    }
}

fn responses_output_has_image(output: Option<&Value>) -> bool {
    match output {
        Some(Value::Array(items)) => items.iter().any(|item| {
            matches!(
                item.get("type").and_then(Value::as_str),
                Some("input_image") | Some("image_url")
            )
        }),
        Some(Value::Object(object)) => {
            matches!(
                object.get("type").and_then(Value::as_str),
                Some("input_image") | Some("image_url")
            ) || object
                .get("items")
                .and_then(Value::as_array)
                .is_some_and(|items| {
                    items.iter().any(|item| {
                        matches!(
                            item.get("type").and_then(Value::as_str),
                            Some("input_image") | Some("image_url")
                        )
                    })
                })
        }
        _ => false,
    }
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

fn normalize_responses_output_items(items: &[Value]) -> Value {
    Value::Array(
        items
            .iter()
            .map(|item| match item.get("type").and_then(Value::as_str) {
                Some("text") | Some("input_text") | Some("output_text") => json!({
                    "type":"input_text",
                    "text":item.get("text").cloned().unwrap_or(Value::String(String::new()))
                }),
                Some("input_image") | Some("image_url") => {
                    let image_url = item
                        .get("image_url")
                        .cloned()
                        .or_else(|| item.get("url").cloned())
                        .unwrap_or(Value::String(String::new()));
                    let detail = item
                        .get("detail")
                        .cloned()
                        .unwrap_or(Value::String("auto".to_string()));
                    json!({"type":"input_image","detail":detail,"image_url":image_url})
                }
                _ => item.clone(),
            })
            .collect(),
    )
}

fn push_responses_content_block(
    content: &mut Vec<InternalContentBlock>,
    part: &Map<String, Value>,
) {
    match part.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") | Some("text") => {
            content.push(InternalContentBlock::Text(
                part.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ));
        }
        Some("input_image") | Some("image_url") => {
            let image_url = part.get("image_url");
            let url = image_url
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .or_else(|| {
                    part.get("url")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                })
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
        Some("input_file") => {
            let url = part
                .get("file_url")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToString::to_string);
            let data = part
                .get("file_data")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(parse_file_data);
            let file_id = part
                .get("file_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToString::to_string);
            if file_id.is_some() || url.is_some() || data.is_some() {
                let (data, media_type) = data
                    .map(|(data, media_type)| (Some(data), media_type))
                    .unwrap_or((None, None));
                content.push(InternalContentBlock::File {
                    file_id,
                    url,
                    data,
                    media_type,
                    filename: part
                        .get("filename")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                });
            }
        }
        Some("input_audio") => {
            if let Some(audio) = part.get("input_audio").and_then(Value::as_object) {
                if let (Some(data), Some(format)) = (
                    audio.get("data").and_then(Value::as_str),
                    audio.get("format").and_then(Value::as_str),
                ) {
                    content.push(InternalContentBlock::Audio {
                        data: data.to_string(),
                        format: format.to_string(),
                    });
                }
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
                id: part
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: part
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments,
            });
        }
        Some("function_call_output") => {
            let output = normalize_responses_tool_output(part.get("output"), None);
            content.push(InternalContentBlock::ToolResult {
                call_id: part
                    .get("call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: part
                    .get("name")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
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
                    } else if let Some(file_data) = part.get("fileData").and_then(Value::as_object)
                    {
                        if let Some(url) = file_data
                            .get("fileUri")
                            .and_then(Value::as_str)
                            .filter(|value| !value.trim().is_empty())
                        {
                            blocks.push(InternalContentBlock::File {
                                file_id: None,
                                url: Some(url.to_string()),
                                data: None,
                                media_type: file_data
                                    .get("mimeType")
                                    .and_then(Value::as_str)
                                    .map(ToString::to_string),
                                filename: None,
                            });
                        }
                    } else if let Some(inline_data) =
                        part.get("inlineData").and_then(Value::as_object)
                    {
                        if let Some(data) = inline_data
                            .get("data")
                            .and_then(Value::as_str)
                            .filter(|value| !value.trim().is_empty())
                        {
                            let media_type = inline_data
                                .get("mimeType")
                                .and_then(Value::as_str)
                                .map(ToString::to_string);
                            if media_type
                                .as_deref()
                                .is_some_and(|value| value.starts_with("audio/"))
                            {
                                blocks.push(InternalContentBlock::Audio {
                                    data: data.to_string(),
                                    format: media_type
                                        .as_deref()
                                        .and_then(|value| value.strip_prefix("audio/"))
                                        .unwrap_or("wav")
                                        .to_string(),
                                });
                            } else {
                                blocks.push(InternalContentBlock::File {
                                    file_id: None,
                                    url: None,
                                    data: Some(data.to_string()),
                                    media_type,
                                    filename: None,
                                });
                            }
                        }
                    } else if let Some(function_call) =
                        part.get("functionCall").and_then(Value::as_object)
                    {
                        blocks.push(InternalContentBlock::ToolCall {
                            id: function_call
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: function_call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            arguments: function_call
                                .get("args")
                                .cloned()
                                .unwrap_or_else(|| json!({})),
                        });
                    } else if let Some(function_response) =
                        part.get("functionResponse").and_then(Value::as_object)
                    {
                        blocks.push(InternalContentBlock::ToolResult {
                            call_id: function_response
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_string(),
                            name: function_response
                                .get("name")
                                .and_then(Value::as_str)
                                .map(ToString::to_string),
                            output: function_response
                                .get("response")
                                .cloned()
                                .unwrap_or_else(|| json!({})),
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
            "cachedContent",
        ],
    );
    if let Some(generation_config) = body.get("generationConfig") {
        extra.insert("generationConfig".to_string(), generation_config.clone());
    }
    if let Some(safety_settings) = body.get("safetySettings") {
        extra.insert("safetySettings".to_string(), safety_settings.clone());
    }
    if let Some(cached_content) = body.get("cachedContent") {
        if cached_content.as_str().is_some() {
            extra.insert("cachedContent".to_string(), cached_content.clone());
        }
    }
    Ok(InternalRequest {
        messages,
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("gemini-2.5-flash")
            .to_string(),
        stream: path.contains("streamGenerateContent"),
        tools: parse_gemini_tools(body.get("tools")),
        tool_choice: body.get("toolConfig").cloned(),
        thinking: None,
        extra,
    })
}

fn emit_openai_chat(req: &InternalRequest) -> Value {
    let mut body = normalize_extra_for_openai_chat(&req.extra, &req.model);
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));
    let mut messages = Vec::new();
    for message in &req.messages {
        if message.role != "tool" {
            messages.push(openai_chat_message(message));
        }
        for block in &message.content {
            if let InternalContentBlock::ToolResult {
                call_id,
                name,
                output,
            } = block
            {
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": openai_tool_result_content(output)
                }));
            }
        }
    }
    body.insert("messages".to_string(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(req.tools.iter().map(openai_tool).collect()),
        );
    }
    if !req.tools.is_empty() {
        if let Some(tool_choice) = &req.tool_choice {
            if let Some(tool_choice) = normalize_tool_choice_for_openai_chat(tool_choice) {
                body.insert("tool_choice".to_string(), tool_choice);
            }
        }
    }
    if !body.contains_key("reasoning_effort") {
        if let Some(effort) = normalize_claude_reasoning_effort(req.thinking.as_ref(), &req.extra)
            .filter(|_| supports_reasoning_effort(&req.model))
        {
            body.insert("reasoning_effort".to_string(), json!(effort));
        }
    }
    Value::Object(body)
}

fn emit_claude_chat(req: &InternalRequest) -> Value {
    let mut body = normalize_extra_for_claude(&req.extra);
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));
    let thinking_plan = normalize_claude_thinking_for_request(req, &body);
    if let Some(plan) = &thinking_plan {
        body.insert("thinking".to_string(), plan.thinking.clone());
        if let Some(output_config) = &plan.output_config {
            body.insert("output_config".to_string(), output_config.clone());
        }
    }

    let mut messages = Vec::new();
    let mut system_text = Vec::new();
    for message in &req.messages {
        if matches!(message.role.as_str(), "system" | "developer") {
            for block in &message.content {
                if let InternalContentBlock::Text(text) = block {
                    if !text.is_empty() {
                        system_text.push(text.clone());
                    }
                }
            }
            continue;
        }
        if let Some(message) = claude_message(message) {
            messages.push(message);
        }
    }

    if !system_text.is_empty() {
        body.insert("system".to_string(), Value::String(system_text.join("\n")));
    }
    body.insert("messages".to_string(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(req.tools.iter().map(claude_tool).collect()),
        );
    }
    if !req.tools.is_empty() {
        if let Some(tool_choice) = &req.tool_choice {
            if let Some(tool_choice) = normalize_tool_choice_for_claude(tool_choice) {
                body.insert("tool_choice".to_string(), tool_choice);
            }
        }
    }

    let mut thinking_enabled = thinking_plan.as_ref().is_some_and(|plan| plan.enabled);
    if !req.tools.is_empty()
        && req
            .extra
            .get("parallel_tool_calls")
            .and_then(Value::as_bool)
            == Some(false)
    {
        let mut tool_choice = body
            .get("tool_choice")
            .cloned()
            .unwrap_or_else(|| json!({"type": "auto"}));
        if let Some(tool_choice) = tool_choice.as_object_mut() {
            tool_choice.insert("disable_parallel_tool_use".to_string(), json!(true));
        }
        body.insert("tool_choice".to_string(), tool_choice);
    }

    let forced_tool_choice = body
        .get("tool_choice")
        .and_then(Value::as_object)
        .and_then(|choice| choice.get("type"))
        .and_then(Value::as_str)
        .is_some_and(|kind| matches!(kind, "any" | "tool"));
    if thinking_enabled && forced_tool_choice {
        body.insert("thinking".to_string(), json!({"type": "disabled"}));
        body.remove("output_config");
        thinking_enabled = false;
    }
    if thinking_enabled {
        body.remove("temperature");
        body.remove("top_p");
    }

    Value::Object(body)
}

fn emit_openai_responses(req: &InternalRequest) -> Value {
    let mut body = normalize_extra_for_openai_responses(&req.extra);
    body.insert("model".to_string(), Value::String(req.model.clone()));
    body.insert("stream".to_string(), Value::Bool(req.stream));

    let mut input = Vec::new();
    let mut instructions = Vec::new();
    for message in &req.messages {
        if matches!(message.role.as_str(), "system" | "developer") {
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
        body.insert(
            "instructions".to_string(),
            Value::String(instructions.join("\n\n")),
        );
    }
    if input.is_empty() {
        body.insert("input".to_string(), Value::String(String::new()));
    } else {
        body.insert("input".to_string(), Value::Array(input));
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(req.tools.iter().map(responses_tool).collect()),
        );
    }
    if !req.tools.is_empty() {
        if let Some(tool_choice) = &req.tool_choice {
            if let Some(tool_choice) = normalize_tool_choice_for_openai_responses(tool_choice) {
                body.insert("tool_choice".to_string(), tool_choice);
            }
        }
    }
    if !body.contains_key("reasoning") {
        if let Some(effort) = normalize_claude_reasoning_effort(req.thinking.as_ref(), &req.extra)
            .filter(|_| supports_reasoning_effort(&req.model))
        {
            body.insert("reasoning".to_string(), json!({"effort": effort}));
        }
    }
    Value::Object(body)
}

fn emit_gemini_chat(req: &InternalRequest) -> Value {
    let mut body = Map::new();
    let contents = req
        .messages
        .iter()
        .filter(|message| !matches!(message.role.as_str(), "system" | "developer"))
        .filter_map(gemini_message)
        .collect::<Vec<_>>();
    body.insert("contents".to_string(), Value::Array(contents));
    let system_text = req
        .messages
        .iter()
        .filter(|message| matches!(message.role.as_str(), "system" | "developer"))
        .flat_map(|message| message.content.iter())
        .filter_map(|block| match block {
            InternalContentBlock::Text(text) if !text.is_empty() => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if !system_text.is_empty() {
        body.insert(
            "systemInstruction".to_string(),
            json!({"parts": [{"text": system_text.join("\n") }]}),
        );
    }
    if !req.tools.is_empty() {
        body.insert(
            "tools".to_string(),
            Value::Array(vec![json!({
                "functionDeclarations": req.tools.iter().map(gemini_tool_decl).collect::<Vec<_>>()
            })]),
        );
    }
    if !req.tools.is_empty() {
        if let Some(tool_choice) = &req.tool_choice {
            if let Some(tool_choice) = normalize_tool_choice_for_gemini(tool_choice) {
                body.insert("toolConfig".to_string(), tool_choice);
            }
        }
    }
    body.extend(normalize_extra_for_gemini(&req.extra));
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
                        InternalContentBlock::ToolCall { .. }
                            | InternalContentBlock::ToolResult { .. }
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
        thinking: req.thinking,
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
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.trim().is_empty())?;
            Some(InternalTool {
                name: name.to_string(),
                description: function
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToString::to_string),
                input_schema: function
                    .get("parameters")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
                strict: function.get("strict").and_then(Value::as_bool),
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
        .filter(|tool| tool.get("type").and_then(Value::as_str) != Some("BatchTool"))
        .filter_map(|tool| {
            let function = tool.get("function").and_then(Value::as_object);
            let name = tool
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| function.and_then(|func| func.get("name").and_then(Value::as_str)))
                .filter(|name| !name.trim().is_empty())?;
            Some(InternalTool {
                name: name.to_string(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .or_else(|| {
                        function
                            .and_then(|func| func.get("description").and_then(Value::as_str))
                            .map(ToString::to_string)
                    }),
                input_schema: tool
                    .get("input_schema")
                    .cloned()
                    .or_else(|| function.and_then(|func| func.get("parameters").cloned()))
                    .unwrap_or_else(|| json!({})),
                strict: None,
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
        .filter_map(|tool| {
            let function = tool.get("function").and_then(Value::as_object);
            let name = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| function.and_then(|func| func.get("name").and_then(Value::as_str)))
                    .filter(|name| !name.trim().is_empty())?;
            Some(InternalTool {
                name: name.to_string(),
                description: tool
                    .get("description")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .or_else(|| {
                        function
                            .and_then(|func| func.get("description").and_then(Value::as_str))
                            .map(ToString::to_string)
                    }),
                input_schema: tool
                    .get("parameters")
                    .cloned()
                    .or_else(|| function.and_then(|func| func.get("parameters").cloned()))
                    .unwrap_or_else(|| json!({})),
                strict: tool
                    .get("strict")
                    .and_then(Value::as_bool)
                    .or_else(|| function.and_then(|func| func.get("strict").and_then(Value::as_bool))),
            })
        })
        .collect()
}

fn parse_gemini_tools(value: Option<&Value>) -> Vec<InternalTool> {
    let mut tools = Vec::new();
    if let Some(Value::Array(tool_sets)) = value {
        for tool_set in tool_sets.iter().filter_map(Value::as_object) {
            if let Some(Value::Array(decls)) = tool_set.get("functionDeclarations") {
                for decl in decls.iter().filter_map(Value::as_object) {
                    let Some(name) = decl
                        .get("name")
                        .and_then(Value::as_str)
                        .filter(|name| !name.trim().is_empty())
                    else {
                        continue;
                    };
                    tools.push(InternalTool {
                        name: name.to_string(),
                        description: decl
                            .get("description")
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                        input_schema: decl.get("parameters").cloned().unwrap_or_else(|| json!({})),
                        strict: None,
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
            InternalContentBlock::File {
                file_id,
                url: _,
                data,
                media_type,
                filename,
            } => {
                let mut file = Map::new();
                if let Some(file_id) = file_id {
                    file.insert("file_id".to_string(), Value::String(file_id.clone()));
                } else if let Some(data) = data {
                    file.insert(
                        "file_data".to_string(),
                        Value::String(file_data_url(data, media_type.as_deref())),
                    );
                }
                if let Some(filename) = filename {
                    file.insert("filename".to_string(), Value::String(filename.clone()));
                }
                if !file.is_empty() {
                    rich_parts.push(json!({"type":"file","file":Value::Object(file)}));
                }
            }
            InternalContentBlock::Audio { data, format } => {
                rich_parts.push(json!({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": format}
                }));
            }
            InternalContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
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
        if rich_parts.iter().any(|part| {
            part.get("type").and_then(Value::as_str) != Some("text")
        }) {
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

fn claude_message(message: &InternalMessage) -> Option<Value> {
    let parts = message
        .content
        .iter()
        .filter_map(|block| match block {
            InternalContentBlock::Text(text) => Some(json!({"type":"text","text": text})),
            InternalContentBlock::ImageUrl { url, .. } => Some(claude_image_block(url)),
            InternalContentBlock::File {
                file_id,
                url,
                data,
                media_type,
                filename,
            } => (file_id.is_none() && (url.is_some() || data.is_some())).then(|| {
                claude_document_block(
                    url.as_deref(),
                    data.as_deref(),
                    media_type.as_deref(),
                    filename.as_deref(),
                )
            }),
            InternalContentBlock::Audio { .. } => None,
            InternalContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(json!({
                "type":"tool_use",
                "id": id,
                "name": name,
                "input": arguments
            })),
            InternalContentBlock::ToolResult {
                call_id, output, ..
            } => Some(json!({
                "type":"tool_result",
                "tool_use_id": call_id,
                "content": claude_tool_result_content(output)
            })),
        })
        .collect::<Vec<_>>();
    (!parts.is_empty()).then(|| json!({
        "role": if message.role == "assistant" { "assistant" } else { "user" },
        "content": parts
    }))
}

fn split_image_data_url(url: &str) -> Option<(&str, &str)> {
    let rest = url.strip_prefix("data:")?;
    let (metadata, data) = rest.split_once(',')?;
    let media_type = metadata.strip_suffix(";base64")?;
    if media_type.is_empty() || data.is_empty() {
        return None;
    }
    Some((media_type, data))
}

fn claude_image_block(url: &str) -> Value {
    if let Some((media_type, data)) = split_image_data_url(url) {
        json!({"type":"image","source":{"type":"base64","media_type":media_type,"data":data}})
    } else {
        json!({"type":"image","source":{"type":"url","url":url}})
    }
}

fn file_data_url(data: &str, media_type: Option<&str>) -> String {
    if data.starts_with("data:") {
        data.to_string()
    } else {
        format!(
            "data:{};base64,{}",
            media_type.unwrap_or("application/octet-stream"),
            data
        )
    }
}

fn parse_file_data(data: &str) -> (String, Option<String>) {
    if let Some((media_type, payload)) = split_image_data_url(data) {
        return (payload.to_string(), Some(media_type.to_string()));
    }
    (data.to_string(), None)
}

fn format_to_audio_mime(format: &str) -> String {
    match format {
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        other if other.starts_with("audio/") => other,
        other => return format!("audio/{other}"),
    }
    .to_string()
}

fn claude_document_block(
    url: Option<&str>,
    data: Option<&str>,
    media_type: Option<&str>,
    filename: Option<&str>,
) -> Value {
    let mut block = if let Some(data) = data {
        let data_url = file_data_url(data, media_type.or(Some("application/pdf")));
        let (media_type, data) = split_image_data_url(&data_url)
            .unwrap_or((media_type.unwrap_or("application/pdf"), data));
        json!({
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": data}
        })
    } else {
        json!({
            "type": "document",
            "source": {"type": "url", "url": url.unwrap_or_default()}
        })
    };
    if let Some(filename) = filename {
        block["title"] = Value::String(filename.to_string());
    }
    block
}

fn openai_tool_result_content(output: &Value) -> Value {
    let Value::Array(items) = output else {
        return match output {
            Value::String(text) => Value::String(text.clone()),
            other => Value::String(serde_json::to_string(other).unwrap_or_default()),
        };
    };
    Value::Array(
        items
            .iter()
            .map(|item| match item.get("type").and_then(Value::as_str) {
                Some("input_text") => json!({"type":"text","text":item.get("text").cloned().unwrap_or(Value::String(String::new()))}),
                Some("input_image") => {
                    let url = item.get("image_url").cloned().unwrap_or(Value::String(String::new()));
                    let detail = item.get("detail").cloned().unwrap_or(Value::String("auto".to_string()));
                    json!({"type":"image_url","image_url":{"url":url,"detail":detail}})
                }
                _ => item.clone(),
            })
            .collect(),
    )
}

fn claude_tool_result_content(output: &Value) -> Value {
    let Value::Array(items) = output else {
        return match output {
            Value::String(text) => Value::Array(vec![json!({"type":"text","text":text})]),
            other => Value::Array(vec![json!({"type":"text","text":serde_json::to_string(other).unwrap_or_default()})]),
        };
    };
    Value::Array(
        items
            .iter()
            .map(|item| match item.get("type").and_then(Value::as_str) {
                Some("input_text") => json!({"type":"text","text":item.get("text").cloned().unwrap_or(Value::String(String::new()))}),
                Some("input_image") => item
                    .get("image_url")
                    .and_then(Value::as_str)
                    .map(claude_image_block)
                    .unwrap_or_else(|| json!({"type":"text","text":""})),
                _ => json!({"type":"text","text":serde_json::to_string(item).unwrap_or_default()}),
            })
            .collect(),
    )
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
            InternalContentBlock::File {
                file_id,
                url,
                data,
                media_type,
                filename,
            } => {
                let mut part = Map::new();
                part.insert("type".to_string(), Value::String("input_file".to_string()));
                if let Some(file_id) = file_id {
                    part.insert("file_id".to_string(), Value::String(file_id.clone()));
                } else if let Some(data) = data {
                    part.insert(
                        "file_data".to_string(),
                        Value::String(file_data_url(data, media_type.as_deref())),
                    );
                } else if let Some(url) = url {
                    part.insert("file_url".to_string(), Value::String(url.clone()));
                }
                if let Some(filename) = filename {
                    part.insert("filename".to_string(), Value::String(filename.clone()));
                }
                if part.len() > 1 {
                    content.push(Value::Object(part));
                }
            }
            InternalContentBlock::Audio { data, format } => {
                content.push(json!({
                    "type": "input_audio",
                    "input_audio": {"data": data, "format": format}
                }));
            }
            InternalContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                return json!({
                    "type":"function_call",
                    "id": id,
                    "call_id": id,
                    "name": name,
                    "arguments": serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string()),
                    "status": "completed"
                });
            }
            InternalContentBlock::ToolResult {
                call_id,
                name,
                output,
            } => {
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
            InternalContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => Some(json!({
                "functionCall": {
                    "id": id,
                    "name": name,
                    "args": arguments
                }
            })),
            InternalContentBlock::ToolResult {
                call_id,
                name,
                output,
            } => Some(json!({
                "functionResponse": {
                    "id": call_id,
                    "name": name,
                    "response": output
                }
            })),
            InternalContentBlock::File {
                file_id,
                url,
                data,
                media_type,
                ..
            } => {
                if file_id.is_some() {
                    None
                } else if let Some(data) = data {
                    Some(json!({
                        "inlineData": {
                            "mimeType": media_type.as_deref().unwrap_or("application/octet-stream"),
                            "data": data
                        }
                    }))
                } else {
                    url.as_ref().map(|url| json!({
                        "fileData": {
                            "mimeType": media_type.as_deref().unwrap_or("application/octet-stream"),
                            "fileUri": url
                        }
                    }))
                }
            }
            InternalContentBlock::Audio { data, format } => Some(json!({
                "inlineData": {
                    "mimeType": format_to_audio_mime(format),
                    "data": data
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
    let mut function = json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": normalize_openai_input_schema(&tool.input_schema)
    });
    if let Some(strict) = tool.strict {
        function["strict"] = Value::Bool(strict);
    }
    json!({
        "type": "function",
        "function": function
    })
}

fn claude_tool(tool: &InternalTool) -> Value {
    let mut output = json!({
        "name": tool.name,
        "input_schema": normalize_claude_input_schema(&tool.input_schema)
    });
    if let Some(description) = &tool.description {
        output["description"] = Value::String(description.clone());
    }
    if let Some(strict) = tool.strict {
        output["strict"] = Value::Bool(strict);
    }
    output
}

fn responses_tool(tool: &InternalTool) -> Value {
    let mut output = json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": normalize_openai_input_schema(&tool.input_schema)
    });
    if let Some(strict) = tool.strict {
        output["strict"] = Value::Bool(strict);
    }
    output
}

fn gemini_tool_decl(tool: &InternalTool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": normalize_openai_input_schema(&tool.input_schema)
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
    if !matches!(schema.get("required"), Some(Value::Array(_)) | None) {
        schema.remove("required");
    }

    Value::Object(schema)
}

fn normalize_openai_input_schema(schema: &Value) -> Value {
    let mut schema = match schema {
        Value::Object(map) => map.clone(),
        _ => Map::new(),
    };
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        schema.insert("type".to_string(), Value::String("object".to_string()));
    }
    if !matches!(schema.get("properties"), Some(Value::Object(_))) {
        schema.insert("properties".to_string(), Value::Object(Map::new()));
    }
    Value::Object(schema)
}

fn normalize_tool_choice_for_openai_chat(tool_choice: &Value) -> Option<Value> {
    match normalize_tool_choice(tool_choice)? {
        NormalizedToolChoice::Auto => Some(json!("auto")),
        NormalizedToolChoice::None => Some(json!("none")),
        NormalizedToolChoice::Required => Some(json!("required")),
        NormalizedToolChoice::Tool(name) => Some(json!({
            "type": "function",
            "function": {"name": name}
        })),
    }
}

fn normalize_tool_choice_for_claude(tool_choice: &Value) -> Option<Value> {
    match normalize_tool_choice(tool_choice).unwrap_or(NormalizedToolChoice::Auto) {
        NormalizedToolChoice::Auto => Some(json!({"type": "auto"})),
        NormalizedToolChoice::Required => Some(json!({"type": "any"})),
        NormalizedToolChoice::Tool(name) => Some(json!({"type": "tool", "name": name})),
        // Anthropic has no `none` tool-choice mode. Omitting it is safer than
        // sending a non-Anthropic discriminator and is consistent with the
        // whitelist rule for unsupported target fields.
        NormalizedToolChoice::None => None,
    }
}

fn normalize_tool_choice_for_openai_responses(tool_choice: &Value) -> Option<Value> {
    match normalize_tool_choice(tool_choice)? {
        NormalizedToolChoice::Auto => Some(json!("auto")),
        NormalizedToolChoice::None => Some(json!("none")),
        NormalizedToolChoice::Required => Some(json!("required")),
        NormalizedToolChoice::Tool(name) => Some(json!({"type": "function", "name": name})),
    }
}

fn normalize_tool_choice_for_gemini(tool_choice: &Value) -> Option<Value> {
    if let Value::Object(choice) = tool_choice {
        if let Some(config) = choice.get("functionCallingConfig").and_then(Value::as_object) {
            let mode = config.get("mode").and_then(Value::as_str)?;
            if !matches!(mode, "AUTO" | "NONE" | "ANY") {
                return None;
            }
            let mut normalized = Map::new();
            normalized.insert("mode".to_string(), Value::String(mode.to_string()));
            if mode == "ANY" {
                if let Some(names) = config.get("allowedFunctionNames").and_then(Value::as_array) {
                    let names = names
                        .iter()
                        .filter_map(Value::as_str)
                        .filter(|name| !name.is_empty())
                        .map(|name| Value::String(name.to_string()))
                        .collect::<Vec<_>>();
                    if !names.is_empty() {
                        normalized.insert("allowedFunctionNames".to_string(), Value::Array(names));
                    }
                }
            }
            return Some(json!({"functionCallingConfig": normalized}));
        }
    }

    let config = match normalize_tool_choice(tool_choice)? {
        NormalizedToolChoice::Auto => json!({"mode": "AUTO"}),
        NormalizedToolChoice::None => json!({"mode": "NONE"}),
        NormalizedToolChoice::Required => json!({"mode": "ANY"}),
        NormalizedToolChoice::Tool(name) => {
            json!({"mode": "ANY", "allowedFunctionNames": [name]})
        }
    };

    Some(json!({"functionCallingConfig": config}))
}

fn normalize_tool_choice(tool_choice: &Value) -> Option<NormalizedToolChoice> {
    match tool_choice {
        Value::String(mode) => normalize_tool_choice_mode(mode),
        Value::Object(choice) => {
            if let Some(config) = choice.get("functionCallingConfig").and_then(Value::as_object) {
                let mode = config.get("mode").and_then(Value::as_str)?;
                return match mode {
                    "AUTO" => Some(NormalizedToolChoice::Auto),
                    "NONE" => Some(NormalizedToolChoice::None),
                    "ANY" => {
                        let names = config
                            .get("allowedFunctionNames")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                            .filter_map(Value::as_str)
                            .filter(|name| !name.is_empty())
                            .collect::<Vec<_>>();
                        if names.len() == 1 {
                            Some(NormalizedToolChoice::Tool(names[0].to_string()))
                        } else {
                            Some(NormalizedToolChoice::Required)
                        }
                    }
                    _ => None,
                };
            }

            match choice.get("type").and_then(Value::as_str) {
                Some("function") | Some("tool") => choice
                    .get("name")
                    .and_then(Value::as_str)
                    .or_else(|| {
                        choice
                            .get("function")
                            .and_then(Value::as_object)
                            .and_then(|function| function.get("name").and_then(Value::as_str))
                    })
                    .filter(|name| !name.is_empty())
                    .map(|name| NormalizedToolChoice::Tool(name.to_string())),
                Some(mode) => normalize_tool_choice_mode(mode),
                None => None,
            }
        }
        _ => None,
    }
}

fn normalize_tool_choice_mode(mode: &str) -> Option<NormalizedToolChoice> {
    match mode {
        "auto" => Some(NormalizedToolChoice::Auto),
        "none" => Some(NormalizedToolChoice::None),
        "required" | "any" => Some(NormalizedToolChoice::Required),
        _ => None,
    }
}

fn normalize_claude_reasoning_effort(
    thinking: Option<&Value>,
    extra: &Map<String, Value>,
) -> Option<&'static str> {
    if let Some(effort) = extra
        .get("output_config")
        .and_then(Value::as_object)
        .and_then(|config| config.get("effort"))
        .and_then(Value::as_str)
    {
        return match effort {
            "low" => Some("low"),
            "medium" => Some("medium"),
            "high" => Some("high"),
            "max" => Some("xhigh"),
            _ => None,
        };
    }

    let thinking = thinking?.as_object()?;
    match thinking.get("type").and_then(Value::as_str) {
        Some("adaptive") => Some("xhigh"),
        Some("enabled") => match thinking.get("budget_tokens").and_then(Value::as_u64) {
            Some(budget) if budget < 4_000 => Some("low"),
            Some(budget) if budget < 16_000 => Some("medium"),
            Some(_) | None => Some("high"),
        },
        _ => None,
    }
}

fn normalize_claude_thinking_for_request(
    req: &InternalRequest,
    body: &Map<String, Value>,
) -> Option<ClaudeThinkingPlan> {
    if let Some(thinking) = normalize_claude_thinking_for_claude(req.thinking.as_ref()) {
        let enabled = thinking
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind != "disabled");
        return Some(ClaudeThinkingPlan {
            thinking,
            output_config: None,
            enabled,
        });
    }

    let effort = req
        .extra
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(Value::as_str)
        .or_else(|| req.extra.get("reasoning_effort").and_then(Value::as_str))?;

    if matches!(effort, "none" | "off" | "disabled") {
        return Some(ClaudeThinkingPlan {
            thinking: json!({"type": "disabled"}),
            output_config: None,
            enabled: false,
        });
    }

    let normalized_effort = normalize_claude_effort(effort)?;
    if is_claude_adaptive_model(&req.model) {
        return Some(ClaudeThinkingPlan {
            thinking: json!({"type": "adaptive"}),
            output_config: Some(json!({"effort": normalized_effort})),
            enabled: true,
        });
    }

    let requested_budget = match normalized_effort {
        "low" => 2_048,
        "medium" => 8_192,
        "high" => 16_384,
        "max" => 24_576,
        _ => return None,
    };
    let max_tokens = body
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_CLAUDE_MAX_TOKENS);
    let budget = requested_budget.min(max_tokens / 2);
    if budget < 1_024 {
        return None;
    }

    Some(ClaudeThinkingPlan {
        thinking: json!({"type": "enabled", "budget_tokens": budget}),
        output_config: None,
        enabled: true,
    })
}

fn normalize_claude_effort(effort: &str) -> Option<&'static str> {
    match effort.trim().to_ascii_lowercase().as_str() {
        "minimal" | "low" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" | "max" => Some("max"),
        _ => None,
    }
}

fn is_claude_adaptive_model(model: &str) -> bool {
    let normalized = model
        .trim()
        .to_ascii_lowercase()
        .replace('.', "-")
        .replace('_', "-");
    [
        "fable-5",
        "mythos-5",
        "mythos-preview",
        "sonnet-5",
        "opus-4-8",
        "opus-4-7",
        "opus-4-6",
        "sonnet-4-6",
    ]
    .iter()
    .any(|needle| normalized.contains(needle))
}

fn supports_reasoning_effort(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    is_openai_o_series(&model)
        || model
            .strip_prefix("gpt-")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|character| character.is_ascii_digit() && character >= '5')
        || model == "grok-4.5"
        || model.starts_with("grok-4.5-")
        || model.starts_with("grok-build-")
}

fn normalize_claude_thinking_for_claude(thinking: Option<&Value>) -> Option<Value> {
    let thinking = thinking?.as_object()?;
    let kind = thinking.get("type").and_then(Value::as_str)?;
    if !matches!(kind, "enabled" | "disabled" | "adaptive") {
        return None;
    }
    let mut out = Map::new();
    out.insert("type".to_string(), Value::String(kind.to_string()));
    if let Some(budget_tokens) = thinking.get("budget_tokens") {
        if budget_tokens.as_u64().is_some() {
            out.insert("budget_tokens".to_string(), budget_tokens.clone());
        }
    }
    Some(Value::Object(out))
}

fn normalize_openai_response_format(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let kind = object.get("type").and_then(Value::as_str)?;
    match kind {
        "text" | "json_object" => Some(json!({"type": kind})),
        "json_schema" => {
            let schema = object
                .get("json_schema")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let mut json_schema = Map::new();
            for key in ["name", "description", "schema", "strict"] {
                if let Some(value) = schema.get(key) {
                    json_schema.insert(key.to_string(), value.clone());
                }
            }
            if !json_schema.contains_key("name") {
                json_schema.insert("name".to_string(), Value::String("response".to_string()));
            }
            if !json_schema.contains_key("schema") {
                json_schema.insert("schema".to_string(), json!({}));
            }
            Some(json!({"type": "json_schema", "json_schema": json_schema}))
        }
        _ => None,
    }
}

fn normalize_claude_output_config(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let effort = object.get("effort").and_then(Value::as_str)?;
    if !matches!(effort, "low" | "medium" | "high" | "max") {
        return None;
    }
    Some(json!({"effort": effort}))
}

fn normalize_claude_context_management(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let edits = object.get("edits").and_then(Value::as_array)?;
    Some(json!({"edits": edits}))
}

fn normalize_extra_for_openai_chat(extra: &Map<String, Value>, model: &str) -> Map<String, Value> {
    let mut out = Map::new();
    copy_allowed_keys(
        &mut out,
        extra,
        &[
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "metadata",
            "n",
            "parallel_tool_calls",
            "presence_penalty",
            "response_format",
            "reasoning_effort",
            "seed",
            "service_tier",
            "stop",
            "temperature",
            "top_p",
            "top_logprobs",
            "user",
            "max_tokens",
            "max_completion_tokens",
        ],
    );
    if let Some(response_format) = extra
        .get("response_format")
        .and_then(normalize_openai_response_format)
    {
        out.insert("response_format".to_string(), response_format);
    } else {
        out.remove("response_format");
    }

    if !out.contains_key("max_tokens") && !out.contains_key("max_completion_tokens") {
        if let Some(value) = extra.get("max_output_tokens") {
            let key = if is_openai_o_series(model) {
                "max_completion_tokens"
            } else {
                "max_tokens"
            };
            out.insert(key.to_string(), value.clone());
        }
    }

    if let Some(Value::Object(generation_config)) = extra.get("generationConfig") {
        if !out.contains_key("max_tokens") && !out.contains_key("max_completion_tokens") {
            if let Some(value) = generation_config.get("maxOutputTokens") {
                let key = if is_openai_o_series(model) {
                    "max_completion_tokens"
                } else {
                    "max_tokens"
                };
                out.insert(key.to_string(), value.clone());
            }
        }
        copy_if_missing(&mut out, generation_config, "temperature", "temperature");
        copy_if_missing(&mut out, generation_config, "topP", "top_p");
        if !out.contains_key("stop") {
            copy_if_missing(&mut out, generation_config, "stopSequences", "stop");
        }
        if !out.contains_key("seed") {
            copy_if_missing(&mut out, generation_config, "seed", "seed");
        }
        if !out.contains_key("frequency_penalty") {
            copy_if_missing(
                &mut out,
                generation_config,
                "frequencyPenalty",
                "frequency_penalty",
            );
        }
        if !out.contains_key("presence_penalty") {
            copy_if_missing(
                &mut out,
                generation_config,
                "presencePenalty",
                "presence_penalty",
            );
        }
        if !out.contains_key("response_format") {
            if let Some(response_format) = response_format_from_gemini_generation(generation_config)
            {
                out.insert("response_format".to_string(), response_format);
            }
        }
    }

    if !out.contains_key("stop") {
        if let Some(stop_sequences) = extra.get("stop_sequences") {
            out.insert("stop".to_string(), stop_sequences.clone());
        }
    }

    if let Some(Value::Object(options)) = extra.get("stream_options") {
        let mut filtered = Map::new();
        copy_allowed_keys(&mut filtered, options, &["include_usage"]);
        if !filtered.is_empty() {
            out.insert("stream_options".to_string(), Value::Object(filtered));
        }
    }

    out
}

fn normalize_extra_for_openai_responses(extra: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    copy_allowed_keys(
        &mut out,
        extra,
        &[
            "metadata",
            "parallel_tool_calls",
            "reasoning",
            "service_tier",
            "store",
            "temperature",
            "top_p",
            "user",
            "include",
            "prompt_cache_key",
            "previous_response_id",
            "conversation",
            "background",
            "max_tool_calls",
            "truncation",
        ],
    );

    if !out.contains_key("reasoning") {
        if let Some(effort) = extra
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .filter(|effort| matches!(*effort, "low" | "medium" | "high" | "xhigh"))
        {
            out.insert("reasoning".to_string(), json!({"effort": effort}));
        }
    }

    if let Some(text) = extra.get("text").and_then(normalize_responses_text_parameter) {
        out.insert("text".to_string(), text);
    }

    if let Some(value) = extra
        .get("max_output_tokens")
        .or_else(|| extra.get("max_tokens"))
        .or_else(|| extra.get("max_completion_tokens"))
    {
        out.insert("max_output_tokens".to_string(), value.clone());
    }

    if let Some(Value::Object(generation_config)) = extra.get("generationConfig") {
        if !out.contains_key("max_output_tokens") {
            if let Some(value) = generation_config.get("maxOutputTokens") {
                out.insert("max_output_tokens".to_string(), value.clone());
            }
        }
        copy_if_missing(&mut out, generation_config, "temperature", "temperature");
        copy_if_missing(&mut out, generation_config, "topP", "top_p");
    }

    let response_format = extra.get("response_format");
    if let Some(response_format) = response_format {
        if !out.contains_key("text") {
            if let Some(text) = normalize_responses_text_format(&response_format) {
                out.insert("text".to_string(), text);
            }
        }
    }

    out
}

fn normalize_extra_for_claude(extra: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    copy_allowed_keys(
        &mut out,
        extra,
        &[
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "metadata",
            "service_tier",
        ],
    );

    if let Some(output_config) = extra
        .get("output_config")
        .and_then(normalize_claude_output_config)
    {
        out.insert("output_config".to_string(), output_config);
    }
    if let Some(context_management) = extra
        .get("context_management")
        .and_then(normalize_claude_context_management)
    {
        out.insert("context_management".to_string(), context_management);
    }

    let max_tokens = [
        extra.get("max_tokens"),
        extra.get("max_output_tokens"),
        extra.get("max_completion_tokens"),
        extra
            .get("generationConfig")
            .and_then(Value::as_object)
            .and_then(|generation_config| generation_config.get("maxOutputTokens")),
    ]
    .into_iter()
    .flatten()
    .find_map(|value| value.as_u64().filter(|value| *value > 0))
    .unwrap_or(DEFAULT_CLAUDE_MAX_TOKENS);
    out.insert("max_tokens".to_string(), json!(max_tokens));

    if let Some(Value::Object(generation_config)) = extra.get("generationConfig") {
        copy_if_missing(&mut out, generation_config, "temperature", "temperature");
        copy_if_missing(&mut out, generation_config, "topP", "top_p");
        if !out.contains_key("stop_sequences") {
            copy_if_missing(
                &mut out,
                generation_config,
                "stopSequences",
                "stop_sequences",
            );
        }
    }

    if let Some(stop) = extra.get("stop") {
        out.insert("stop_sequences".to_string(), stop.clone());
    } else if let Some(stop_sequences) = extra.get("stop_sequences") {
        out.insert("stop_sequences".to_string(), stop_sequences.clone());
    }

    out
}

fn normalize_extra_for_gemini(extra: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    let mut generation_config = Map::new();

    if let Some(Value::Object(existing)) = extra.get("generationConfig") {
        copy_allowed_keys(
            &mut generation_config,
            existing,
            &[
                "candidateCount",
                "maxOutputTokens",
                "temperature",
                "topP",
                "topK",
                "stopSequences",
                "responseMimeType",
                "responseSchema",
                "presencePenalty",
                "frequencyPenalty",
                "seed",
                "responseLogprobs",
                "logprobs",
                "thinkingConfig",
            ],
        );
    }
    if let Some(value) = extra
        .get("max_tokens")
        .or_else(|| extra.get("max_output_tokens"))
        .or_else(|| extra.get("max_completion_tokens"))
    {
        generation_config.insert("maxOutputTokens".to_string(), value.clone());
    }
    copy_alias(&mut generation_config, extra, "temperature", "temperature");
    copy_alias(&mut generation_config, extra, "top_p", "topP");
    copy_alias(&mut generation_config, extra, "top_k", "topK");
    copy_alias(&mut generation_config, extra, "stop_sequences", "stopSequences");
    if !generation_config.contains_key("stopSequences") {
        copy_alias(&mut generation_config, extra, "stop", "stopSequences");
    }
    copy_alias(
        &mut generation_config,
        extra,
        "presence_penalty",
        "presencePenalty",
    );
    copy_alias(
        &mut generation_config,
        extra,
        "frequency_penalty",
        "frequencyPenalty",
    );
    copy_alias(&mut generation_config, extra, "seed", "seed");

    if !generation_config.contains_key("responseMimeType") {
        if let Some(response_format) = extra.get("response_format") {
            if let Some((mime_type, schema)) = gemini_response_format(response_format) {
                generation_config.insert("responseMimeType".to_string(), json!(mime_type));
                if let Some(schema) = schema {
                    generation_config.insert("responseSchema".to_string(), schema);
                }
            }
        }
    }

    if !generation_config.is_empty() {
        out.insert(
            "generationConfig".to_string(),
            Value::Object(generation_config),
        );
    }
    if let Some(value) = extra.get("safetySettings") {
        out.insert("safetySettings".to_string(), value.clone());
    }
    if let Some(value) = extra.get("cachedContent").and_then(Value::as_str) {
        out.insert("cachedContent".to_string(), Value::String(value.to_string()));
    }

    out
}

fn copy_allowed_keys(
    target: &mut Map<String, Value>,
    source: &Map<String, Value>,
    allowed: &[&str],
) {
    for key in allowed {
        if let Some(value) = source.get(*key) {
            target.insert((*key).to_string(), value.clone());
        }
    }
}

fn copy_alias(
    target: &mut Map<String, Value>,
    source: &Map<String, Value>,
    source_key: &str,
    target_key: &str,
) {
    if let Some(value) = source.get(source_key) {
        target.insert(target_key.to_string(), value.clone());
    }
}

fn copy_if_missing(
    target: &mut Map<String, Value>,
    source: &Map<String, Value>,
    source_key: &str,
    target_key: &str,
) {
    if !target.contains_key(target_key) {
        if let Some(value) = source.get(source_key) {
            target.insert(target_key.to_string(), value.clone());
        }
    }
}

fn is_openai_o_series(model: &str) -> bool {
    model.len() > 1
        && model.starts_with('o')
        && model.as_bytes().get(1).is_some_and(|byte| byte.is_ascii_digit())
}

fn normalize_responses_text_parameter(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let mut out = Map::new();
    if let Some(verbosity) = object.get("verbosity").and_then(Value::as_str) {
        if matches!(verbosity, "low" | "medium" | "high") {
            out.insert("verbosity".to_string(), Value::String(verbosity.to_string()));
        }
    }
    if let Some(format) = object.get("format").and_then(normalize_responses_format) {
        out.insert("format".to_string(), format);
    }
    (!out.is_empty()).then_some(Value::Object(out))
}

fn normalize_responses_format(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    let kind = object.get("type").and_then(Value::as_str)?;
    let mut out = Map::new();
    match kind {
        "text" | "json_object" => {
            out.insert("type".to_string(), Value::String(kind.to_string()));
        }
        "json_schema" => {
            out.insert("type".to_string(), Value::String(kind.to_string()));
            for key in ["name", "description", "schema", "strict"] {
                if let Some(value) = object.get(key) {
                    out.insert(key.to_string(), value.clone());
                }
            }
            if !out.contains_key("name") {
                out.insert("name".to_string(), Value::String("response".to_string()));
            }
            if !out.contains_key("schema") {
                out.insert("schema".to_string(), json!({}));
            }
        }
        _ => return None,
    }
    Some(Value::Object(out))
}

fn gemini_response_format(value: &Value) -> Option<(&'static str, Option<Value>)> {
    let object = value.as_object()?;
    match object.get("type").and_then(Value::as_str)? {
        "json_object" => Some(("application/json", None)),
        "json_schema" => {
            let schema = object
                .get("json_schema")
                .and_then(Value::as_object)
                .and_then(|schema| schema.get("schema"))
                .cloned()
                .or_else(|| Some(json!({})));
            Some(("application/json", schema))
        }
        "text" => Some(("text/plain", None)),
        _ => None,
    }
}

fn response_format_from_gemini_generation(generation_config: &Map<String, Value>) -> Option<Value> {
    if generation_config.get("responseMimeType").and_then(Value::as_str)
        != Some("application/json")
    {
        return None;
    }
    if let Some(schema) = generation_config.get("responseSchema") {
        return Some(json!({
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": schema
            }
        }));
    }
    Some(json!({"type": "json_object"}))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_openai_chat_response_format_into_responses_text_format() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        });
        let body = json!({
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"}
                        }
                    },
                    "strict": true
                }
            },
            "max_tokens": 64
        });

        let plan = process_request(&config, "/v1/chat/completions", &[], body)
            .expect("request should transform");

        assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
        assert_eq!(plan.path, "/v1/responses");
        assert_eq!(plan.body.get("response_format"), None);
        assert_eq!(plan.body.get("max_tokens"), None);
        assert_eq!(plan.body.get("max_output_tokens"), Some(&json!(64)));
        assert_eq!(
            plan.body.get("text"),
            Some(&json!({
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"}
                        }
                    },
                    "strict": true
                }
            }))
        );
    }

    #[test]
    fn transforms_system_messages_into_gemini_system_instruction() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "to": "gemini_chat"
            }
        });
        let body = json!({
            "model": "gemini-2.5-pro",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {"role": "user", "content": "Hello"}
            ]
        });

        let plan = process_request(&config, "/v1/chat/completions", &[], body)
            .expect("request should transform");

        assert_eq!(plan.target_format, Some(RequestFormat::GeminiChat));
        assert_eq!(
            plan.body.get("systemInstruction"),
            Some(&json!({"parts": [{"text": "Be terse"}]}))
        );
        assert_eq!(
            plan.body.get("contents"),
            Some(&json!([
                {"role": "user", "parts": [{"text": "Hello"}]}
            ]))
        );
    }

    #[test]
    fn normalizes_openai_tool_choice_for_claude_requests() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "to": "claude_chat"
            }
        });
        let body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_weather"}
            },
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {"properties": {"city": {"type": "string"}}}
                }
            }]
        });

        let plan = process_request(&config, "/v1/chat/completions", &[], body)
            .expect("request should transform");

        assert_eq!(
            plan.body.get("tool_choice"),
            Some(&json!({"type": "tool", "name": "lookup_weather"}))
        );
        assert_eq!(
            plan.body.pointer("/tools/0/input_schema/type"),
            Some(&json!("object"))
        );
        assert_eq!(
            plan.body
                .pointer("/tools/0/input_schema/properties/city/type"),
            Some(&json!("string"))
        );
    }

    #[test]
    fn strict_parse_reports_format_mismatch_when_request_matches_excluded_format() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "to": "claude_chat",
                "strict_parse": true
            }
        });
        let body = json!({
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello"}]
            }]
        });

        let err = process_request(
            &config,
            "/v1beta/models/gemini-2.5-flash:generateContent",
            &[],
            body,
        )
        .expect_err("strict parse should reject mismatched format");

        match err {
            RequestProcessError::StrictParse(message) => {
                assert!(message.contains("Format mismatch"), "{message}");
                assert!(message.contains("gemini_chat"), "{message}");
                assert!(message.contains("openai_chat"), "{message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn strict_parse_reports_openai_chat_json_path_for_missing_messages() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "strict_parse": true
            }
        });
        let err = process_request(
            &config,
            "/v1/chat/completions",
            &[],
            json!({"model": "gpt-4.1-mini"}),
        )
        .expect_err("missing messages should be reported");

        match err {
            RequestProcessError::StrictParse(message) => {
                assert!(message.contains("JSON path '$.messages'"), "{message}");
                assert!(message.contains("missing required field"), "{message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn strict_parse_reports_openai_chat_json_types() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "openai_chat",
                "strict_parse": true
            }
        });
        let err = process_request(
            &config,
            "/v1/chat/completions",
            &[],
            json!({"messages": {}, "stream": "true"}),
        )
        .expect_err("invalid field types should be reported");

        match err {
            RequestProcessError::StrictParse(message) => {
                assert!(message.contains("JSON path '$.messages'"), "{message}");
                assert!(message.contains("expected an array, got object"), "{message}");
                assert!(message.contains("JSON path '$.stream'"), "{message}");
                assert!(message.contains("expected a boolean, got string"), "{message}");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn transforms_claude_thinking_into_openai_chat_reasoning() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "claude_chat",
                "to": "openai_chat"
            }
        });
        let body = json!({
            "model": "gpt-5",
            "thinking": {
                "type": "enabled",
                "budget_tokens": 2048
            },
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let plan =
            process_request(&config, "/v1/messages", &[], body).expect("request should transform");

        assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
        assert_eq!(plan.body.get("thinking"), None);
        assert_eq!(plan.body.get("reasoning_effort"), Some(&json!("low")));
    }

    #[test]
    fn strips_claude_only_extra_fields_for_openai_chat_requests() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "claude_chat",
                "to": "openai_chat"
            }
        });
        let body = json!({
            "model": "gpt-4.1-mini",
            "stream": false,
            "anthropic_version": "2023-06-01",
            "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]},
            "output_config": {"type": "json"},
            "stop_sequences": ["END"],
            "top_k": 20,
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let plan =
            process_request(&config, "/v1/messages", &[], body).expect("request should transform");

        assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
        assert_eq!(plan.path, "/v1/chat/completions");
        assert_eq!(plan.body.get("stream"), Some(&json!(false)));
        assert_eq!(plan.body.get("context_management"), None);
        assert_eq!(plan.body.get("output_config"), None);
        assert_eq!(plan.body.get("anthropic_version"), None);
        assert_eq!(plan.body.get("stop_sequences"), None);
        assert_eq!(plan.body.get("top_k"), None);
        assert_eq!(plan.body.get("stop"), Some(&json!(["END"])));
    }

    #[test]
    fn transforms_claude_thinking_into_openai_responses_reasoning() {
        let config = json!({
            "format_transform": {
                "enabled": true,
                "from": "claude_chat",
                "to": "openai_responses"
            }
        });
        let body = json!({
            "model": "gpt-5",
            "thinking": {
                "type": "enabled",
                "budget_tokens": 1024
            },
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let plan =
            process_request(&config, "/v1/messages", &[], body).expect("request should transform");

        assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
        assert_eq!(plan.body.get("thinking"), None);
        assert_eq!(plan.body.get("reasoning"), Some(&json!({"effort": "low"})));
    }
}
