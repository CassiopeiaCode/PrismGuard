use serde_json::Value;

pub fn extract_text_for_moderation(body: &Value, request_format: &str) -> String {
    match request_format {
        "claude_chat" => extract_from_claude_chat(body),
        "openai_response" | "openai_responses" => extract_from_openai_response(body),
        _ => extract_from_openai_chat(body),
    }
}

fn extract_from_openai_chat(body: &Value) -> String {
    body.get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|msg| msg.get("role").and_then(Value::as_str) != Some("tool"))
        .filter_map(|msg| msg.get("content"))
        .flat_map(|content| match content {
            Value::String(text) => vec![text.to_string()],
            Value::Array(parts) => parts
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|part| part.get("text").and_then(Value::as_str).map(ToString::to_string))
                .collect::<Vec<_>>(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_from_claude_chat(body: &Value) -> String {
    let mut texts = Vec::new();

    collect_claude_text_parts(body.get("system"), &mut texts);

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for msg in messages {
            collect_claude_text_parts(msg.get("content"), &mut texts);
        }
    }

    texts.join("\n")
}

fn extract_from_openai_response(body: &Value) -> String {
    let mut texts = Vec::new();

    collect_non_empty_text(body.get("instructions"), &mut texts);

    if let Some(prompt) = body.get("prompt") {
        match prompt {
            Value::String(_) | Value::Array(_) => collect_non_empty_text(Some(prompt), &mut texts),
            _ => {}
        }
    }

    collect_openai_responses_input(body.get("input"), &mut texts);

    if !texts.is_empty() {
        return texts.join("\n");
    }

    if body.get("messages").is_some() {
        return extract_from_openai_chat(body);
    }

    String::new()
}

fn collect_claude_text_parts(value: Option<&Value>, texts: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) => push_if_not_empty(text, texts),
        Some(Value::Array(parts)) => {
            for part in parts {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    collect_non_empty_text(part.get("text"), texts);
                }
            }
        }
        _ => {}
    }
}

fn collect_openai_responses_input(value: Option<&Value>, texts: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) => push_if_not_empty(text, texts),
        Some(Value::Array(items)) => {
            for item in items {
                collect_openai_responses_item(item, texts);
            }
        }
        Some(Value::Object(_)) => collect_openai_responses_item(value.expect("checked Some"), texts),
        _ => {}
    }
}

fn collect_openai_responses_item(value: &Value, texts: &mut Vec<String>) {
    let Some(item) = value.as_object() else {
        collect_non_empty_text(Some(value), texts);
        return;
    };

    if let Some(content) = item.get("content") {
        collect_openai_responses_content(content, texts);
    }

    if item.get("content").is_none() {
        collect_non_empty_text(item.get("text"), texts);
    }

    if item.get("type").and_then(Value::as_str) == Some("function_call_output") {
        collect_non_empty_text(item.get("output"), texts);
    }
}

fn collect_openai_responses_content(value: &Value, texts: &mut Vec<String>) {
    match value {
        Value::String(text) => push_if_not_empty(text, texts),
        Value::Array(parts) => {
            for part in parts {
                let part_type = part.get("type").and_then(Value::as_str);
                match part_type {
                    Some("text") | Some("input_text") | Some("output_text") => {
                        collect_non_empty_text(part.get("text"), texts);
                    }
                    Some("function_call_output") => {
                        collect_non_empty_text(part.get("output"), texts);
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn collect_non_empty_text(value: Option<&Value>, texts: &mut Vec<String>) {
    match value {
        Some(Value::String(text)) => push_if_not_empty(text, texts),
        Some(Value::Array(values)) => {
            for value in values {
                collect_non_empty_text(Some(value), texts);
            }
        }
        _ => {}
    }
}

fn push_if_not_empty(text: &str, texts: &mut Vec<String>) {
    if !text.is_empty() {
        texts.push(text.to_string());
    }
}
