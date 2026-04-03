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

    if let Some(system) = body.get("system").and_then(Value::as_str) {
        if !system.is_empty() {
            texts.push(system.to_string());
        }
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for msg in messages {
            match msg.get("content") {
                Some(Value::String(text)) => texts.push(text.to_string()),
                Some(Value::Array(parts)) => {
                    for part in parts {
                        if part.get("type").and_then(Value::as_str) == Some("text") {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                texts.push(text.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    texts.join("\n")
}

fn extract_from_openai_response(body: &Value) -> String {
    if let Some(prompt) = body.get("prompt") {
        match prompt {
            Value::String(text) => return text.to_string(),
            Value::Array(items) => {
                return items
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            _ => {}
        }
    }

    if let Some(input) = body.get("input").and_then(Value::as_str) {
        return input.to_string();
    }

    if body.get("messages").is_some() {
        return extract_from_openai_chat(body);
    }

    String::new()
}
