use axum::http::StatusCode;
use serde_json::{json, Map, Value};

use crate::format::RequestFormat;
use crate::routes::ApiError;

pub fn maybe_transform_json_response(
    body: Value,
    upstream_format: Option<RequestFormat>,
    client_format: Option<RequestFormat>,
) -> Result<Value, ApiError> {
    match (upstream_format, client_format) {
        (Some(RequestFormat::OpenAiResponses), Some(RequestFormat::OpenAiChat)) => {
            openai_responses_to_openai_chat(body)
        }
        _ => Ok(body),
    }
}

fn openai_responses_to_openai_chat(body: Value) -> Result<Value, ApiError> {
    let object = body.as_object().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "response body is not a JSON object")
            .with_code("PROXY_ERROR")
    })?;

    let id = object.get("id").cloned().unwrap_or_else(|| json!(""));
    let model = object.get("model").cloned().unwrap_or_else(|| json!(""));
    let created = object
        .get("created_at")
        .cloned()
        .or_else(|| object.get("created").cloned());

    let output_items = object
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let last_relevant_index = output_items.iter().rposition(|item| {
        matches!(
            item.get("type").and_then(Value::as_str),
            Some("message") | Some("function_call") | Some("reasoning")
        )
    });

    let mut message = Map::new();
    message.insert("role".to_string(), json!("assistant"));
    if let Some(last_relevant_index) = last_relevant_index {
        let last_item = &output_items[last_relevant_index];
        match last_item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content_parts = response_message_parts(last_item.get("content"))
                    .into_iter()
                    .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => part.get("text").and_then(Value::as_str).map(|text| {
                            json!({
                                "type": "text",
                                "text": text
                            })
                        }),
                        Some("input_image") | Some("image_url") => part
                            .get("image_url")
                            .and_then(|image_url_value| {
                                if let Some(url) = image_url_value.as_str() {
                                    let mut image_url = Map::new();
                                    image_url.insert("url".to_string(), json!(url));
                                    if let Some(detail) = part.get("detail").cloned() {
                                        image_url.insert("detail".to_string(), detail);
                                    }
                                    return Some(json!({
                                        "type": "image_url",
                                        "image_url": image_url
                                    }));
                                }

                                image_url_value.as_object().and_then(|image_url_obj| {
                                    image_url_obj.get("url").and_then(Value::as_str).map(|url| {
                                        let mut image_url = Map::new();
                                        image_url.insert("url".to_string(), json!(url));
                                        if let Some(detail) = image_url_obj.get("detail").cloned() {
                                            image_url.insert("detail".to_string(), detail);
                                        }
                                        json!({
                                            "type": "image_url",
                                            "image_url": image_url
                                        })
                                    })
                                })
                            })
                            .or_else(|| {
                                part.get("url").and_then(Value::as_str).map(|url| {
                                    let mut image_url = Map::new();
                                    image_url.insert("url".to_string(), json!(url));
                                    if let Some(detail) = part.get("detail").cloned() {
                                        image_url.insert("detail".to_string(), detail);
                                    }
                                    json!({
                                        "type": "image_url",
                                        "image_url": image_url
                                    })
                                })
                            })
                            .or_else(|| {
                                part.get("image")
                                    .and_then(Value::as_object)
                                    .and_then(|image| image.get("url"))
                                    .and_then(Value::as_str)
                                    .map(|url| {
                                        let mut image_url = Map::new();
                                        image_url.insert("url".to_string(), json!(url));
                                        if let Some(detail) = part.get("detail").cloned() {
                                            image_url.insert("detail".to_string(), detail);
                                        }
                                        json!({
                                            "type": "image_url",
                                            "image_url": image_url
                                        })
                                    })
                            }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();

                let has_image_parts = content_parts
                    .iter()
                    .any(|part| part.get("type").and_then(Value::as_str) == Some("image_url"));
                if has_image_parts {
                    message.insert("content".to_string(), Value::Array(content_parts));
                } else {
                    let text_content = content_parts
                        .iter()
                        .filter(|part| part.get("type").and_then(Value::as_str) == Some("text"))
                        .filter_map(|part| part.get("text").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text_content.is_empty() {
                        message.insert("content".to_string(), json!(text_content));
                    }
                }
            }
            Some("function_call") => {
                let start_index = output_items[..=last_relevant_index]
                    .iter()
                    .rposition(|item| item.get("type").and_then(Value::as_str) != Some("function_call"))
                    .map(|index| index + 1)
                    .unwrap_or(0);
                let tool_calls = output_items[start_index..=last_relevant_index]
                    .iter()
                    .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
                    .map(|item| {
                        let arguments = item
                            .get("arguments")
                            .map(stringify_function_arguments)
                            .unwrap_or_else(|| json!(""));
                        json!({
                            "id": item
                                .get("call_id")
                                .or_else(|| item.get("id"))
                                .cloned()
                                .unwrap_or_else(|| json!("")),
                            "type": "function",
                            "function": {
                                "name": item.get("name").cloned().unwrap_or_else(|| json!("")),
                                "arguments": arguments
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                message.insert(
                    "tool_calls".to_string(),
                    Value::Array(tool_calls),
                );
            }
            Some("reasoning") => {
                let text_content = last_item
                    .get("summary")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|summary| summary.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text_content.is_empty() {
                    message.insert("content".to_string(), json!(text_content));
                }
            }
            _ => {}
        }
    }

    let mut response = json!({
        "id": id,
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": map_finish_reason(object.get("status").and_then(Value::as_str))
        }]
    });

    if let Some(created) = created.filter(|value| !value.is_null()) {
        response["created"] = created;
    }

    response["usage"] = object
        .get("usage")
        .and_then(Value::as_object)
        .map(|usage| {
            json!({
                "prompt_tokens": usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0)),
                "completion_tokens": usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0)),
                "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
                "responses_usage": Value::Object(usage.clone()),
            })
        })
        .unwrap_or(Value::Null);

    if let Some(response_obj) = response.as_object_mut() {
        for (key, value) in object {
            if matches!(
                key.as_str(),
                "id" | "object" | "output" | "model" | "created_at" | "created" | "status" | "usage"
            ) {
                continue;
            }
            response_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(response)
}

fn response_message_parts(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::Array(parts)) => parts.clone(),
        Some(Value::Object(obj)) => {
            if let Some(items) = obj.get("items").and_then(Value::as_array) {
                items.clone()
            } else {
                vec![Value::Object(obj.clone())]
            }
        }
        _ => Vec::new(),
    }
}

fn stringify_function_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::String(_) => arguments.clone(),
        _ => Value::String(serde_json::to_string(arguments).unwrap_or_else(|_| String::new())),
    }
}

fn map_finish_reason(status: Option<&str>) -> Value {
    match status.map(str::to_ascii_lowercase).as_deref() {
        Some("completed") => json!("stop"),
        Some("incomplete") => json!("length"),
        Some("failed") => json!("error"),
        _ => Value::Null,
    }
}
