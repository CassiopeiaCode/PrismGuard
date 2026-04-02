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
        .or_else(|| object.get("created").cloned())
        .unwrap_or_else(|| json!(0));

    let output_items = object
        .get("output")
        .and_then(Value::as_array)
        .and_then(|items| items.last());

    let mut message = Map::new();
    message.insert("role".to_string(), json!("assistant"));
    if let Some(last_item) = output_items {
        match last_item.get("type").and_then(Value::as_str) {
            Some("message") => {
                let content_parts = last_item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
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
                message.insert(
                    "tool_calls".to_string(),
                    json!([{
                        "id": last_item
                            .get("call_id")
                            .or_else(|| last_item.get("id"))
                            .cloned()
                            .unwrap_or_else(|| json!("")),
                        "type": "function",
                        "function": {
                            "name": last_item.get("name").cloned().unwrap_or_else(|| json!("")),
                            "arguments": last_item
                                .get("arguments")
                                .cloned()
                                .unwrap_or_else(|| json!(""))
                        }
                    }]),
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
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": map_finish_reason(object.get("status").and_then(Value::as_str))
        }]
    });

    if let Some(usage) = object.get("usage").and_then(Value::as_object) {
        response["usage"] = json!({
            "prompt_tokens": usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0)),
            "completion_tokens": usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0)),
            "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
            "responses_usage": Value::Object(usage.clone()),
        });
    }

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

fn map_finish_reason(status: Option<&str>) -> &'static str {
    match status.unwrap_or_default().to_ascii_lowercase().as_str() {
        "incomplete" => "length",
        "failed" => "error",
        _ => "stop",
    }
}
