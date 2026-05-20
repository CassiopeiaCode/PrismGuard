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
        (Some(RequestFormat::OpenAiChat), Some(RequestFormat::OpenAiResponses)) => {
            openai_chat_to_openai_responses_response(body)
        }
        (Some(RequestFormat::ClaudeChat), Some(RequestFormat::OpenAiResponses)) => {
            claude_to_openai_responses_response(body)
        }
        (Some(RequestFormat::GeminiChat), Some(RequestFormat::OpenAiResponses)) => {
            gemini_to_openai_responses_response(body)
        }
        (Some(RequestFormat::OpenAiChat), Some(RequestFormat::ClaudeChat)) => {
            openai_chat_to_claude_response(body)
        }
        (Some(RequestFormat::OpenAiResponses), Some(RequestFormat::OpenAiChat)) => {
            openai_responses_to_openai_chat(body)
        }
        _ => Ok(body),
    }
}

fn openai_chat_to_claude_response(body: Value) -> Result<Value, ApiError> {
    let object = body.as_object().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "response body is not a JSON object")
            .with_code("PROXY_ERROR")
    })?;

    let choice = object
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "chat response choices[0] is missing")
                .with_code("PROXY_ERROR")
        })?;

    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "chat response message is missing")
                .with_code("PROXY_ERROR")
        })?;

    let mut content = Vec::new();

    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            content.push(json!({
                "type": "text",
                "text": text,
            }));
        }
    } else if let Some(parts) = message.get("content").and_then(Value::as_array) {
        for part in parts {
            let Some(part) = part.as_object() else {
                continue;
            };
            match part.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        if !text.is_empty() {
                            content.push(json!({
                                "type": "text",
                                "text": text,
                            }));
                        }
                    }
                }
                Some("tool_use") => {
                    content.push(json!({
                        "type": "tool_use",
                        "id": part.get("id").cloned().unwrap_or_else(|| json!("")),
                        "name": part.get("name").cloned().unwrap_or_else(|| json!("")),
                        "input": part.get("input").cloned().unwrap_or_else(|| json!({})),
                    }));
                }
                _ => {}
            }
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls.iter().filter_map(Value::as_object) {
            let function = tool_call
                .get("function")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let input = function
                .get("arguments")
                .map(parse_tool_arguments)
                .unwrap_or_else(|| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.get("id").cloned().unwrap_or_else(|| json!("")),
                "name": function.get("name").cloned().unwrap_or_else(|| json!("")),
                "input": input,
            }));
        }
    }

    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            content.push(json!({
                "type": "thinking",
                "thinking": reasoning,
                "signature": ""
            }));
        }
    }

    if content.is_empty() {
        content.push(json!({
            "type": "text",
            "text": "",
        }));
    }

    let mut response = json!({
        "id": object.get("id").cloned().unwrap_or_else(|| json!("")),
        "model": object.get("model").cloned().unwrap_or_else(|| json!("")),
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": map_chat_finish_reason(choice.get("finish_reason").and_then(Value::as_str)),
        "stop_sequence": Value::Null,
    });

    if let Some(usage) = object
        .get("usage")
        .and_then(Value::as_object)
        .and_then(chat_usage_to_claude_usage)
    {
        response["usage"] = usage;
    }

    if let Some(response_obj) = response.as_object_mut() {
        for (key, value) in object {
            if matches!(key.as_str(), "id" | "model" | "choices" | "usage") {
                continue;
            }
            response_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(response)
}

fn openai_chat_to_openai_responses_response(body: Value) -> Result<Value, ApiError> {
    let object = body.as_object().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "response body is not a JSON object")
            .with_code("PROXY_ERROR")
    })?;

    let choice = object
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "chat response choices[0] is missing")
                .with_code("PROXY_ERROR")
        })?;

    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "chat response message is missing")
                .with_code("PROXY_ERROR")
        })?;

    let mut output = Vec::new();

    let message_content = chat_message_to_responses_content(message.get("content"));
    if !message_content.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("{}_msg_0", object.get("id").and_then(Value::as_str).unwrap_or("resp")),
            "role": "assistant",
            "content": message_content
        }));
    }

    if let Some(reasoning) = message.get("reasoning_content").and_then(Value::as_str) {
        if !reasoning.is_empty() {
            output.push(json!({
                "type": "reasoning",
                "id": format!("{}_reasoning_0", object.get("id").and_then(Value::as_str).unwrap_or("resp")),
                "summary": [{
                    "type": "summary_text",
                    "text": reasoning
                }],
                "content": [{
                    "type": "reasoning_text",
                    "text": reasoning
                }]
            }));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls.iter().filter_map(Value::as_object) {
            let function = tool_call
                .get("function")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            output.push(json!({
                "type": "function_call",
                "id": tool_call.get("id").cloned().unwrap_or_else(|| json!("")),
                "call_id": tool_call.get("id").cloned().unwrap_or_else(|| json!("")),
                "name": function.get("name").cloned().unwrap_or_else(|| json!("")),
                "arguments": function.get("arguments").cloned().unwrap_or_else(|| json!(""))
            }));
        }
    }

    let finish_reason = choice.get("finish_reason").and_then(Value::as_str);
    let status = match finish_reason {
        Some("length") => "incomplete",
        Some("error") => "failed",
        _ => "completed",
    };

    let mut response = json!({
        "id": object.get("id").cloned().unwrap_or_else(|| json!("")),
        "object": "response",
        "model": object.get("model").cloned().unwrap_or_else(|| json!("")),
        "created_at": object
            .get("created")
            .cloned()
            .or_else(|| object.get("created_at").cloned())
            .unwrap_or_else(|| json!(0)),
        "status": status,
        "output": output
    });

    if let Some(usage) = object.get("usage").and_then(Value::as_object) {
        response["usage"] = json!({
            "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or_else(|| json!(0)),
            "output_tokens": usage.get("completion_tokens").cloned().unwrap_or_else(|| json!(0)),
            "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0))
        });
    }

    if let Some(response_obj) = response.as_object_mut() {
        for (key, value) in object {
            if matches!(key.as_str(), "id" | "object" | "choices" | "model" | "created" | "created_at" | "usage") {
                continue;
            }
            response_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(response)
}

fn claude_to_openai_responses_response(body: Value) -> Result<Value, ApiError> {
    let object = body.as_object().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "response body is not a JSON object")
            .with_code("PROXY_ERROR")
    })?;

    let content_blocks = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "claude response content is missing")
                .with_code("PROXY_ERROR")
        })?;

    let response_id = object.get("id").and_then(Value::as_str).unwrap_or("resp");
    let mut output = Vec::new();
    let mut message_content = Vec::new();
    let mut reasoning_parts = Vec::new();

    for block in content_blocks.iter().filter_map(Value::as_object) {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        message_content.push(json!({
                            "type": "output_text",
                            "text": text
                        }));
                    }
                }
            }
            Some("thinking") => {
                if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                    if !text.is_empty() {
                        reasoning_parts.push(text.to_string());
                    }
                }
            }
            Some("tool_use") => {
                let arguments = stringify_function_arguments(
                    block.get("input").unwrap_or(&Value::Null),
                );
                output.push(json!({
                    "type": "function_call",
                    "id": block.get("id").cloned().unwrap_or_else(|| json!("")),
                    "call_id": block.get("id").cloned().unwrap_or_else(|| json!("")),
                    "name": block.get("name").cloned().unwrap_or_else(|| json!("")),
                    "arguments": arguments
                }));
            }
            _ => {}
        }
    }

    if !message_content.is_empty() {
        output.insert(0, json!({
            "type": "message",
            "id": format!("{response_id}_msg_0"),
            "role": "assistant",
            "content": message_content
        }));
    }

    if !reasoning_parts.is_empty() {
        let reasoning_text = reasoning_parts.join("\n");
        let insert_at = usize::from(!output.is_empty() && output[0].get("type").and_then(Value::as_str) == Some("message"));
        output.insert(insert_at, json!({
            "type": "reasoning",
            "id": format!("{response_id}_reasoning_0"),
            "summary": [{
                "type": "summary_text",
                "text": reasoning_text
            }],
            "content": [{
                "type": "reasoning_text",
                "text": reasoning_text
            }]
        }));
    }

    let stop_reason = object.get("stop_reason").and_then(Value::as_str);
    let status = match stop_reason {
        Some("max_tokens") => "incomplete",
        Some("error") => "failed",
        _ => "completed",
    };

    let mut response = json!({
        "id": object.get("id").cloned().unwrap_or_else(|| json!("")),
        "object": "response",
        "model": object.get("model").cloned().unwrap_or_else(|| json!("")),
        "created_at": numeric_response_timestamp(object.get("created_at")),
        "status": status,
        "output": output
    });

    if let Some(usage) = object.get("usage").and_then(Value::as_object) {
        let input_tokens = usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0));
        let output_tokens = usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0));
        let total_tokens = usage
            .get("total_tokens")
            .cloned()
            .unwrap_or_else(|| {
                json!(
                    input_tokens.as_i64().unwrap_or(0) + output_tokens.as_i64().unwrap_or(0)
                )
            });
        response["usage"] = json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        });
    }

    if let Some(response_obj) = response.as_object_mut() {
        for (key, value) in object {
            if matches!(
                key.as_str(),
                "id" | "type" | "role" | "content" | "model" | "created_at" | "stop_reason" | "usage"
            ) {
                continue;
            }
            response_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(response)
}

fn gemini_to_openai_responses_response(body: Value) -> Result<Value, ApiError> {
    let object = body.as_object().ok_or_else(|| {
        ApiError::new(StatusCode::BAD_GATEWAY, "response body is not a JSON object")
            .with_code("PROXY_ERROR")
    })?;

    let candidate = object
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ApiError::new(StatusCode::BAD_GATEWAY, "gemini response candidates[0] is missing")
                .with_code("PROXY_ERROR")
        })?;

    let response_id = object
        .get("responseId")
        .and_then(Value::as_str)
        .unwrap_or("gemini-response");
    let mut output = Vec::new();
    let mut message_content = Vec::new();
    let mut reasoning_parts = Vec::new();

    if let Some(parts) = candidate
        .get("content")
        .and_then(Value::as_object)
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
    {
        for part in parts.iter().filter_map(Value::as_object) {
            if part.get("thought").and_then(Value::as_bool) == Some(true) {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        reasoning_parts.push(text.to_string());
                    }
                }
                continue;
            }

            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    message_content.push(json!({
                        "type": "output_text",
                        "text": text
                    }));
                }
                continue;
            }

            if let Some(function_call) = part.get("functionCall").and_then(Value::as_object) {
                let call_id = function_call
                    .get("id")
                    .cloned()
                    .unwrap_or_else(|| json!("gemini_call"));
                output.push(json!({
                    "type": "function_call",
                    "id": call_id.clone(),
                    "call_id": call_id,
                    "name": function_call.get("name").cloned().unwrap_or_else(|| json!("")),
                    "arguments": stringify_function_arguments(
                        function_call.get("args").unwrap_or(&Value::Null)
                    )
                }));
            }
        }
    }

    if !message_content.is_empty() {
        output.insert(0, json!({
            "type": "message",
            "id": format!("{response_id}_msg_0"),
            "role": "assistant",
            "content": message_content
        }));
    }

    if !reasoning_parts.is_empty() {
        let reasoning_text = reasoning_parts.join("\n");
        let insert_at = usize::from(!output.is_empty() && output[0].get("type").and_then(Value::as_str) == Some("message"));
        output.insert(insert_at, json!({
            "type": "reasoning",
            "id": format!("{response_id}_reasoning_0"),
            "summary": [{
                "type": "summary_text",
                "text": reasoning_text
            }],
            "content": [{
                "type": "reasoning_text",
                "text": reasoning_text
            }]
        }));
    }

    let finish_reason = candidate.get("finishReason").and_then(Value::as_str);
    let status = match finish_reason {
        Some("MAX_TOKENS") => "incomplete",
        Some("ERROR") => "failed",
        _ => "completed",
    };

    let mut response = json!({
        "id": response_id,
        "object": "response",
        "model": object.get("modelVersion").cloned().unwrap_or_else(|| json!("gemini")),
        "created_at": numeric_response_timestamp(object.get("createTime")),
        "status": status,
        "output": output
    });

    if let Some(usage) = object.get("usageMetadata").and_then(Value::as_object) {
        let input_tokens = usage
            .get("promptTokenCount")
            .cloned()
            .unwrap_or_else(|| json!(0));
        let candidate_tokens = usage
            .get("candidatesTokenCount")
            .cloned()
            .unwrap_or_else(|| json!(0));
        let thought_tokens = usage
            .get("thoughtsTokenCount")
            .cloned()
            .unwrap_or_else(|| json!(0));
        let output_tokens = json!(
            candidate_tokens.as_i64().unwrap_or(0) + thought_tokens.as_i64().unwrap_or(0)
        );
        let total_tokens = usage
            .get("totalTokenCount")
            .cloned()
            .unwrap_or_else(|| {
                json!(
                    input_tokens.as_i64().unwrap_or(0) + output_tokens.as_i64().unwrap_or(0)
                )
            });
        response["usage"] = json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        });
    }

    if let Some(response_obj) = response.as_object_mut() {
        for (key, value) in object {
            if matches!(
                key.as_str(),
                "candidates" | "responseId" | "modelVersion" | "createTime" | "usageMetadata"
            ) {
                continue;
            }
            response_obj.insert(key.clone(), value.clone());
        }
    }

    Ok(response)
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
                    message.insert("reasoning_content".to_string(), json!(text_content));
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

fn chat_message_to_responses_content(content: Option<&Value>) -> Vec<Value> {
    match content {
        Some(Value::String(text)) if !text.is_empty() => {
            vec![json!({
                "type": "output_text",
                "text": text
            })]
        }
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                let part = part.as_object()?;
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => part.get("text").and_then(Value::as_str).map(|text| {
                        json!({
                            "type": "output_text",
                            "text": text
                        })
                    }),
                    Some("image_url") => {
                        let image_url = part.get("image_url")?;
                        Some(json!({
                            "type": "image_url",
                            "image_url": image_url
                        }))
                    }
                    _ => None,
                }
            })
            .collect(),
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

fn map_chat_finish_reason(finish_reason: Option<&str>) -> Value {
    match finish_reason {
        Some("stop") | None => json!("end_turn"),
        Some("length") => json!("max_tokens"),
        Some("tool_calls") => json!("tool_use"),
        Some(other) => json!(other),
    }
}

fn chat_usage_to_claude_usage(usage: &Map<String, Value>) -> Option<Value> {
    Some(json!({
        "input_tokens": usage.get("prompt_tokens").cloned().unwrap_or_else(|| usage.get("input_tokens").cloned().unwrap_or_else(|| json!(0))),
        "output_tokens": usage.get("completion_tokens").cloned().unwrap_or_else(|| usage.get("output_tokens").cloned().unwrap_or_else(|| json!(0))),
        "total_tokens": usage.get("total_tokens").cloned().unwrap_or_else(|| json!(0)),
    }))
}

fn parse_tool_arguments(arguments: &Value) -> Value {
    match arguments {
        Value::String(raw) => serde_json::from_str(raw).unwrap_or_else(|_| json!({})),
        Value::Object(_) | Value::Array(_) => arguments.clone(),
        _ => json!({}),
    }
}

fn numeric_response_timestamp(value: Option<&Value>) -> Value {
    match value {
        Some(Value::Number(number)) => Value::Number(number.clone()),
        Some(Value::String(text)) => text.parse::<i64>().map_or_else(|_| json!(0), |value| json!(value)),
        _ => json!(0),
    }
}
