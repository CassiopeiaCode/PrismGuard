mod format_harness;

use format_harness::format::{process_request, RequestFormat, RequestProcessError};
use serde_json::{json, Value};

fn transform_config(strict_parse: bool, to: &str) -> Value {
    json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": strict_parse,
            "to": to
        }
    })
}

#[test]
fn detects_openai_chat_and_rewrites_path_for_responses_target() {
    let plan = process_request(
        &transform_config(true, "openai_responses"),
        "/proxy/openai/v1/chat/completions",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Ping"}
            ]
        }),
    )
    .expect("openai chat request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
    assert!(plan.stream);
    assert_eq!(plan.path, "/proxy/openai/v1/responses");
    assert_eq!(plan.moderation_text.as_deref(), Some("Be terse.\nPing"));
    assert_eq!(plan.body["instructions"], "Be terse.");
    assert_eq!(plan.body["input"][0]["role"], "user");
    assert_eq!(plan.body["input"][0]["content"][0]["text"], "Ping");
}

#[test]
fn preserves_moderation_text_when_chat_transforms_into_responses_instructions() {
    let plan = process_request(
        &transform_config(true, "openai_responses"),
        "/v1/chat/completions",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "stream": false,
            "messages": [
                {"role": "system", "content": "forbidden system text"},
                {"role": "user", "content": "safe user text"}
            ]
        }),
    )
    .expect("openai chat request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(
        plan.moderation_text.as_deref(),
        Some("forbidden system text\nsafe user text")
    );
    assert_eq!(plan.body["instructions"], "forbidden system text");
}

#[test]
fn preserves_moderation_text_for_native_openai_responses_requests() {
    let plan = process_request(
        &transform_config(true, "claude_chat"),
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "instructions": "forbidden instruction text",
            "input": "safe user text"
        }),
    )
    .expect("responses request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(
        plan.moderation_text.as_deref(),
        Some("forbidden instruction text\nsafe user text")
    );
}

#[test]
fn detects_claude_chat_from_headers_and_rewrites_path_for_openai_chat_target() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/relay/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [
                {"role": "user", "content": "Summarize this"}
            ]
        }),
    )
    .expect("claude request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/relay/v1/chat/completions");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"], "Summarize this");
}

#[test]
fn detects_openai_responses_and_rewrites_path_for_claude_target() {
    let plan = process_request(
        &transform_config(true, "claude_chat"),
        "/edge/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1",
            "instructions": "Stay factual.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Hello"}
                    ]
                }
            ]
        }),
    )
    .expect("responses request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/edge/v1/messages");
    assert_eq!(plan.body["system"], "Stay factual.");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"][0]["text"], "Hello");
}

#[test]
fn detects_gemini_chat_and_rewrites_streaming_model_path() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/proxy/google/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        &[],
        json!({
            "model": "gemini-2.5-flash",
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "Hi"}
                    ]
                }
            ]
        }),
    )
    .expect("gemini request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert!(plan.stream);
    assert_eq!(plan.path, "/proxy/google/v1/chat/completions");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"], "Hi");
}

#[test]
fn strict_parse_returns_error_when_detection_fails() {
    let error = process_request(
        &transform_config(true, "openai_chat"),
        "/unknown",
        &[],
        json!({
            "foo": "bar"
        }),
    )
    .expect_err("strict mode should reject undetected format");

    match error {
        RequestProcessError::StrictParse(message) => {
            assert!(message.contains("unable to detect request format"));
        }
        other => panic!("expected strict parse error, got {other:?}"),
    }
}

#[test]
fn strict_parse_reports_detected_but_disallowed_format() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_chat"
        }
    });

    let error = process_request(
        &config,
        "/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .expect_err("strict mode should reject mismatched formats");

    match error {
        RequestProcessError::StrictParse(message) => {
            assert!(message.contains("Format mismatch"));
            assert!(message.contains("claude_chat"));
            assert!(message.contains("openai_chat"));
        }
        other => panic!("expected strict parse error, got {other:?}"),
    }
}

#[test]
fn preserves_business_prefix_when_rewriting_path() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/secret_endpoint/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.path, "/secret_endpoint/v1/chat/completions");
}

#[test]
fn non_strict_mode_preserves_request_when_detection_fails() {
    let original_body = json!({
        "foo": "bar"
    });
    let plan = process_request(
        &transform_config(false, "openai_chat"),
        "/unknown",
        &[],
        original_body.clone(),
    )
    .expect("non-strict mode should pass through undetected requests");

    assert_eq!(plan.source_format, None);
    assert_eq!(plan.target_format, None);
    assert_eq!(plan.path, "/unknown");
    assert_eq!(plan.body, original_body);
    assert!(!plan.stream);
}

#[test]
fn disable_tools_strips_openai_responses_tool_fields_without_format_change() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "disable_tools": true,
            "from": "openai_responses",
            "to": "openai_responses"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1",
            "tools": [{
                "type": "function",
                "name": "lookup_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }],
            "tool_choice": "required",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup_weather",
                    "arguments": "{\"city\":\"Shanghai\"}"
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "name": "lookup_weather",
                    "output": "{\"temp\":20}"
                }
            ]
        }),
    )
    .expect("responses request should still parse");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.path, "/v1/responses");
    assert!(plan.body.get("tools").is_none());
    assert!(plan.body.get("tool_choice").is_none());
    assert_eq!(
        plan.body["input"],
        json!([{
            "role": "assistant",
            "type": "message",
            "content": [{
                "type": "input_text",
                "text": ""
            }]
        }])
    );
}

#[test]
fn disable_tools_strips_claude_tool_fields_without_format_change() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "disable_tools": true,
            "from": "claude_chat",
            "to": "claude_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "tools": [{
                "name": "lookup_weather",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    }
                }
            }],
            "tool_choice": {"type": "tool", "name": "lookup_weather"},
            "messages": [{
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "lookup_weather",
                        "input": {"city": "Shanghai"}
                    }
                ]
            }, {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "20C"
                }]
            }]
        }),
    )
    .expect("claude request should still parse");

    assert_eq!(plan.source_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/v1/messages");
    assert!(plan.body.get("tools").is_none());
    assert!(plan.body.get("tool_choice").is_none());
    assert_eq!(
        plan.body["messages"],
        json!([
            {"role": "assistant", "content": [{"type": "text", "text": ""}]},
            {"role": "user", "content": [{"type": "text", "text": ""}]}
        ])
    );
}

#[test]
fn disable_tools_strips_gemini_tool_fields_without_format_change() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "disable_tools": true,
            "from": "gemini_chat",
            "to": "gemini_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1beta/models/gemini-2.5-flash:generateContent",
        &[],
        json!({
            "model": "gemini-2.5-flash",
            "contents": [{
                "role": "model",
                "parts": [{
                    "functionCall": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "args": {"city": "Shanghai"}
                    }
                }]
            }, {
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "response": {"temp": 20}
                    }
                }]
            }],
            "tools": [{
                "functionDeclarations": [{
                    "name": "lookup_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }]
            }],
            "toolConfig": {
                "functionCallingConfig": {
                    "mode": "ANY"
                }
            },
            "generationConfig": {
                "temperature": 0.2
            }
        }),
    )
    .expect("gemini request should still parse");

    assert_eq!(plan.source_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.path, "/v1beta/models/gemini-2.5-flash:generateContent");
    assert!(plan.body.get("tools").is_none());
    assert!(plan.body.get("toolConfig").is_none());
    assert_eq!(plan.body["contents"], json!([]));
    assert_eq!(plan.body["generationConfig"], json!({"temperature": 0.2}));
}

#[test]
fn openai_chat_system_message_maps_to_claude_system_string_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "claude_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/chat/completions",
        &[],
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "Be terse"},
                {"role": "system", "content": "Stay factual"},
                {"role": "user", "content": "Hello"}
            ]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/v1/messages");
    assert_eq!(plan.body["system"], "Be terse\nStay factual");
    assert_eq!(
        plan.body["messages"],
        json!([
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ])
    );
}

#[test]
fn openai_responses_tool_items_include_completed_status_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    });

    let plan = process_request(
        &config,
        "/v1/chat/completions",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_weather",
                            "arguments": "{\"city\":\"Shanghai\"}"
                        }
                    }]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "name": "lookup_weather",
                    "content": "{\"temp\":20}"
                }
            ]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.path, "/v1/responses");
    assert_eq!(
        plan.body["input"],
        json!([
            {
                "type": "function_call",
                "id": "call_1",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": "{\"city\":\"Shanghai\"}",
                "status": "completed"
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "name": "lookup_weather",
                "output": "{\"temp\":20}",
                "status": "completed"
            }
        ])
    );
}

#[test]
fn gemini_tool_result_maps_to_openai_chat_string_content_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "gemini_chat",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1beta/models/gemini-2.5-flash:generateContent",
        &[],
        json!({
            "model": "gemini-2.5-flash",
            "contents": [{
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "id": "call_1",
                        "name": "lookup_weather",
                        "response": {"temp": 20}
                    }
                }]
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(plan.body["messages"][0]["role"], "user");
    assert_eq!(plan.body["messages"][0]["content"], "");
    assert_eq!(plan.body["messages"][1]["role"], "tool");
    assert_eq!(plan.body["messages"][1]["tool_call_id"], "call_1");
    assert_eq!(plan.body["messages"][1]["name"], "lookup_weather");
    assert_eq!(plan.body["messages"][1]["content"], "{\"temp\":20}");
}

#[test]
fn openai_chat_tool_result_maps_to_claude_text_tool_result_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "claude_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/chat/completions",
        &[],
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lookup_weather",
                "content": "{\"temp\":20}"
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/v1/messages");
    assert_eq!(
        plan.body["messages"],
        json!([
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": [{"type": "text", "text": "{\"temp\":20}"}]
                }]
            }
        ])
    );
}

#[test]
fn openai_responses_message_content_items_object_parses_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": [{
                "type": "message",
                "role": "user",
                "content": {
                    "items": [
                        {"type": "input_text", "text": "Describe this image"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/cat.png",
                            "detail": "high"
                        }
                    ]
                }
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/cat.png",
                        "detail": "high"
                    }
                }
            ]
        }])
    );
}

#[test]
fn gemini_system_instruction_maps_to_openai_chat_system_message_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "gemini_chat",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1beta/models/gemini-2.5-flash:generateContent",
        &[],
        json!({
            "model": "gemini-2.5-flash",
            "systemInstruction": {
                "parts": [{"text": "Be terse"}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": "Hello"}]
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::GeminiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "Hello"}
        ])
    );
}

#[test]
fn openai_responses_reasoning_item_maps_to_openai_chat_assistant_text_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": [{
                "type": "reasoning",
                "summary": [
                    {"text": "Need weather lookup"},
                    {"text": "Then summarize briefly"}
                ]
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "assistant",
            "content": "Need weather lookup\nThen summarize briefly"
        }])
    );
}

#[test]
fn openai_chat_tools_normalize_claude_input_schema_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "claude_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/chat/completions",
        &[],
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "parameters": {
                        "properties": {
                            "city": {"type": "string"}
                        }
                    }
                }
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/v1/messages");
    assert_eq!(
        plan.body.pointer("/tools/0/input_schema/type"),
        Some(&json!("object"))
    );
    assert_eq!(
        plan.body.pointer("/tools/0/input_schema/properties/city/type"),
        Some(&json!("string"))
    );
}

#[test]
fn openai_responses_preserves_extra_generation_fields_when_mapping_to_openai_chat() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "temperature": 0.3,
            "top_p": 0.8,
            "metadata": {"trace_id": "abc"},
            "response_format": {"type": "json_object"},
            "input": "Hello"
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(plan.body["temperature"], json!(0.3));
    assert_eq!(plan.body["top_p"], json!(0.8));
    assert_eq!(plan.body["metadata"], json!({"trace_id": "abc"}));
    assert_eq!(plan.body["response_format"], json!({"type": "json_object"}));
}

#[test]
fn openai_responses_single_input_object_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello from dict"}]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": "Hello from dict"
        }])
    );
}

#[test]
fn openai_responses_message_text_field_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "text": "Hello from message.text"
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": "Hello from message.text"
        }])
    );
}

#[test]
fn openai_responses_input_image_flat_url_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "url": "https://example.com/flat-input.png",
                    "detail": "high"
                }]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/flat-input.png",
                    "detail": "high"
                }
            }]
        }])
    );
}

#[test]
fn openai_responses_input_image_nested_image_url_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image": {
                        "url": "https://example.com/nested-input.png"
                    },
                    "detail": "low"
                }]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/nested-input.png",
                    "detail": "low"
                }
            }]
        }])
    );
}

#[test]
fn openai_responses_image_url_part_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": "https://example.com/direct-image-url.png",
                    "detail": "auto"
                }]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/direct-image-url.png",
                    "detail": "auto"
                }
            }]
        }])
    );
}

#[test]
fn openai_responses_image_url_object_part_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/direct-image-url-object.png",
                        "detail": "high"
                    }
                }]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/direct-image-url-object.png",
                    "detail": "high"
                }
            }]
        }])
    );
}

#[test]
fn openai_responses_image_url_nested_image_url_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": {
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image": {
                        "url": "https://example.com/direct-image-nested.png"
                    },
                    "detail": "low"
                }]
            }
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/direct-image-nested.png",
                    "detail": "low"
                }
            }]
        }])
    );
}

#[test]
fn openai_responses_function_call_output_content_items_map_to_openai_chat_tool_text_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "name": "lookup_weather",
                "output": {
                    "items": [
                        {"type": "output_text", "text": "Sunny"},
                        {"type": "output_text", "text": "25C"}
                    ]
                }
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([
            {"role": "tool", "tool_call_id": "call_1", "name": "lookup_weather", "content": "Sunny\n25C"}
        ])
    );
}

#[test]
fn openai_responses_function_call_output_single_object_text_maps_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_responses",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/responses",
        &[],
        json!({
            "model": "gpt-4.1-mini",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_obj_text_1",
                "name": "lookup_weather",
                "output": {
                    "type": "output_text",
                    "text": "Sunny from object"
                }
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiResponses));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([
            {"role": "tool", "tool_call_id": "call_obj_text_1", "name": "lookup_weather", "content": "Sunny from object"}
        ])
    );
}

#[test]
fn openai_chat_tool_result_array_maps_to_claude_array_content_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "claude_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/chat/completions",
        &[],
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "lookup_weather",
                "content": [
                    {"type": "text", "text": "Sunny"},
                    {"type": "text", "text": "25C"}
                ]
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.target_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.path, "/v1/messages");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": [
                    {"type": "text", "text": "Sunny"},
                    {"type": "text", "text": "25C"}
                ]
            }]
        }])
    );
}

#[test]
fn claude_tool_use_non_object_input_maps_to_openai_chat_empty_arguments_like_python() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "claude_chat",
            "to": "openai_chat"
        }
    });

    let plan = process_request(
        &config,
        "/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "lookup_weather",
                    "input": ["Shanghai"]
                }]
            }]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.source_format, Some(RequestFormat::ClaudeChat));
    assert_eq!(plan.target_format, Some(RequestFormat::OpenAiChat));
    assert_eq!(plan.path, "/v1/chat/completions");
    assert_eq!(
        plan.body["messages"],
        json!([{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup_weather",
                    "arguments": "{}"
                }
            }]
        }])
    );
}
