#![allow(dead_code)]

#[path = "../src/config.rs"]
mod config;
#[path = "../src/format.rs"]
mod format;
#[path = "../src/moderation/mod.rs"]
mod moderation;
#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/proxy.rs"]
mod proxy;
#[path = "../src/response.rs"]
mod response;
#[path = "../src/routes.rs"]
mod routes;
#[path = "../src/sample_rpc.rs"]
mod sample_rpc;
#[cfg(feature = "storage-debug")]
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/streaming.rs"]
mod streaming;
#[path = "../src/training.rs"]
mod training;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use config::Settings;
use routes::{router as proxy_router, AppState};
use serde_json::{json, Value};

#[tokio::test]
async fn upstream_openai_responses_json_is_transformed_back_to_openai_chat() {
    let upstream_body = json!({
        "id": "resp_123",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
}

#[tokio::test]
async fn pass_through_non_stream_response_keeps_plain_text_body() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "text/plain; charset=utf-8",
        b"plain text body".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "pass_through"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; charset=utf-8")
    );
    assert_eq!(
        response.text().await.expect("response text"),
        "plain text body"
    );
}

#[tokio::test]
async fn upstream_openai_chat_json_is_transformed_back_to_openai_responses() {
    let upstream_body = json!({
        "id": "chatcmpl_resp_123",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "hello"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 5,
            "total_tokens": 8
        }
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_responses",
                "to": "openai_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/responses"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "input": "hi"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["role"], "assistant");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "hello");
    assert_eq!(body["usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["output_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn upstream_openai_chat_tool_calls_are_transformed_back_to_openai_responses_function_calls() {
    let upstream_body = json!({
        "id": "chatcmpl_tool_resp",
        "object": "chat.completion",
        "created": 1710000001,
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": "{\"city\":\"Paris\"}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_responses",
                "to": "openai_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/responses"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "input": "weather?"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["output"][0]["type"], "function_call");
    assert_eq!(body["output"][0]["call_id"], "call_1");
    assert_eq!(body["output"][0]["name"], "lookup_weather");
    assert_eq!(body["output"][0]["arguments"], "{\"city\":\"Paris\"}");
}

#[tokio::test]
async fn upstream_openai_chat_reasoning_content_is_transformed_back_to_openai_responses_reasoning_item(
) {
    let upstream_body = json!({
        "id": "chatcmpl_reasoning_resp",
        "object": "chat.completion",
        "created": 1710000002,
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "step one"
            },
            "finish_reason": "stop"
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_responses",
                "to": "openai_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/responses"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "input": "think"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["output"][0]["type"], "reasoning");
    assert_eq!(body["output"][0]["summary"][0]["type"], "summary_text");
    assert_eq!(body["output"][0]["summary"][0]["text"], "step one");
    assert_eq!(body["output"][0]["content"][0]["type"], "reasoning_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "step one");
}

#[tokio::test]
async fn upstream_openai_chat_reasoning_response_transforms_to_claude_thinking_content() {
    let upstream_body = json!({
        "id": "chatcmpl_reasoning",
        "object": "chat.completion",
        "model": "gpt-4.1-mini",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "reasoning_content": "step one"
            },
            "finish_reason": "stop"
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "claude_chat",
                "to": "openai_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/messages"))
        .json(&json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(body["content"][0]["thinking"], "step one");
    assert_eq!(body["content"][0]["signature"], "");
}

#[tokio::test]
async fn upstream_openai_chat_tool_response_transforms_to_canonical_claude_message() {
    let upstream_body = json!({
        "id": "chatcmpl_claude_tool_resp",
        "object": "chat.completion",
        "created": 1710000010,
        "model": "gpt-5.5",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "我来查看当前目录位置和下面的文件。 \n",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "arguments": "{\"command\":\"pwd\",\"dangerouslyDisableSandbox\":false,\"description\":\"Show current working directory\",\"run_in_background\":false,\"timeout\":120000}"
                    }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {
            "prompt_tokens": 19859,
            "completion_tokens": 118,
            "total_tokens": 19977,
            "prompt_tokens_details": {
                "cached_creation_tokens": 0,
                "cached_tokens": 8704
            }
        }
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "claude_chat",
                "to": "openai_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/messages"))
        .json(&json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["created_at"], 1710000010);
    assert_eq!(body.get("created"), None);
    assert_eq!(body.get("object"), None);
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][1]["type"], "tool_use");
    assert_eq!(body["content"][1]["name"], "Bash");
    assert_eq!(body["content"][1]["input"]["command"], "pwd");
    assert_eq!(body["stop_reason"], "tool_use");
    assert_eq!(body["usage"]["input_tokens"], 19859);
    assert_eq!(body["usage"]["output_tokens"], 118);
    assert_eq!(body["usage"]["cache_creation_input_tokens"], 0);
    assert_eq!(body["usage"]["cache_read_input_tokens"], 8704);
}

#[tokio::test]
async fn upstream_claude_json_is_transformed_back_to_openai_responses() {
    let upstream_body = json!({
        "id": "msg_claude_resp_123",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "created_at": 1710000100,
        "content": [
            {
                "type": "thinking",
                "thinking": "step one"
            },
            {
                "type": "text",
                "text": "hello from claude"
            },
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "lookup_weather",
                "input": {
                    "city": "Paris"
                }
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 3,
            "output_tokens": 5
        }
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_responses",
                "to": "claude_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/responses"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "input": "hi"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["object"], "response");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "hello from claude");
    assert_eq!(body["output"][1]["type"], "reasoning");
    assert_eq!(body["output"][1]["summary"][0]["text"], "step one");
    assert_eq!(body["output"][2]["type"], "function_call");
    assert_eq!(body["output"][2]["call_id"], "toolu_123");
    assert_eq!(body["output"][2]["name"], "lookup_weather");
    assert_eq!(body["output"][2]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(body["usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["output_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn upstream_gemini_json_is_transformed_back_to_openai_responses() {
    let upstream_body = json!({
        "responseId": "gem_resp_123",
        "modelVersion": "gemini-2.5-flash",
        "createTime": "2026-05-20T00:00:00Z",
        "candidates": [{
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": "hidden chain",
                        "thought": true
                    },
                    {
                        "text": "hello from gemini"
                    },
                    {
                        "functionCall": {
                            "id": "call_g1",
                            "name": "lookup_weather",
                            "args": {
                                "city": "Paris"
                            }
                        }
                    }
                ]
            },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 11,
            "totalTokenCount": 18
        }
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_responses",
                "to": "gemini_chat"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/responses"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "input": "hi"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["id"], "gem_resp_123");
    assert_eq!(body["model"], "gemini-2.5-flash");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["content"][0]["text"], "hello from gemini");
    assert_eq!(body["output"][1]["type"], "reasoning");
    assert_eq!(body["output"][1]["summary"][0]["text"], "hidden chain");
    assert_eq!(body["output"][2]["type"], "function_call");
    assert_eq!(body["output"][2]["call_id"], "call_g1");
    assert_eq!(body["output"][2]["name"], "lookup_weather");
    assert_eq!(body["output"][2]["arguments"], "{\"city\":\"Paris\"}");
    assert_eq!(body["usage"]["input_tokens"], 7);
    assert_eq!(body["usage"]["output_tokens"], 11);
    assert_eq!(body["usage"]["total_tokens"], 18);
}

#[tokio::test]
async fn upstream_openai_responses_message_content_items_object_maps_to_chat() {
    let upstream_body = json!({
        "id": "resp_items_obj",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": {
                "items": [
                    {
                        "type": "output_text",
                        "text": "hello from items"
                    }
                ]
            }
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "hello from items");
}

#[tokio::test]
async fn upstream_openai_responses_message_content_single_object_maps_to_chat() {
    let upstream_body = json!({
        "id": "resp_single_obj",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": {
                "type": "output_text",
                "text": "hello from object"
            }
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "hello from object"
    );
}

#[tokio::test]
async fn upstream_openai_responses_function_call_is_transformed_back_to_chat_tool_calls() {
    let upstream_body = json!({
        "id": "resp_tool",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup_weather",
            "arguments": "{\"city\":\"Paris\"}"
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup_weather"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
}

#[tokio::test]
async fn upstream_openai_responses_function_call_object_arguments_are_stringified_for_chat() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_tool_object_args",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup_weather",
            "arguments": {
                "city": "Paris"
            }
        }]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
}

#[tokio::test]
async fn upstream_openai_responses_multiple_function_calls_are_preserved_in_chat_message() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_tool_multi",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": "{\"city\":\"Paris\"}"
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "lookup_time",
                "arguments": "{\"timezone\":\"UTC\"}"
            }
        ]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "weather and time?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_null(), "{body}");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "lookup_weather"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][1]["id"],
        "call_2"
    );
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][1]["function"]["name"],
        "lookup_time"
    );
}

#[tokio::test]
async fn upstream_openai_responses_usage_is_preserved_in_chat_response() {
    let upstream_body = json!({
        "id": "resp_usage",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }],
        "usage": {
            "input_tokens": 3,
            "output_tokens": 5,
            "total_tokens": 8
        }
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["usage"]["prompt_tokens"], 3);
    assert_eq!(body["usage"]["completion_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 8);
    assert_eq!(body["usage"]["responses_usage"]["input_tokens"], 3);
    assert_eq!(body["usage"]["responses_usage"]["output_tokens"], 5);
}

#[tokio::test]
async fn upstream_openai_responses_incomplete_status_maps_to_chat_length_finish_reason() {
    let upstream_body = json!({
        "id": "resp_incomplete",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "incomplete",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn upstream_openai_responses_failed_status_maps_to_chat_error_finish_reason() {
    let upstream_body = json!({
        "id": "resp_failed",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "failed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["finish_reason"], "error");
}

#[tokio::test]
async fn upstream_openai_responses_without_status_keeps_chat_finish_reason_null() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_no_status",
        "object": "response",
        "model": "gpt-4.1-mini",
        "created_at": 321,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "partial"
            }]
        }]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "partial");
    assert_eq!(body["choices"][0]["finish_reason"], Value::Null);
}

#[tokio::test]
async fn upstream_openai_responses_without_created_does_not_emit_zero_created_field() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_no_created",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "partial"
            }]
        }]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "partial");
    assert_eq!(body.get("created"), None);
}

#[tokio::test]
async fn upstream_openai_responses_null_created_does_not_emit_created_field() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_null_created",
        "object": "response",
        "model": "gpt-4.1-mini",
        "created_at": null,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "partial"
            }]
        }]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "partial");
    assert_eq!(body.get("created"), None);
}

#[tokio::test]
async fn upstream_openai_responses_without_usage_emits_null_usage_field() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_no_usage",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "partial"
            }]
        }]
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "partial");
    assert_eq!(body.get("usage"), Some(&Value::Null));
}

#[tokio::test]
async fn upstream_openai_responses_extra_fields_are_preserved_in_chat_response() {
    let upstream_body = json!({
        "id": "resp_extra",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "created_at": 123,
        "service_tier": "default",
        "metadata": {
            "trace_id": "trace-1"
        },
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["service_tier"], "default");
    assert_eq!(body["metadata"]["trace_id"], "trace-1");
}

#[tokio::test]
async fn upstream_openai_responses_final_chat_message_uses_last_assistant_item() {
    let upstream_body = json!({
        "id": "resp_mixed",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "hello"
                }]
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup_weather",
                "arguments": "{\"city\":\"Paris\"}"
            }
        ]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert!(body["choices"][0]["message"]["content"].is_null(), "{body}");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
}

#[tokio::test]
async fn upstream_openai_responses_function_call_output_does_not_surface_in_final_chat_message() {
    let upstream_body = json!({
        "id": "resp_tool_output",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "function_call_output",
            "call_id": "call_1",
            "name": "lookup_weather",
            "output": "{\"temp\":20}"
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_null(), "{body}");
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "{body}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn upstream_openai_responses_trailing_function_call_output_keeps_previous_assistant_message()
{
    let upstream_body = json!({
        "id": "resp_tool_output_trailing_message",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "final answer"
                }]
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "name": "lookup_weather",
                "output": "{\"temp\":20}"
            }
        ]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "final answer");
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "{body}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn upstream_openai_responses_image_content_maps_to_chat_content_parts() {
    let upstream_body = json!({
        "id": "resp_image",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "look"
                },
                {
                    "type": "input_image",
                    "image_url": "https://example.com/cat.png",
                    "detail": "high"
                }
            ]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"][0]["type"], "text");
    assert_eq!(body["choices"][0]["message"]["content"][0]["text"], "look");
    assert_eq!(
        body["choices"][0]["message"]["content"][1]["type"],
        "image_url"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][1]["image_url"]["url"],
        "https://example.com/cat.png"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][1]["image_url"]["detail"],
        "high"
    );
}

#[tokio::test]
async fn upstream_openai_responses_image_url_object_maps_to_chat_content_parts() {
    let upstream_body = json!({
        "id": "resp_image_obj",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/dog.png",
                    "detail": "low"
                }
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["type"],
        "image_url"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["url"],
        "https://example.com/dog.png"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["detail"],
        "low"
    );
}

#[tokio::test]
async fn upstream_openai_responses_image_url_string_field_maps_to_chat_content_parts() {
    let upstream_body = json!({
        "id": "resp_image_string_field",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "image_url",
                "image_url": "https://example.com/string.png",
                "detail": "high"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["type"],
        "image_url"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["url"],
        "https://example.com/string.png"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["detail"],
        "high"
    );
}

#[tokio::test]
async fn upstream_openai_responses_image_url_flat_url_field_maps_to_chat_content_parts() {
    let upstream_body = json!({
        "id": "resp_image_flat_url",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "image_url",
                "url": "https://example.com/flat.png",
                "detail": "auto"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["type"],
        "image_url"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["url"],
        "https://example.com/flat.png"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["detail"],
        "auto"
    );
}

#[tokio::test]
async fn upstream_openai_responses_image_object_url_field_maps_to_chat_content_parts() {
    let upstream_body = json!({
        "id": "resp_image_nested_url",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "input_image",
                "image": {
                    "url": "https://example.com/nested.png"
                },
                "detail": "low"
            }]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["type"],
        "image_url"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["url"],
        "https://example.com/nested.png"
    );
    assert_eq!(
        body["choices"][0]["message"]["content"][0]["image_url"]["detail"],
        "low"
    );
}

#[tokio::test]
async fn upstream_openai_responses_multiple_text_parts_join_with_newlines() {
    let upstream_body = json!({
        "id": "resp_multitext",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "hello"
                },
                {
                    "type": "output_text",
                    "text": "world"
                }
            ]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "hello\nworld");
}

#[tokio::test]
async fn upstream_openai_responses_only_uses_last_message_item_for_final_chat_message() {
    let upstream_body = json!({
        "id": "resp_last_message",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "first"
                }]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "second"
                }]
            }
        ]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "second");
}

#[tokio::test]
async fn upstream_openai_responses_reasoning_item_maps_to_chat_reasoning_content() {
    let upstream_body = json!({
        "id": "resp_reasoning",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": [{
            "type": "reasoning",
            "summary": [
                {"text": "step one"},
                {"text": "step two"}
            ]
        }]
    });

    let upstream_base = spawn_json_upstream(upstream_body).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body["choices"][0]["message"]["reasoning_content"],
        "step one\nstep two"
    );
    assert!(body["choices"][0]["message"].get("content").is_none());
}

#[tokio::test]
async fn upstream_openai_responses_empty_output_keeps_empty_assistant_message_shape() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_empty_output",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": []
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert!(body["choices"][0]["message"]["content"].is_null(), "{body}");
    assert!(
        body["choices"][0]["message"]["tool_calls"].is_null(),
        "{body}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body.get("usage"), Some(&Value::Null));
}

#[tokio::test]
async fn non_json_success_response_is_wrapped_as_json_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "text/plain; charset=utf-8",
        b"plain text body".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body,
        json!({
            "text": "plain text body",
            "status_code": 200
        })
    );
}

#[tokio::test]
async fn non_json_success_response_respects_declared_charset_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "text/plain; charset=iso-8859-1",
        b"caf\xe9".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(
        body,
        json!({
            "text": "café",
            "status_code": 200
        })
    );
}

#[tokio::test]
async fn successful_json_response_does_not_preserve_upstream_custom_headers_like_python() {
    let upstream_base = spawn_json_upstream_with_header(
        json!({
            "ok": true
        }),
        ("x-upstream-marker", "kept-by-rust-today"),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-marker")
            .and_then(|value| value.to_str().ok()),
        None
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["ok"], true);
}

#[tokio::test]
async fn json_array_success_response_is_passed_through_when_response_transform_fails() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "application/json",
        br#"[{"type":"unexpected"}]"#.to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses"
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body, json!([{ "type": "unexpected" }]));
}

#[tokio::test]
async fn json_scalar_success_response_is_passed_through_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "application/json",
        br#""hello from upstream""#.to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body, json!("hello from upstream"));
}

#[tokio::test]
async fn delay_stream_header_allows_json_scalar_success_response_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "application/json",
        br#""hello from upstream""#.to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "delay_stream_header": true
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body, json!("hello from upstream"));
}

#[tokio::test]
async fn delay_stream_header_skips_validation_for_non_200_success_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::CREATED,
        "text/plain; charset=utf-8",
        b"x".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "delay_stream_header": true
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain; charset=utf-8")
    );
    let body = response.text().await.expect("text body");
    assert_eq!(body, "x");
}

#[tokio::test]
async fn non_200_ok_empty_response_is_passed_through_like_python() {
    let upstream_base =
        spawn_fixed_upstream(StatusCode::NO_CONTENT, "text/plain", Vec::new()).await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/plain")
    );
    let body = response.bytes().await.expect("body bytes");
    assert!(body.is_empty(), "{body:?}");
}

#[tokio::test]
async fn non_200_response_preserves_upstream_custom_headers_like_python() {
    let upstream_base = spawn_fixed_upstream_with_extra_header(
        StatusCode::BAD_REQUEST,
        "application/json",
        br#"{"error":"bad request"}"#.to_vec(),
        ("x-upstream-marker", "kept-on-pass-through"),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-marker")
            .and_then(|value| value.to_str().ok()),
        Some("kept-on-pass-through")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body, json!({"error": "bad request"}));
}

#[tokio::test]
async fn non_200_response_strips_denied_headers_like_python() {
    let upstream_base = spawn_fixed_upstream_with_headers(
        StatusCode::BAD_REQUEST,
        "application/json",
        br#"{"error":"bad request"}"#.to_vec(),
        vec![
            ("x-upstream-marker", "kept-on-pass-through"),
            ("set-cookie", "session=secret"),
            ("x-frame-options", "DENY"),
        ],
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get("x-upstream-marker")
            .and_then(|value| value.to_str().ok()),
        Some("kept-on-pass-through")
    );
    assert_eq!(response.headers().get("set-cookie"), None);
    assert_eq!(response.headers().get("x-frame-options"), None);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body, json!({"error": "bad request"}));
}

#[tokio::test]
async fn delay_stream_header_validates_non_stream_empty_response_like_python() {
    let upstream_base = spawn_json_upstream(json!({
        "id": "resp_empty_for_validation",
        "object": "response",
        "model": "gpt-4.1-mini",
        "status": "completed",
        "output": []
    }))
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses",
                "delay_stream_header": true
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "EMPTY_RESPONSE");
    assert_eq!(body["error"]["type"], "content_validation_error");
}

#[tokio::test]
async fn delay_stream_header_rejects_non_json_success_response_like_python() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "text/plain; charset=utf-8",
        b"plain text body".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "delay_stream_header": true
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "EMPTY_RESPONSE");
    assert_eq!(body["error"]["type"], "content_validation_error");
}

#[tokio::test]
async fn delay_stream_header_non_json_transform_failure_uses_client_format_for_validation_like_python(
) {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::OK,
        "text/plain; charset=utf-8",
        b"plain text body".to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(
        &json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "openai_chat",
                "to": "openai_responses",
                "delay_stream_header": true
            }
        })
        .to_string(),
    );

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!(
            "{proxy_base}/{config}${upstream_base}/v1/chat/completions"
        ))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "EMPTY_RESPONSE");
    assert_eq!(body["error"]["type"], "content_validation_error");
    assert!(body["error"]["message"]
        .as_str()
        .expect("error message")
        .contains("Format name: openai_chat"));
}

async fn spawn_json_upstream(body: Value) -> String {
    spawn_json_upstream_with_header(body, ("x-unused", "unused")).await
}

async fn spawn_json_upstream_with_header(
    body: Value,
    header: (&'static str, &'static str),
) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let body = body.clone();
            let header = header;
            async move { (StatusCode::OK, [(header.0, header.1)], Json(body)) }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_fixed_upstream(
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
) -> String {
    spawn_fixed_upstream_with_headers(status, content_type, body, vec![("x-unused", "unused")])
        .await
}

async fn spawn_fixed_upstream_with_extra_header(
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
    header: (&'static str, &'static str),
) -> String {
    spawn_fixed_upstream_with_headers(status, content_type, body, vec![header]).await
}

async fn spawn_fixed_upstream_with_headers(
    status: StatusCode,
    content_type: &'static str,
    body: Vec<u8>,
    headers: Vec<(&'static str, &'static str)>,
) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let body = body.clone();
            let headers = headers.clone();
            async move {
                let mut response = Response::builder()
                    .status(status)
                    .header(axum::http::header::CONTENT_TYPE, content_type);
                for (name, value) in headers {
                    response = response.header(name, value);
                }
                response.body(Body::from(body)).expect("upstream response")
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let addr = listener.local_addr().expect("upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_proxy_server() -> String {
    let app = proxy_router(AppState {
        settings: Arc::new(test_settings()),
        http_client: reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("http client"),
    });

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("proxy server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

fn test_settings() -> Settings {
    Settings {
        host: "127.0.0.1".to_string(),
        port: 0,
        debug: true,
        log_level: "info".to_string(),
        upstream_http_timeout_secs: 60,
        access_log_file: "logs/access.log".to_string(),
        moderation_log_file: "logs/moderation.log".to_string(),
        training_log_file: "logs/training.log".to_string(),
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: "/tmp/prismguard-test.sock".to_string(),
        training_scheduler_enabled: true,
        training_scheduler_interval_minutes: 10,
        training_scheduler_failure_cooldown_minutes: 30,
        training_subprocess_allowed_cpus: "0".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map: HashMap::new(),
    }
}

fn percent_encode(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}
