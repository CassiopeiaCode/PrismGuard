#[path = "../src/config.rs"]
mod config;
#[path = "../src/format.rs"]
mod format;
#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/proxy.rs"]
mod proxy;
#[path = "../src/response.rs"]
mod response;
#[path = "../src/routes.rs"]
mod routes;
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/streaming.rs"]
mod streaming;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    assert_eq!(body["choices"][0]["message"]["tool_calls"][0]["id"], "call_1");
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    assert!(body["choices"][0]["message"]["tool_calls"].is_null(), "{body}");
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    assert_eq!(body["choices"][0]["message"]["content"][1]["type"], "image_url");
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"][0]["type"], "image_url");
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
async fn upstream_openai_responses_reasoning_item_maps_to_chat_text_content() {
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
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "step one\nstep two");
}

#[tokio::test]
async fn non_json_success_response_is_passed_through_as_text_body() {
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
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
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
        Some("text/plain; charset=utf-8")
    );
    let body = response.text().await.expect("text body");
    assert_eq!(body, "plain text body");
}

async fn spawn_json_upstream(body: Value) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let body = body.clone();
            async move { (StatusCode::OK, Json(body)) }
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

async fn spawn_fixed_upstream(status: StatusCode, content_type: &'static str, body: Vec<u8>) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let body = body.clone();
            async move {
                (
                    status,
                    [(axum::http::header::CONTENT_TYPE, content_type)],
                    body,
                )
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
        access_log_file: "logs/access.log".to_string(),
        moderation_log_file: "logs/moderation.log".to_string(),
        training_log_file: "logs/training.log".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map: HashMap::new(),
    }
}

fn percent_encode(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}
