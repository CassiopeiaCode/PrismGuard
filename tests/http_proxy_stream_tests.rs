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
#[path = "../src/training.rs"]
mod training;
#[path = "../src/moderation/mod.rs"]
mod moderation;
#[path = "../src/streaming.rs"]
mod streaming;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{header::CONTENT_TYPE, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};
use config::Settings;
use routes::{router as proxy_router, AppState};
use serde_json::json;

#[tokio::test]
async fn openai_responses_sse_is_transformed_back_to_openai_chat_sse() {
    let upstream_base = spawn_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );

    let body = response.text().await.expect("sse body");
    assert!(body.contains("chat.completion.chunk"), "{body}");
    assert!(body.contains("\"role\":\"assistant\""), "{body}");
    assert!(body.contains("\"content\":\"hello\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(
        body.contains("\"usage\":{\"input_tokens\":3,\"output_tokens\":5,\"total_tokens\":8}"),
        "{body}"
    );
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn delay_stream_header_returns_json_error_before_committing_empty_stream() {
    let upstream_base = spawn_empty_sse_upstream().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );

    let body: serde_json::Value = response.json().await.expect("json body");
    assert!(body["error"]["message"].as_str().unwrap().contains("Stream"));
}

#[tokio::test]
async fn delay_stream_header_rejects_too_short_stream_content_like_python() {
    let upstream_base = spawn_too_short_sse_upstream().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );

    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "UPSTREAM_STREAM_ERROR");
    assert_eq!(body["error"]["type"], "upstream_stream_error");
}

#[tokio::test]
async fn delay_stream_header_allows_large_unrecognized_stream_after_protection_limit() {
    let upstream_base = spawn_large_unrecognized_sse_upstream().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
}

#[tokio::test]
async fn non_200_sse_is_passed_through_without_delay_validation_like_python() {
    let upstream_base = spawn_status_sse_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "data: {\"error\":\"busy\"}\n\n",
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"error\":\"busy\""), "{body}");
}

#[tokio::test]
async fn non_200_ok_sse_is_still_passed_through_like_python() {
    let upstream_base = spawn_status_sse_upstream(
        StatusCode::CREATED,
        "data: {\"created\":\"yes\"}\n\n",
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::CREATED);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"created\":\"yes\""), "{body}");
    assert!(!body.contains("chat.completion.chunk"), "{body}");
}

#[tokio::test]
async fn non_200_sse_preserves_custom_headers_and_strips_denied_like_python() {
    let upstream_base = spawn_status_sse_upstream_with_headers(
        StatusCode::TOO_MANY_REQUESTS,
        "data: {\"error\":\"busy\"}\n\n",
        vec![
            ("x-upstream-marker", "kept-on-pass-through"),
            ("set-cookie", "session=secret"),
            ("x-frame-options", "DENY"),
        ],
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    assert_eq!(
        response
            .headers()
            .get("x-upstream-marker")
            .and_then(|value| value.to_str().ok()),
        Some("kept-on-pass-through")
    );
    assert_eq!(response.headers().get("set-cookie"), None);
    assert_eq!(response.headers().get("x-frame-options"), None);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"error\":\"busy\""), "{body}");
}

#[tokio::test]
async fn stream_request_non_200_json_error_preserves_json_content_type_like_python() {
    let upstream_base = spawn_status_json_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        json!({
            "error": {
                "message": "busy"
            }
        }),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }).to_string());

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["message"], "busy");
}

#[tokio::test]
async fn stream_request_with_non_stream_json_success_stays_json_like_python() {
    let upstream_base = spawn_status_json_upstream(
        StatusCode::OK,
        json!({
            "id": "resp_sync",
            "object": "response",
            "model": "gpt-4.1-mini",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "hello"
                }]
            }]
        }),
    )
    .await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
}

#[tokio::test]
async fn stream_request_with_mislabelled_sse_response_is_still_treated_as_stream_like_python() {
    let upstream_base = spawn_mislabelled_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("text/event-stream")
    );
    let body = response.text().await.expect("stream body");
    assert!(body.contains("chat.completion.chunk"), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_failed_event_maps_to_chat_error_finish_reason() {
    let upstream_base = spawn_failed_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"finish_reason\":\"error\""), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_incomplete_event_maps_to_chat_length_finish_reason() {
    let upstream_base = spawn_incomplete_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"finish_reason\":\"length\""), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_error_event_maps_to_chat_error_finish_reason() {
    let upstream_base = spawn_error_event_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"finish_reason\":\"error\""), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_completed_event_emits_done_even_without_upstream_done_marker() {
    let upstream_base = spawn_completed_without_done_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
    assert!(body.contains("[DONE]"), "{body}");
}

#[tokio::test]
async fn responses_function_call_events_map_to_chat_tool_call_deltas() {
    let upstream_base = spawn_tool_call_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"tool_calls\":[{"), "{body}");
    assert!(body.contains("\"id\":\"call_1\""), "{body}");
    assert!(body.contains("\"name\":\"lookup_weather\""), "{body}");
    assert!(body.contains("\\\"city\\\":\\\"Paris\\\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
}

#[tokio::test]
async fn responses_tool_call_arguments_buffer_until_output_item_added() {
    let upstream_base = spawn_buffered_tool_call_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"id\":\"call_buf_1\""), "{body}");
    assert!(body.contains("\"name\":\"lookup_weather\""), "{body}");
    assert!(body.contains("\\\"city\\\":\\\"Paris\\\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
}

#[tokio::test]
async fn responses_function_call_delta_arguments_field_maps_to_chat_tool_call_delta() {
    let upstream_base = spawn_function_call_delta_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "weather?"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"id\":\"call_delta_1\""), "{body}");
    assert!(body.contains("\"name\":\"lookup_weather\""), "{body}");
    assert!(body.contains("\\\"city\\\":\\\"Paris\\\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
}

#[tokio::test]
async fn responses_stream_without_created_does_not_emit_zero_created_field() {
    let upstream_base = spawn_missing_created_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"content\":\"hello\""), "{body}");
    assert!(!body.contains("\"created\":0"), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
}

#[tokio::test]
async fn responses_stream_usage_is_reduced_to_chat_usage_shape() {
    let upstream_base = spawn_usage_shape_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"usage\":{\"input_tokens\":3,\"output_tokens\":5,\"total_tokens\":8}"), "{body}");
    assert!(!body.contains("\"reasoning_tokens\""), "{body}");
    assert!(!body.contains("\"output_token_details\""), "{body}");
}

#[tokio::test]
async fn responses_in_progress_event_can_start_chat_stream_without_created() {
    let upstream_base = spawn_in_progress_only_sse_upstream().await;
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
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let body = response.text().await.expect("sse body");
    assert!(body.contains("\"role\":\"assistant\""), "{body}");
    assert!(body.contains("\"content\":\"hello\""), "{body}");
    assert!(body.contains("\"finish_reason\":\"stop\""), "{body}");
}

async fn spawn_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"hello\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_123\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":5,\"total_tokens\":8}}}\n\n",
        "data: [DONE]\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("sse response")
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

async fn spawn_empty_sse_upstream() -> String {
    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(Vec::<u8>::new()))
                .expect("empty sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind empty upstream");
    let addr = listener.local_addr().expect("empty upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("empty upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_failed_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_failed\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.failed\n",
        "data: {\"type\":\"response.failed\",\"response\":{\"id\":\"resp_failed\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"failed\"}}\n\n",
        "data: [DONE]\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("failed sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind failed upstream");
    let addr = listener.local_addr().expect("failed upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("failed upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_incomplete_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_incomplete\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.incomplete\n",
        "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_incomplete\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"incomplete\"}}\n\n",
        "data: [DONE]\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("incomplete sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind incomplete upstream");
    let addr = listener.local_addr().expect("incomplete upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("incomplete upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_error_event_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_error_event\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"message\":\"boom\"}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("error event sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind error event upstream");
    let addr = listener.local_addr().expect("error event upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("error event upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_too_short_sse_upstream() -> String {
    let body = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_short\",\"model\":\"gpt-4.1-mini\",\"created_at\":1710000000}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"a\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
        "data: [DONE]\n\n"
    );
    spawn_raw_sse_upstream(body).await
}

async fn spawn_large_unrecognized_sse_upstream() -> String {
    let filler = "x".repeat(1200);
    let body = format!("data: {{{filler}}}\n\n");
    let leaked: &'static str = Box::leak(body.into_boxed_str());
    spawn_raw_sse_upstream(leaked).await
}

async fn spawn_raw_sse_upstream(body: &'static str) -> String {
    spawn_status_sse_upstream(StatusCode::OK, body).await
}

async fn spawn_status_sse_upstream(status: StatusCode, body: &'static str) -> String {
    spawn_status_sse_upstream_with_headers(status, body, Vec::new()).await
}

async fn spawn_status_sse_upstream_with_headers(
    status: StatusCode,
    body: &'static str,
    headers: Vec<(&'static str, &'static str)>,
) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let headers = headers.clone();
            async move {
                let mut response = Response::builder()
                    .status(status)
                    .header(CONTENT_TYPE, "text/event-stream");
                for (name, value) in headers {
                    response = response.header(name, value);
                }
                response
                    .body(Body::from(body.to_string()))
                    .expect("raw sse response")
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind raw sse upstream");
    let addr = listener.local_addr().expect("raw sse upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("raw sse upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_mislabelled_sse_upstream() -> String {
    let payload = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_mislabelled\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_mislabelled\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n",
        "data: [DONE]\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(payload.to_string()))
                .expect("mislabelled sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mislabelled sse upstream");
    let addr = listener
        .local_addr()
        .expect("mislabelled sse upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("mislabelled sse upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_status_json_upstream(status: StatusCode, body: serde_json::Value) -> String {
    let app = Router::new().route(
        "/*path",
        post(move || {
            let body = body.clone();
            async move { (status, Json(body)) }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind status json upstream");
    let addr = listener.local_addr().expect("status json upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("status json upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_completed_without_done_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_no_done\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_no_done\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("completed without done sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind completed without done upstream");
    let addr = listener.local_addr().expect("completed without done upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("completed without done upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_tool_call_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_1\",\"name\":\"lookup_weather\"}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("tool call sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind tool call upstream");
    let addr = listener.local_addr().expect("tool call upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("tool call upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_buffered_tool_call_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool_buf\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.function_call_arguments.delta\n",
        "data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_buf_1\",\"delta\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n",
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_buf_1\",\"call_id\":\"call_buf_1\",\"name\":\"lookup_weather\"}}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool_buf\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("buffered tool call sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind buffered tool call upstream");
    let addr = listener.local_addr().expect("buffered tool call upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("buffered tool call upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_function_call_delta_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_tool_delta\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_item.added\n",
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"id\":\"fc_delta_1\",\"call_id\":\"call_delta_1\",\"name\":\"lookup_weather\"}}\n\n",
        "event: response.function_call.delta\n",
        "data: {\"type\":\"response.function_call.delta\",\"call_id\":\"call_delta_1\",\"name\":\"lookup_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_tool_delta\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("function call delta sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind function call delta upstream");
    let addr = listener.local_addr().expect("function call delta upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("function call delta upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_missing_created_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_no_created\",\"model\":\"gpt-4.1-mini\"}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"hello\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_no_created\",\"model\":\"gpt-4.1-mini\",\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("missing created sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind missing created upstream");
    let addr = listener.local_addr().expect("missing created upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("missing created upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_usage_shape_sse_upstream() -> String {
    let payload = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_usage_shape\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_usage_shape\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":5,\"total_tokens\":8,\"output_token_details\":{\"reasoning_tokens\":2}}}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("usage shape sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind usage shape upstream");
    let addr = listener.local_addr().expect("usage shape upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("usage shape upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_in_progress_only_sse_upstream() -> String {
    let payload = concat!(
        "event: response.in_progress\n",
        "data: {\"type\":\"response.in_progress\",\"response\":{\"id\":\"resp_in_progress\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
        "event: response.completed\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_in_progress\",\"model\":\"gpt-4.1-mini\",\"created_at\":1,\"status\":\"completed\"}}\n\n"
    );

    let app = Router::new().route(
        "/*path",
        post(move || async move {
            Response::builder()
                .status(StatusCode::OK)
                .header(CONTENT_TYPE, "text/event-stream")
                .body(Body::from(payload.to_string()))
                .expect("in progress only sse response")
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind in progress only upstream");
    let addr = listener.local_addr().expect("in progress only upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("in progress only upstream server");
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
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: "/tmp/prismguard-test.sock".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map: HashMap::new(),
    }
}

fn percent_encode(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}
