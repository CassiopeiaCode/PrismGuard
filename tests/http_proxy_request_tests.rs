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
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use config::Settings;
use flate2::write::GzEncoder;
use flate2::Compression;
use routes::{router as proxy_router, AppState};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct SeenRequest {
    path: String,
    path_and_query: String,
    json_body: Value,
}

#[derive(Clone)]
struct UpstreamState {
    seen: Arc<Mutex<Vec<SeenRequest>>>,
}

#[tokio::test]
async fn compressed_json_request_is_decoded_and_preserves_upstream_prefix() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let payload = gzip_json(json!({
        "model": "claude-sonnet-4-5",
        "anthropic_version": "2023-06-01",
        "messages": [{"role": "user", "content": "hello"}]
    }));

    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "claude_chat",
            "to": "openai_chat"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/secret_endpoint/v1/messages");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .header("content-encoding", "gzip")
        .header("content-type", "application/json")
        .body(payload)
        .send()
        .await
        .expect("proxy response");
    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/secret_endpoint/v1/chat/completions");
    assert_eq!(request.json_body["messages"][0]["content"], "hello");
}

#[tokio::test]
async fn upstream_non_200_json_is_passed_through_without_rewrapping() {
    let upstream_base = spawn_fixed_upstream(
        StatusCode::TOO_MANY_REQUESTS,
        "application/json",
        br#"{"error":{"message":"busy"}}"#.to_vec(),
    )
    .await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());
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

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["message"], "busy");
}

#[tokio::test]
async fn gemini_stream_request_appends_alt_sse_query_parameter() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "gemini_chat"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gemini-2.5-flash",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert!(
        request
            .path_and_query
            .contains("/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"),
        "{}",
        request.path_and_query
    );
}

#[tokio::test]
async fn gemini_non_stream_request_does_not_append_alt_sse_query_parameter() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "gemini_chat"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gemini-2.5-flash",
            "stream": false,
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(
        request.path_and_query,
        "/v1beta/models/gemini-2.5-flash:generateContent"
    );
}

#[tokio::test]
async fn env_backed_proxy_config_is_accepted_in_url() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_PROXY_CFG".to_string(),
        json!({
            "format_transform": {
                "enabled": true,
                "strict_parse": true,
                "from": "claude_chat",
                "to": "openai_chat"
            }
        })
        .to_string(),
    )]))
    .await;
    let proxy_url = format!("{proxy_base}/!TEST_PROXY_CFG${upstream_base}/v1/messages");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.json_body["messages"][0]["content"], "hello");
}

#[tokio::test]
async fn openai_chat_request_normalizes_response_format_for_responses_upstream() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
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
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(request.json_body.get("response_format"), None);
    assert_eq!(request.json_body["max_output_tokens"], 64);
    assert_eq!(request.json_body["text"]["format"]["type"], "json_schema");
    assert_eq!(request.json_body["text"]["format"]["name"], "answer");
    assert_eq!(request.json_body["text"]["format"]["strict"], true);
}

#[tokio::test]
async fn openai_chat_request_normalizes_max_completion_tokens_for_responses_upstream() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "max_completion_tokens": 77
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(request.json_body.get("max_completion_tokens"), None);
    assert_eq!(request.json_body["max_output_tokens"], 77);
}

#[tokio::test]
async fn openai_chat_request_strips_stream_include_usage_for_responses_upstream() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "hi"}],
            "stream_options": {
                "include_usage": true,
                "test_marker": 1
            }
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(request.json_body["stream"], true);
    assert_eq!(request.json_body["stream_options"].get("include_usage"), None);
    assert_eq!(request.json_body["stream_options"]["test_marker"], 1);
}

#[tokio::test]
async fn openai_chat_request_normalizes_function_tool_choice_for_responses_upstream() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {
                "type": "function",
                "function": {
                    "name": "lookup_weather"
                }
            }
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(
        request.json_body["tool_choice"],
        json!({
            "type": "function",
            "name": "lookup_weather"
        })
    );
}

#[tokio::test]
async fn openai_chat_request_normalizes_text_response_format_for_responses_upstream() {
    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }).to_string());
    let upstream_full = format!("{upstream_base}/v1/chat/completions");
    let proxy_url = format!("{proxy_base}/{config}${upstream_full}");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {
                "type": "text"
            }
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.json_body.get("response_format"), None);
    assert_eq!(
        request.json_body["text"],
        json!({
            "format": {
                "type": "text"
            }
        })
    );
}

async fn spawn_upstream_echo_server() -> (String, Arc<Mutex<Vec<SeenRequest>>>) {
    let state = UpstreamState {
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let seen = state.seen.clone();

    let app = Router::new()
        .route("/*path", post(upstream_echo))
        .with_state(state);

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

    (format!("http://{}", addr), seen)
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
        .expect("bind fixed upstream");
    let addr = listener.local_addr().expect("fixed upstream addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("fixed upstream server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn upstream_echo(
    State(state): State<UpstreamState>,
    uri: Uri,
    body: Bytes,
) -> impl IntoResponse {
    let json_body: Value = serde_json::from_slice(&body).expect("json body");
    state
        .seen
        .lock()
        .expect("seen lock")
        .push(SeenRequest {
            path: uri.path().to_string(),
            path_and_query: uri
                .path_and_query()
                .map(|value| value.as_str().to_string())
                .unwrap_or_else(|| uri.path().to_string()),
            json_body: json_body.clone(),
        });

    (StatusCode::OK, Json(json!({"ok": true})))
}

async fn spawn_proxy_server() -> String {
    spawn_proxy_server_with_env(HashMap::new()).await
}

async fn spawn_proxy_server_with_env(env_map: HashMap<String, String>) -> String {
    let app = proxy_router(AppState {
        settings: Arc::new(test_settings(env_map)),
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

fn test_settings(env_map: HashMap<String, String>) -> Settings {
    Settings {
        host: "127.0.0.1".to_string(),
        port: 0,
        debug: true,
        log_level: "info".to_string(),
        access_log_file: "logs/access.log".to_string(),
        moderation_log_file: "logs/moderation.log".to_string(),
        training_log_file: "logs/training.log".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map,
    }
}

fn gzip_json(value: Value) -> Vec<u8> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(value.to_string().as_bytes())
        .expect("write gzip payload");
    encoder.finish().expect("finish gzip payload")
}

fn percent_encode(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}
