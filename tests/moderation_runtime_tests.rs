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
#[path = "../src/sample_rpc.rs"]
mod sample_rpc;
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/training.rs"]
mod training;
#[path = "../src/moderation/basic.rs"]
mod basic;
#[path = "../src/moderation/extract.rs"]
mod extract;
#[path = "../src/moderation/mod.rs"]
mod moderation;
#[path = "../src/streaming.rs"]
mod streaming;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use config::Settings;
use routes::{router as proxy_router, AppState};
use serde_json::{json, Value};

#[derive(Clone, Debug)]
struct SeenRequest {
    path: String,
    json_body: Value,
}

#[derive(Clone)]
struct UpstreamState {
    seen: Arc<Mutex<Vec<SeenRequest>>>,
}

fn temp_keywords_path(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prismguard-basic-moderation-{label}-{}-{unique}.txt",
        std::process::id()
    ))
}

#[test]
fn basic_moderation_returns_none_when_disabled() {
    let blocked = basic::basic_moderation("blocked phrase", &json!({ "enabled": false }))
        .expect("basic moderation result");

    assert!(blocked.is_none());
}

#[test]
fn basic_moderation_blocks_matching_keyword_case_insensitively() {
    let keyword_file = temp_keywords_path("match");
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let blocked = basic::basic_moderation(
        "This contains BLOCKED PHRASE in mixed case.",
        &json!({
            "enabled": true,
            "keywords_file": keyword_file,
            "error_code": "BASIC_MODERATION_BLOCKED"
        }),
    )
    .expect("basic moderation result");

    let blocked = blocked.expect("keyword should block");
    assert_eq!(
        blocked.reason,
        "[BASIC_MODERATION_BLOCKED] Matched keyword: blocked phrase"
    );

    let _ = std::fs::remove_file(keyword_file);
}

#[test]
fn basic_moderation_reloads_keywords_after_file_changes() {
    let keyword_file = temp_keywords_path("reload");
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let cfg = json!({
        "enabled": true,
        "keywords_file": keyword_file,
        "error_code": "BASIC_MODERATION_BLOCKED"
    });

    let first = basic::basic_moderation("blocked phrase appears here", &cfg)
        .expect("first moderation result");
    assert!(first.is_some(), "initial keyword should block");

    std::thread::sleep(Duration::from_millis(5));
    std::fs::write(&keyword_file, "different phrase\n").expect("rewrite keyword file");

    let stale = basic::basic_moderation("blocked phrase appears here", &cfg)
        .expect("stale moderation result");
    assert!(
        stale.is_none(),
        "cache should refresh after keyword file content changes"
    );

    let refreshed = basic::basic_moderation("different phrase appears here", &cfg)
        .expect("refreshed moderation result");
    let refreshed = refreshed.expect("updated keyword should block");
    assert_eq!(
        refreshed.reason,
        "[BASIC_MODERATION_BLOCKED] Matched keyword: different phrase"
    );

    let _ = std::fs::remove_file(keyword_file);
}

#[test]
fn extract_openai_chat_text_joins_string_and_text_parts() {
    let body = json!({
        "messages": [
            {"role": "system", "content": "system guardrails"},
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "text", "text": "blocked phrase"}
            ]},
            {"role": "tool", "content": "tool output should be ignored"}
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "openai_chat");

    assert_eq!(text, "system guardrails\nhello\nblocked phrase");
}

#[test]
fn extract_claude_chat_text_includes_system_blocks_and_message_parts() {
    let body = json!({
        "system": [
            {"type": "text", "text": "system prelude"},
            {"type": "tool_use", "name": "lookup_weather"}
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first question"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
            ]},
            {"role": "assistant", "content": "assistant reply"}
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "claude_chat");

    assert_eq!(text, "system prelude\nfirst question\nassistant reply");
}

#[test]
fn extract_openai_responses_text_includes_instructions_and_structured_input() {
    let body = json!({
        "instructions": "system instructions",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                    {"type": "input_text", "text": "blocked phrase"}
                ]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "assistant context"}
                ]
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tool result text"
            }
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "openai_responses");

    assert_eq!(
        text,
        "system instructions\nhello\nblocked phrase\nassistant context\ntool result text"
    );
}

#[tokio::test]
async fn basic_moderation_blocks_before_upstream_proxy_call() {
    let keyword_file = std::env::temp_dir().join(format!(
        "prismguard-runtime-basic-keywords-{}.txt",
        std::process::id()
    ));
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "basic_moderation": {
            "enabled": true,
            "keywords_file": keyword_file,
            "error_code": "BASIC_MODERATION_BLOCKED"
        }
    }).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello blocked phrase world"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["source_format"], "openai_chat");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("[BASIC_MODERATION_BLOCKED] Matched keyword: blocked phrase")
    );
    assert!(seen.lock().expect("seen lock").is_empty());
}

#[tokio::test]
async fn smart_moderation_local_hashlinear_blocks_before_upstream_proxy_call() {
    let profile_name = format!("runtime-hashlinear-block-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{}/v1", unused_http_base()),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 1,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "analyzer": "char",
                "ngram_range": [2, 4],
                "n_features": 1,
                "alternate_sign": false,
                "norm": "l2",
                "use_jieba": false
            },
            "probability": {
                "ai_review_rate": 0.0,
                "random_seed": 42,
                "low_risk_threshold": 0.2,
                "high_risk_threshold": 0.8
            }
        })
        .to_string(),
    )
    .expect("write profile");
    std::fs::write(profile_dir.join("ai_prompt.txt"), "审核文本：{{text}}")
        .expect("write prompt");
    std::fs::write(profile_dir.join("hashlinear_model.pkl"), b"stub-model")
        .expect("write model marker");
    write_hashlinear_runtime(&profile_dir, -5.0, &[10.0]);

    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "smart_moderation": {
            "enabled": true,
            "profile": profile_name
        }
    }).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hashlinear should decide locally"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["source_format"], "openai_chat");
    assert_eq!(body["error"]["moderation_details"]["source"], "hashlinear_model");
    assert!(
        body["error"]["moderation_details"]["confidence"]
            .as_f64()
            .expect("confidence")
            > 0.8
    );
    assert!(seen.lock().expect("seen lock").is_empty());
}

#[tokio::test]
async fn smart_moderation_low_risk_hashlinear_still_reaches_upstream() {
    let profile_name = format!("runtime-hashlinear-allow-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{}/v1", unused_http_base()),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 1,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "analyzer": "char",
                "ngram_range": [2, 4],
                "n_features": 1,
                "alternate_sign": false,
                "norm": "l2",
                "use_jieba": false
            },
            "probability": {
                "ai_review_rate": 0.0,
                "random_seed": 42,
                "low_risk_threshold": 0.2,
                "high_risk_threshold": 0.8
            }
        })
        .to_string(),
    )
    .expect("write profile");
    std::fs::write(profile_dir.join("ai_prompt.txt"), "审核文本：{{text}}")
        .expect("write prompt");
    std::fs::write(profile_dir.join("hashlinear_model.pkl"), b"stub-model")
        .expect("write model marker");
    write_hashlinear_runtime(&profile_dir, 5.0, &[-10.0]);

    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({
        "smart_moderation": {
            "enabled": true,
            "profile": profile_name
        }
    }).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "local low risk should pass"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(
        request.json_body["messages"][0]["content"],
        "local low risk should pass"
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
            json_body: json_body.clone(),
        });

    (StatusCode::OK, Json(json!({"ok": true})))
}

async fn spawn_proxy_server() -> String {
    let app = proxy_router(AppState {
        settings: Arc::new(test_settings(HashMap::new())),
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
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: "/tmp/prismguard-test.sock".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map,
    }
}

fn percent_encode(input: &str) -> String {
    url::form_urlencoded::byte_serialize(input.as_bytes()).collect()
}

fn unused_http_base() -> String {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind temp listener");
    let addr = listener.local_addr().expect("temp listener addr");
    drop(listener);
    format!("http://{}", addr)
}

fn write_hashlinear_runtime(profile_dir: &PathBuf, intercept: f32, coef: &[f32]) {
    std::fs::write(
        profile_dir.join("hashlinear_runtime.json"),
        json!({
            "runtime_version": 1,
            "source_model": profile_dir.join("hashlinear_model.pkl"),
            "n_features": coef.len(),
            "intercept": intercept,
            "classes": [0, 1],
            "cfg": {
                "analyzer": "char",
                "ngram_range": [2, 4],
                "n_features": coef.len(),
                "alternate_sign": false,
                "norm": "l2",
                "lowercase": true,
                "use_jieba": false
            }
        })
        .to_string(),
    )
    .expect("write runtime json");

    let mut bytes = Vec::with_capacity(std::mem::size_of_val(coef));
    for value in coef {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(profile_dir.join("hashlinear_runtime.coef.f32"), bytes)
        .expect("write runtime coef");
}
