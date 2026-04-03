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
#[path = "../src/moderation/mod.rs"]
mod moderation;
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
use axum::routing::{delete, patch, post};
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
async fn healthz_uses_prismguard_service_name() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/healthz"))
        .send()
        .await
        .expect("healthz response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["ok"], true);
    assert_eq!(body["service"], "PrismGuard");
}

#[tokio::test]
async fn healthz_head_is_supported_like_fastapi_get_routes() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .head(format!("{proxy_base}/healthz"))
        .send()
        .await
        .expect("healthz response");

    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn healthz_method_not_allowed_uses_json_detail_body() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .patch(format!("{proxy_base}/healthz"))
        .send()
        .await
        .expect("healthz response");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["detail"], "Method Not Allowed");
}

#[tokio::test]
async fn openapi_json_exposes_prismguard_metadata() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/openapi.json"))
        .send()
        .await
        .expect("openapi response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["openapi"], "3.1.0");
    assert_eq!(body["info"]["title"], "PrismGuard");
    assert_eq!(
        body["info"]["description"],
        "高级 AI API 中间件 - 智能审核 · 格式转换 · 透明代理"
    );
    assert_eq!(body["info"]["version"], "1.0.0");
    assert!(body["paths"]["/healthz"]["get"].is_object());
    assert!(body["paths"]["/docs/oauth2-redirect"]["get"].is_object());
    assert!(body["paths"]["/{cfg_and_upstream}"]["get"].is_object());
    assert!(body["paths"]["/{cfg_and_upstream}"]["post"].is_object());
    assert!(body["paths"]["/{cfg_and_upstream}"]["put"].is_object());
    assert!(body["paths"]["/{cfg_and_upstream}"]["delete"].is_object());
}

#[tokio::test]
async fn docs_serves_html_that_references_openapi_schema() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/docs"))
        .send()
        .await
        .expect("docs response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    let body = response.text().await.expect("html body");
    assert!(body.contains("Swagger UI"));
    assert!(body.contains("/openapi.json"));
    assert!(body.contains("PrismGuard"));
}

#[tokio::test]
async fn redoc_serves_html_that_references_openapi_schema() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/redoc"))
        .send()
        .await
        .expect("redoc response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    let body = response.text().await.expect("html body");
    assert!(body.contains("ReDoc"));
    assert!(body.contains("/openapi.json"));
    assert!(body.contains("PrismGuard"));
}

#[tokio::test]
async fn docs_oauth2_redirect_serves_html_page() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/docs/oauth2-redirect"))
        .send()
        .await
        .expect("oauth2 redirect response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    let body = response.text().await.expect("html body");
    assert!(body.contains("oauth2"));
    assert!(body.contains("window.opener"));
}

#[tokio::test]
async fn docs_trailing_slash_redirects_to_docs_like_fastapi() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/docs/"))
        .send()
        .await
        .expect("docs response");

    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response
            .headers()
            .get("location")
            .and_then(|value| value.to_str().ok()),
        Some("/docs")
    );
}

#[tokio::test]
async fn redoc_trailing_slash_redirects_to_redoc_like_fastapi() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/redoc/"))
        .send()
        .await
        .expect("redoc response");

    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(
        response
            .headers()
            .get("location")
            .and_then(|value| value.to_str().ok()),
        Some("/redoc")
    );
}

#[tokio::test]
async fn unknown_route_without_proxy_config_returns_config_parse_error() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/totally-missing-route"))
        .send()
        .await
        .expect("unknown route response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "CONFIG_PARSE_ERROR");
    assert_eq!(body["error"]["type"], "config_error");
}

#[tokio::test]
async fn root_path_without_proxy_config_returns_config_parse_error() {
    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/"))
        .send()
        .await
        .expect("root route response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "CONFIG_PARSE_ERROR");
    assert_eq!(body["error"]["type"], "config_error");
}

#[tokio::test]
async fn debug_profile_exposes_training_and_model_status() {
    let profile_name = format!("debug-profile-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "local_model_type": "hashlinear"
        })
        .to_string(),
    )
    .expect("write profile");
    std::fs::write(profile_dir.join("hashlinear_model.pkl"), b"model")
        .expect("write model marker");
    std::fs::write(
        profile_dir.join(".train_status.json"),
        json!({
            "status": "failed",
            "timestamp": 1710000000,
            "message": "previous run failed",
            "sample_count": 5
        })
        .to_string(),
    )
    .expect("write training status");

    let proxy_base = spawn_proxy_server().await;
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .get(format!("{proxy_base}/debug/profile/{profile_name}"))
        .send()
        .await
        .expect("debug profile response");

    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["profile_name"], profile_name);
    assert_eq!(body["local_model_exists"], true);
    assert!(
        body["hashlinear_model_path"]
            .as_str()
            .expect("hashlinear model path")
            .ends_with("/hashlinear_model.pkl")
    );
    assert_eq!(body["training_status"]["status"], "failed");
    assert_eq!(body["training_status"]["message"], "previous run failed");
    assert_eq!(body["training_decision"]["should_train"], false);
    assert_eq!(body["training_decision"]["reason"], "insufficient_samples");
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
async fn unreachable_upstream_returns_500_proxy_error_like_python() {
    let proxy_base = spawn_proxy_server().await;
    let upstream_base = unused_http_base();
    let config = percent_encode(&json!({}).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "PROXY_ERROR");
    assert_eq!(body["error"]["type"], "proxy_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("upstream request failed")
    );
}

#[tokio::test]
async fn basic_moderation_blocks_matching_keyword_like_python() {
    let keyword_file = std::env::temp_dir().join(format!(
        "prismguard-basic-keywords-{}.txt",
        std::process::id()
    ));
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let proxy_base = spawn_proxy_server().await;
    let upstream_base = unused_http_base();
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
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "blocked phrase"}
            ]
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
}

#[tokio::test]
async fn basic_moderation_ignores_tool_role_content_like_python() {
    let keyword_file = std::env::temp_dir().join(format!(
        "prismguard-basic-keywords-tool-{}.txt",
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
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "tool", "content": "blocked phrase"}
            ]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/chat/completions");
}

#[tokio::test]
async fn smart_moderation_falls_back_to_llm_and_blocks_like_python() {
    let profile_name = format!("smart-llm-fallback-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let llm_base = spawn_openai_chat_server(json!({
        "choices": [{
            "message": {
                "content": "{\"violation\": true, \"category\": \"abuse\", \"reason\": \"llm blocked\"}"
            }
        }]
    }))
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
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

    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let upstream_base = unused_http_base();
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
            "messages": [{"role": "user", "content": "please moderate this"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["source_format"], "openai_chat");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(body["error"]["moderation_details"]["category"], "abuse");
    assert_eq!(body["error"]["moderation_details"]["reason"], "llm blocked");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("Smart moderation blocked by ai")
    );
}

#[tokio::test]
async fn smart_moderation_falls_back_to_llm_and_allows_clean_request() {
    let profile_name = format!("smart-llm-allow-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let llm_base = spawn_openai_chat_server(json!({
        "choices": [{
            "message": {
                "content": "{\"violation\": false, \"category\": null, \"reason\": \"clean\"}"
            }
        }]
    }))
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
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

    let (upstream_base, seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
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
            "messages": [{"role": "user", "content": "normal request"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/chat/completions");
    assert_eq!(request.json_body["messages"][0]["content"], "normal request");
}

#[tokio::test]
async fn smart_moderation_returns_concurrency_limit_like_python() {
    let profile_name = format!("smart-llm-concurrency-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let llm_base = spawn_delayed_openai_chat_server(
        json!({
            "choices": [{
                "message": {
                    "content": "{\"violation\": false, \"category\": null, \"reason\": \"clean\"}"
                }
            }]
        }),
        Duration::from_millis(400),
    )
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
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

    let (upstream_base, _seen) = spawn_upstream_echo_server().await;
    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let config = percent_encode(&json!({
        "smart_moderation": {
            "enabled": true,
            "profile": profile_name
        }
    }).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let mut handles = Vec::new();
    for idx in 0..5 {
        let proxy_url = proxy_url.clone();
        handles.push(tokio::spawn(async move {
            reqwest::Client::builder()
                .timeout(Duration::from_secs(5))
                .build()
                .expect("reqwest client")
                .post(proxy_url)
                .json(&json!({
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": format!("occupy slot {idx}")}]
                }))
                .send()
                .await
                .expect("proxy response")
        }));
    }

    tokio::time::sleep(Duration::from_millis(100)).await;

    let blocked = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "should hit concurrency limit"}]
        }))
        .send()
        .await
        .expect("blocked response");

    assert_eq!(blocked.status(), StatusCode::BAD_REQUEST);
    let body: Value = blocked.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["source_format"], "openai_chat");
    assert_eq!(body["error"]["moderation_details"]["source"], "concurrency_limit");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("并发数超限")
    );

    for handle in handles {
        let response = handle.await.expect("join request");
        assert_eq!(response.status(), StatusCode::OK);
    }
}

#[tokio::test]
async fn smart_moderation_uses_local_hashlinear_runtime_before_llm() {
    let profile_name = format!("smart-hashlinear-local-{}", std::process::id());
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

    let proxy_base = spawn_proxy_server().await;
    let upstream_base = unused_http_base();
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
}

#[tokio::test]
async fn smart_moderation_allows_low_risk_local_hashlinear_without_llm() {
    let profile_name = format!("smart-hashlinear-allow-{}", std::process::id());
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
    assert_eq!(request.json_body["messages"][0]["content"], "local low risk should pass");
}

#[tokio::test]
async fn smart_moderation_falls_back_to_llm_when_hashlinear_is_uncertain() {
    let profile_name = format!("smart-hashlinear-uncertain-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let llm_base = spawn_openai_chat_server(json!({
        "choices": [{
            "message": {
                "content": "{\"violation\": true, \"category\": \"policy\", \"reason\": \"llm fallback blocked\"}"
            }
        }]
    }))
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
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
    write_hashlinear_runtime(&profile_dir, 0.0, &[0.0]);

    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let upstream_base = unused_http_base();
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
            "messages": [{"role": "user", "content": "uncertain hashlinear should fallback"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["source_format"], "openai_chat");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(body["error"]["moderation_details"]["category"], "policy");
    assert_eq!(body["error"]["moderation_details"]["reason"], "llm fallback blocked");
}

#[tokio::test]
async fn smart_moderation_ai_review_rate_takes_priority_over_local_model() {
    let profile_name = format!("smart-ai-review-rate-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let llm_base = spawn_openai_chat_server(json!({
        "choices": [{
            "message": {
                "content": "{\"violation\": true, \"category\": \"sampled\", \"reason\": \"forced ai review\"}"
            }
        }]
    }))
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
            "probability": {
                "ai_review_rate": 1.0,
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

    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let upstream_base = unused_http_base();
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
            "messages": [{"role": "user", "content": "sampled path should use ai first"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(body["error"]["moderation_details"]["category"], "sampled");
    assert_eq!(body["error"]["moderation_details"]["reason"], "forced ai review");
}

#[tokio::test]
async fn smart_moderation_reuses_cached_result_for_same_text() {
    let profile_name = format!("smart-cache-hit-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let (llm_base, hits) = spawn_counting_openai_chat_server(json!({
        "choices": [{
            "message": {
                "content": "{\"violation\": true, \"category\": \"cache\", \"reason\": \"same response\"}"
            }
        }]
    }))
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 0
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
            "probability": {
                "ai_review_rate": 1.0,
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

    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let upstream_base = unused_http_base();
    let config = percent_encode(&json!({
        "smart_moderation": {
            "enabled": true,
            "profile": profile_name
        }
    }).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let first_body = loop {
        let response = reqwest::Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .expect("reqwest client")
            .post(&proxy_url)
            .json(&json!({
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "cache me"}]
            }))
            .send()
            .await
            .expect("proxy response");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.expect("json body");
        if body["error"]["moderation_details"]["source"] == "concurrency_limit" {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        }
        break body;
    };

    assert_eq!(first_body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(first_body["error"]["moderation_details"]["category"], "cache");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(&proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "cache me"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let second_body: Value = response.json().await.expect("json body");
    assert_eq!(second_body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(second_body["error"]["moderation_details"]["category"], "cache");

    assert_eq!(*hits.lock().expect("hits lock"), 1);
}

#[tokio::test]
async fn smart_moderation_retries_llm_after_server_error_like_python() {
    let profile_name = format!("smart-llm-retry-{}", std::process::id());
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(&profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");

    let (llm_base, hits) = spawn_sequenced_openai_chat_server(vec![
        (StatusCode::INTERNAL_SERVER_ERROR, json!({"error": {"message": "temporary"}})),
        (
            StatusCode::OK,
            json!({
                "choices": [{
                    "message": {
                        "content": "{\"violation\": true, \"category\": \"retry\", \"reason\": \"second attempt\"}"
                    }
                }]
            }),
        ),
    ])
    .await;

    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "ai": {
                "provider": "openai",
                "base_url": format!("{llm_base}/v1"),
                "model": "fake-moderation-model",
                "api_key_env": "TEST_MOD_AI_API_KEY",
                "timeout": 3,
                "max_retries": 1
            },
            "prompt": {
                "template_file": "ai_prompt.txt",
                "max_text_length": 4000
            },
            "local_model_type": "hashlinear",
            "probability": {
                "ai_review_rate": 1.0,
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

    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "TEST_MOD_AI_API_KEY".to_string(),
        "test-key".to_string(),
    )]))
    .await;
    let upstream_base = unused_http_base();
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
        .post(&proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "retry this moderation"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
    assert_eq!(body["error"]["moderation_details"]["category"], "retry");
    assert_eq!(body["error"]["moderation_details"]["reason"], "second attempt");
    assert_eq!(*hits.lock().expect("hits lock"), 2);
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
async fn invalid_env_backed_proxy_config_returns_400_config_error() {
    let proxy_base = spawn_proxy_server_with_env(HashMap::from([(
        "BROKEN_PROXY_CFG".to_string(),
        "{not-json}".to_string(),
    )]))
    .await;
    let proxy_url =
        format!("{proxy_base}/!BROKEN_PROXY_CFG$https://example.com/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "CONFIG_PARSE_ERROR");
    assert_eq!(body["error"]["type"], "config_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("failed to parse env BROKEN_PROXY_CFG as JSON")
    );
}

#[tokio::test]
async fn invalid_inline_proxy_config_returns_400_config_error() {
    let proxy_base = spawn_proxy_server().await;
    let proxy_url =
        format!("{proxy_base}/%7Bnot-json%7D$https://example.com/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .post(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "CONFIG_PARSE_ERROR");
    assert_eq!(body["error"]["type"], "config_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("error message")
            .contains("Config parse error")
    );
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

#[tokio::test]
async fn openai_chat_request_normalizes_json_object_response_format_for_responses_upstream() {
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
                "type": "json_object"
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
                "type": "json_object"
            }
        })
    );
}

#[tokio::test]
async fn openai_chat_request_keeps_string_tool_choice_auto_for_responses_upstream() {
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
            "tool_choice": "auto"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.json_body["tool_choice"], "auto");
}

#[tokio::test]
async fn openai_chat_request_keeps_string_tool_choice_none_for_responses_upstream() {
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
            "tool_choice": "none"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.json_body["tool_choice"], "none");
}

#[tokio::test]
async fn openai_chat_request_keeps_string_tool_choice_required_for_responses_upstream() {
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
            "tool_choice": "required"
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.json_body["tool_choice"], "required");
}

#[tokio::test]
async fn openai_chat_request_normalizes_tool_tool_choice_for_responses_upstream() {
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
                "type": "tool",
                "name": "lookup_weather"
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
async fn patch_method_is_rejected_to_match_python_router() {
    let (upstream_base, seen) = spawn_upstream_patch_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .patch(proxy_url)
        .json(&json!({
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.expect("json body");
    assert_eq!(body["detail"], "Method Not Allowed");
    assert!(seen.lock().expect("seen lock").is_empty());
}

#[tokio::test]
async fn proxy_entry_supports_head_like_fastapi_get_route() {
    let (upstream_base, seen) = spawn_upstream_head_echo_server().await;
    let proxy_base = spawn_proxy_server().await;
    let config = percent_encode(&json!({}).to_string());
    let proxy_url = format!("{proxy_base}/{config}${upstream_base}/v1/chat/completions");

    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .expect("reqwest client")
        .head(proxy_url)
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);
    assert!(seen.lock().expect("seen lock").last().is_some());
}

#[tokio::test]
async fn delete_request_body_is_ignored_to_match_python_router() {
    let (upstream_base, seen) = spawn_upstream_delete_echo_server().await;
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
        .delete(proxy_url)
        .header("content-type", "application/json")
        .body(
            json!({
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": "hi"}]
            })
            .to_string(),
        )
        .send()
        .await
        .expect("proxy response");

    assert_eq!(response.status(), StatusCode::OK);

    let seen = seen.lock().expect("seen lock");
    let request = seen.last().expect("upstream request");
    assert_eq!(request.path, "/v1/responses");
    assert_eq!(
        request.json_body,
        json!({
            "model": "",
            "stream": false,
            "input": ""
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

async fn spawn_upstream_delete_echo_server() -> (String, Arc<Mutex<Vec<SeenRequest>>>) {
    let state = UpstreamState {
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let seen = state.seen.clone();

    let app = Router::new()
        .route("/*path", delete(upstream_echo))
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

async fn spawn_upstream_patch_echo_server() -> (String, Arc<Mutex<Vec<SeenRequest>>>) {
    let state = UpstreamState {
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let seen = state.seen.clone();

    let app = Router::new()
        .route("/*path", patch(upstream_echo))
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

async fn spawn_upstream_head_echo_server() -> (String, Arc<Mutex<Vec<SeenRequest>>>) {
    let state = UpstreamState {
        seen: Arc::new(Mutex::new(Vec::new())),
    };
    let seen = state.seen.clone();

    let app = Router::new()
        .route("/*path", axum::routing::head(upstream_head_echo))
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

async fn upstream_head_echo(
    State(state): State<UpstreamState>,
    uri: Uri,
) -> impl IntoResponse {
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
            json_body: json!({}),
        });

    StatusCode::OK
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

async fn spawn_openai_chat_server(response_body: Value) -> String {
    spawn_delayed_openai_chat_server(response_body, Duration::ZERO).await
}

async fn spawn_delayed_openai_chat_server(response_body: Value, delay: Duration) -> String {
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let response_body = response_body.clone();
            async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                (StatusCode::OK, Json(response_body))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind openai chat server");
    let addr = listener.local_addr().expect("openai chat addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("openai chat server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    format!("http://{}", addr)
}

async fn spawn_counting_openai_chat_server(response_body: Value) -> (String, Arc<Mutex<usize>>) {
    let hits = Arc::new(Mutex::new(0usize));
    let app_hits = hits.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let response_body = response_body.clone();
            let hits = app_hits.clone();
            async move {
                *hits.lock().expect("hits lock") += 1;
                (StatusCode::OK, Json(response_body))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind counting openai chat server");
    let addr = listener.local_addr().expect("counting openai chat addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("counting openai chat server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    (format!("http://{}", addr), hits)
}

async fn spawn_sequenced_openai_chat_server(
    responses: Vec<(StatusCode, Value)>,
) -> (String, Arc<Mutex<usize>>) {
    let hits = Arc::new(Mutex::new(0usize));
    let remaining = Arc::new(Mutex::new(responses));
    let app_hits = hits.clone();
    let app_remaining = remaining.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let hits = app_hits.clone();
            let remaining = app_remaining.clone();
            async move {
                *hits.lock().expect("hits lock") += 1;
                let (status, body) = {
                    let mut guard = remaining.lock().expect("remaining lock");
                    if guard.is_empty() {
                        (StatusCode::INTERNAL_SERVER_ERROR, json!({"error": {"message": "no response scripted"}}))
                    } else {
                        guard.remove(0)
                    }
                };
                (status, Json(body))
            }
        }),
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind sequenced openai chat server");
    let addr = listener.local_addr().expect("sequenced openai chat addr");
    tokio::spawn(async move {
        axum::Server::from_tcp(listener.into_std().expect("std listener"))
            .expect("server from tcp")
            .serve(app.into_make_service())
            .await
            .expect("sequenced openai chat server");
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    (format!("http://{}", addr), hits)
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
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: "/tmp/prismguard-test.sock".to_string(),
        training_scheduler_enabled: true,
        training_scheduler_interval_minutes: 10,
        training_scheduler_failure_cooldown_minutes: 30,
        training_subprocess_allowed_cpus: "0".to_string(),
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
    .expect("write hashlinear runtime json");

    let mut bytes = Vec::with_capacity(coef.len() * std::mem::size_of::<f32>());
    for value in coef {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(profile_dir.join("hashlinear_runtime.coef.f32"), bytes)
        .expect("write hashlinear runtime coef");
}
