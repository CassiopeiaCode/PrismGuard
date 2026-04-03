use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Redirect};
use axum::routing::get;
#[cfg(all(feature = "storage-debug", not(test)))]
use axum::routing::post;
use axum::{Json, Router};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::Settings;
use crate::profile::ModerationProfile;
use crate::proxy::{proxy_entry, proxy_entry_root};
#[cfg(all(feature = "storage-debug", not(test)))]
use crate::storage::SampleStorage;
use crate::training::evaluate_training_need;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub http_client: Client,
}

pub fn router(state: AppState) -> Router {
    let router = Router::new()
        .route(
            "/openapi.json",
            get(openapi_json).fallback(method_not_allowed),
        )
        .route(
            "/docs/oauth2-redirect",
            get(docs_oauth2_redirect_html).fallback(method_not_allowed),
        )
        .route("/docs/", get(docs_redirect))
        .route("/docs", get(docs_html).fallback(method_not_allowed))
        .route("/redoc/", get(redoc_redirect))
        .route("/redoc", get(redoc_html).fallback(method_not_allowed))
        .route("/healthz", get(healthz).fallback(method_not_allowed))
        .route(
            "/debug/settings",
            get(debug_settings).fallback(method_not_allowed),
        )
        .route(
            "/debug/proxy-config/:key",
            get(debug_proxy_config).fallback(method_not_allowed),
        )
        .route(
            "/debug/profile/:profile",
            get(debug_profile).fallback(method_not_allowed),
        )
        .route(
            "/debug/url-config",
            get(debug_url_config).fallback(method_not_allowed),
        )
        .route(
            "/",
            get(proxy_entry_root)
                .post(proxy_entry_root)
                .put(proxy_entry_root)
                .delete(proxy_entry_root)
                .fallback(method_not_allowed),
        )
        .route(
            "/*cfg_and_upstream",
            get(proxy_entry)
                .post(proxy_entry)
                .put(proxy_entry)
                .delete(proxy_entry)
                .fallback(method_not_allowed),
        );

    attach_storage_debug_routes(router).with_state(state)
}

async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": "PrismGuard",
        "host": state.settings.host,
        "port": state.settings.port,
        "debug": state.settings.debug,
    }))
}

async fn openapi_json() -> impl IntoResponse {
    Json(json!({
        "openapi": "3.1.0",
        "info": {
            "title": "PrismGuard",
            "description": "高级 AI API 中间件 - 智能审核 · 格式转换 · 透明代理",
            "version": "1.0.0"
        },
        "paths": {
            "/openapi.json": {
                "get": {"summary": "OpenAPI Schema"}
            },
            "/docs": {
                "get": {"summary": "Swagger UI"}
            },
            "/docs/oauth2-redirect": {
                "get": {"summary": "Swagger UI OAuth2 Redirect"}
            },
            "/redoc": {
                "get": {"summary": "ReDoc"}
            },
            "/healthz": {
                "get": {"summary": "Health Check"}
            },
            "/debug/settings": {
                "get": {"summary": "Debug Settings"}
            },
            "/debug/proxy-config/{key}": {
                "get": {"summary": "Debug Proxy Config"}
            },
            "/debug/profile/{profile}": {
                "get": {"summary": "Debug Profile"}
            },
            "/debug/storage/{profile}/meta": {
                "get": {"summary": "Debug Storage Meta"}
            },
            "/debug/storage/{profile}/sample/{id}": {
                "get": {"summary": "Debug Storage Sample"}
            },
            "/debug/storage/{profile}/find-by-text": {
                "post": {"summary": "Debug Storage Find By Text"}
            },
            "/debug/url-config": {
                "get": {"summary": "Debug URL Config"}
            },
            "/{cfg_and_upstream}": {
                "get": {"summary": "Proxy Entry"},
                "post": {"summary": "Proxy Entry"},
                "put": {"summary": "Proxy Entry"},
                "delete": {"summary": "Proxy Entry"}
            }
        }
    }))
}

async fn docs_html() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        r#"<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>PrismGuard - Swagger UI</title>
  </head>
  <body>
    <h1>PrismGuard Swagger UI</h1>
    <p>Swagger UI schema: <a href="/openapi.json">/openapi.json</a></p>
  </body>
</html>"#,
    )
}

async fn docs_oauth2_redirect_html() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        r#"<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>PrismGuard - OAuth2 Redirect</title>
  </head>
  <body>
    <script>
      window.opener.postMessage({ source: "oauth2" }, "*");
    </script>
    <p>oauth2 redirect</p>
  </body>
</html>"#,
    )
}

async fn docs_redirect() -> Redirect {
    Redirect::temporary("/docs")
}

async fn redoc_html() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        r#"<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="utf-8" />
    <title>PrismGuard - ReDoc</title>
  </head>
  <body>
    <h1>PrismGuard ReDoc</h1>
    <p>ReDoc schema: <a href="/openapi.json">/openapi.json</a></p>
  </body>
</html>"#,
    )
}

async fn redoc_redirect() -> Redirect {
    Redirect::temporary("/redoc")
}

async fn debug_settings(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!(state.settings.as_ref()))
}

async fn method_not_allowed() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "detail": "Method Not Allowed"
        })),
    )
}

async fn debug_proxy_config(
    State(state): State<AppState>,
    Path(key): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let Some(value) = state.settings.parse_proxy_config(&key)? else {
        return Err(ApiError::new(
            StatusCode::NOT_FOUND,
            format!("env {key} not found"),
        ));
    };
    Ok(Json(json!({
        "key": key,
        "value": value
    })))
}

async fn debug_profile(
    State(state): State<AppState>,
    Path(profile): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let profile = ModerationProfile::load(&state.settings.root_dir, &profile)?;
    let model_path = match profile.config.local_model_type.as_str() {
        "fasttext" => profile.fasttext_model_path(),
        "hashlinear" => profile.hashlinear_model_path(),
        _ => profile.bow_model_path(),
    };
    let model_mtime = std::fs::metadata(&model_path)
        .ok()
        .and_then(|meta| meta.modified().ok());
    let sample_count = profile
        .training_status()
        .as_ref()
        .and_then(|status| status.get("sample_count"))
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let training_decision = evaluate_training_need(
        &profile,
        sample_count,
        model_mtime,
        std::time::SystemTime::now(),
    )
    .ok();
    Ok(Json(json!({
        "profile_name": profile.profile_name,
        "base_dir": profile.base_dir,
        "history_rocks_path": profile.history_rocks_path(),
        "training_status_path": profile.training_status_path(),
        "bow_model_path": profile.bow_model_path(),
        "bow_vectorizer_path": profile.bow_vectorizer_path(),
        "fasttext_model_path": profile.fasttext_model_path(),
        "hashlinear_model_path": profile.hashlinear_model_path(),
        "local_model_exists": profile.local_model_exists(),
        "training_status": profile.training_status(),
        "training_decision": training_decision,
        "config": profile.config
    })))
}

#[cfg(all(feature = "storage-debug", not(test)))]
async fn debug_storage_meta(
    State(state): State<AppState>,
    Path(profile): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let profile = ModerationProfile::load(&state.settings.root_dir, &profile)?;
    let storage = SampleStorage::open_read_only(profile.history_rocks_path())?;
    Ok(Json(json!({
        "profile_name": profile.profile_name,
        "meta": storage.metadata(),
        "preview": storage.first_samples(3),
    })))
}

#[cfg(all(feature = "storage-debug", not(test)))]
async fn debug_storage_sample(
    State(state): State<AppState>,
    Path((profile, id)): Path<(String, u64)>,
) -> Result<Json<Value>, ApiError> {
    let profile = ModerationProfile::load(&state.settings.root_dir, &profile)?;
    let storage = SampleStorage::open_read_only(profile.history_rocks_path())?;
    let sample = storage.sample_by_id(id)?;
    Ok(Json(json!({
        "profile_name": profile.profile_name,
        "sample": sample,
    })))
}

#[cfg_attr(any(test, not(feature = "storage-debug")), allow(dead_code))]
#[derive(Debug, Deserialize)]
struct FindByTextRequest {
    text: String,
}

#[cfg(all(feature = "storage-debug", not(test)))]
async fn debug_find_by_text(
    State(state): State<AppState>,
    Path(profile): Path<String>,
    Json(payload): Json<FindByTextRequest>,
) -> Result<Json<Value>, ApiError> {
    let profile = ModerationProfile::load(&state.settings.root_dir, &profile)?;
    let storage = SampleStorage::open_read_only(profile.history_rocks_path())?;
    let sample = storage.find_by_text(&payload.text)?;
    Ok(Json(json!({
        "profile_name": profile.profile_name,
        "sample": sample,
    })))
}

#[cfg(all(feature = "storage-debug", not(test)))]
fn attach_storage_debug_routes(router: Router<AppState>) -> Router<AppState> {
    router
        .route(
            "/debug/storage/:profile/meta",
            get(debug_storage_meta).fallback(method_not_allowed),
        )
        .route(
            "/debug/storage/:profile/sample/:id",
            get(debug_storage_sample).fallback(method_not_allowed),
        )
        .route(
            "/debug/storage/:profile/find-by-text",
            post(debug_find_by_text).fallback(method_not_allowed),
        )
}

#[cfg(any(test, not(feature = "storage-debug")))]
fn attach_storage_debug_routes(router: Router<AppState>) -> Router<AppState> {
    router
}

#[derive(Debug, Deserialize)]
struct UrlConfigQuery {
    value: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct ParsedUrlConfig {
    pub(crate) config: Value,
    pub(crate) upstream: String,
    pub(crate) source: String,
}

async fn debug_url_config(
    State(state): State<AppState>,
    Query(query): Query<UrlConfigQuery>,
) -> Result<Json<ParsedUrlConfig>, ApiError> {
    let parsed = parse_url_config(&state.settings, &query.value)?;
    Ok(Json(parsed))
}

pub fn parse_url_config(settings: &Settings, raw: &str) -> Result<ParsedUrlConfig, ApiError> {
    let (cfg_part, upstream) = raw
        .split_once('$')
        .ok_or_else(|| config_parse_error("expected {config}${upstream}"))?;

    if cfg_part.starts_with('!') {
        let key = &cfg_part[1..];
        let parsed = settings.parse_proxy_config(key).map_err(|error| {
            ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("{error:#}"),
            )
            .with_code("CONFIG_PARSE_ERROR")
            .with_error_type("config_error")
        })?;
        let Some(config) = parsed else {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("Environment variable {key} not found"),
            )
            .with_code("CONFIG_PARSE_ERROR")
            .with_error_type("config_error"));
        };
        return Ok(ParsedUrlConfig {
            config,
            upstream: upstream.to_string(),
            source: format!("env:{key}"),
        });
    }

    let decoded = percent_decode(cfg_part)?;
    let config = serde_json::from_str(&decoded).map_err(|e| {
        config_parse_error(format!("Config parse error: {e}"))
    })?;
    Ok(ParsedUrlConfig {
        config,
        upstream: upstream.to_string(),
        source: "inline".to_string(),
    })
}

fn percent_decode(input: &str) -> Result<String, ApiError> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'%' if idx + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[idx + 1..idx + 3]).map_err(|e| {
                    config_parse_error(format!("invalid percent encoding: {e}"))
                })?;
                let value = u8::from_str_radix(hex, 16).map_err(|e| {
                    config_parse_error(format!("invalid percent encoding: {e}"))
                })?;
                out.push(value);
                idx += 3;
            }
            byte => {
                out.push(byte);
                idx += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|e| {
        config_parse_error(format!("decoded config is not utf-8: {e}"))
    })
}

fn config_parse_error(message: impl Into<String>) -> ApiError {
    ApiError::new(StatusCode::BAD_REQUEST, message)
        .with_code("CONFIG_PARSE_ERROR")
        .with_error_type("config_error")
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    code: &'static str,
    error_type: &'static str,
    source_format: Option<String>,
    moderation_details: Option<Value>,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: "INTERNAL_ERROR",
            error_type: "proxy_error",
            source_format: None,
            moderation_details: None,
        }
    }

    pub(crate) fn with_code(mut self, code: &'static str) -> Self {
        self.code = code;
        self
    }

    pub(crate) fn with_error_type(mut self, error_type: &'static str) -> Self {
        self.error_type = error_type;
        self
    }

    pub(crate) fn with_source_format(mut self, source_format: Option<impl Into<String>>) -> Self {
        self.source_format = source_format.map(Into::into);
        self
    }

    pub(crate) fn with_moderation_details(mut self, moderation_details: Option<Value>) -> Self {
        self.moderation_details = moderation_details;
        self
    }

    pub(crate) fn moderation_blocked(
        message: impl Into<String>,
        source_format: Option<impl Into<String>>,
        moderation_details: Option<Value>,
    ) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
            .with_code("MODERATION_BLOCKED")
            .with_error_type("moderation_error")
            .with_source_format(source_format)
            .with_moderation_details(moderation_details)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let mut error = serde_json::Map::new();
        error.insert("code".to_string(), Value::String(self.code.to_string()));
        error.insert("message".to_string(), Value::String(self.message));
        error.insert("type".to_string(), Value::String(self.error_type.to_string()));
        if let Some(source_format) = self.source_format {
            error.insert("source_format".to_string(), Value::String(source_format));
        }
        if let Some(moderation_details) = self.moderation_details {
            error.insert("moderation_details".to_string(), moderation_details);
        }
        let body = Json(json!({ "error": Value::Object(error) }));
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{value:#}"))
            .with_code("PROXY_ERROR")
            .with_error_type("proxy_error")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::to_bytes;
    use std::collections::HashMap;
    use std::path::PathBuf;

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

    #[tokio::test]
    async fn api_error_includes_optional_proxy_metadata() {
        let response = ApiError::new(StatusCode::BAD_REQUEST, "blocked")
            .with_code("MODERATION_BLOCKED")
            .with_error_type("moderation_error")
            .with_source_format(Some("claude_chat"))
            .with_moderation_details(Some(json!({
                "source": "concurrency_limit"
            })))
            .into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body()).await.expect("body bytes");
        let payload: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(payload["error"]["source_format"], "claude_chat");
        assert_eq!(payload["error"]["moderation_details"]["source"], "concurrency_limit");
    }

    #[tokio::test]
    async fn moderation_blocked_helper_uses_existing_error_shape() {
        let response = ApiError::moderation_blocked(
            "blocked by moderation",
            Some("openai_chat"),
            Some(json!({
                "source": "hashlinear_model"
            })),
        )
        .into_response();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);

        let body = to_bytes(response.into_body()).await.expect("body bytes");
        let payload: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(payload["error"]["code"], "MODERATION_BLOCKED");
        assert_eq!(payload["error"]["type"], "moderation_error");
        assert_eq!(payload["error"]["source_format"], "openai_chat");
        assert_eq!(payload["error"]["moderation_details"]["source"], "hashlinear_model");
    }

    #[tokio::test]
    async fn anyhow_conversion_uses_python_style_proxy_error_code() {
        let response = ApiError::from(anyhow::anyhow!("boom")).into_response();

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);

        let body = to_bytes(response.into_body()).await.expect("body bytes");
        let payload: Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(payload["error"]["code"], "PROXY_ERROR");
        assert_eq!(payload["error"]["type"], "proxy_error");
        assert_eq!(payload["error"]["message"], "boom");
    }

    #[test]
    fn parse_url_config_preserves_plus_signs_in_inline_config() {
        let settings = test_settings();
        let parsed = parse_url_config(
            &settings,
            "%7B%22label%22%3A%22a+b%22%7D$https://example.com/v1/chat/completions",
        )
        .expect("parsed config");

        assert_eq!(parsed.config["label"], "a+b");
        assert_eq!(parsed.upstream, "https://example.com/v1/chat/completions");
        assert_eq!(parsed.source, "inline");
    }
}
