use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{any, get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::Settings;
use crate::profile::ModerationProfile;
use crate::proxy::proxy_entry;
use crate::storage::SampleStorage;

#[derive(Clone)]
pub struct AppState {
    pub settings: Arc<Settings>,
    pub http_client: Client,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/debug/settings", get(debug_settings))
        .route("/debug/proxy-config/:key", get(debug_proxy_config))
        .route("/debug/profile/:profile", get(debug_profile))
        .route("/debug/storage/:profile/meta", get(debug_storage_meta))
        .route("/debug/storage/:profile/sample/:id", get(debug_storage_sample))
        .route("/debug/storage/:profile/find-by-text", post(debug_find_by_text))
        .route("/debug/url-config", get(debug_url_config))
        .route("/*cfg_and_upstream", any(proxy_entry))
        .with_state(state)
}

async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "ok": true,
        "service": "Prismguand-Rust",
        "host": state.settings.host,
        "port": state.settings.port,
        "debug": state.settings.debug,
    }))
}

async fn debug_settings(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!(state.settings.as_ref()))
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
    Ok(Json(json!({
        "profile_name": profile.profile_name,
        "base_dir": profile.base_dir,
        "history_rocks_path": profile.history_rocks_path(),
        "config": profile.config
    })))
}

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

#[derive(Debug, Deserialize)]
struct FindByTextRequest {
    text: String,
}

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
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "expected {config}${upstream}"))?;

    if cfg_part.starts_with('!') {
        let key = &cfg_part[1..];
        let Some(config) = settings.parse_proxy_config(key)? else {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                format!("Environment variable {key} not found"),
            ));
        };
        return Ok(ParsedUrlConfig {
            config,
            upstream: upstream.to_string(),
            source: format!("env:{key}"),
        });
    }

    let decoded = percent_decode(cfg_part)?;
    let config = serde_json::from_str(&decoded).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("Config parse error: {e}"),
        )
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
                    ApiError::new(StatusCode::BAD_REQUEST, format!("invalid percent encoding: {e}"))
                })?;
                let value = u8::from_str_radix(hex, 16).map_err(|e| {
                    ApiError::new(StatusCode::BAD_REQUEST, format!("invalid percent encoding: {e}"))
                })?;
                out.push(value);
                idx += 3;
            }
            b'+' => {
                out.push(b' ');
                idx += 1;
            }
            byte => {
                out.push(byte);
                idx += 1;
            }
        }
    }
    String::from_utf8(out).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("decoded config is not utf-8: {e}"),
        )
    })
}

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    code: &'static str,
    error_type: &'static str,
}

impl ApiError {
    pub(crate) fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            code: "INTERNAL_ERROR",
            error_type: "proxy_error",
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
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = Json(json!({
            "error": {
                "code": self.code,
                "message": self.message,
                "type": self.error_type,
            }
        }));
        (self.status, body).into_response()
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(value: anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, format!("{value:#}"))
    }
}
