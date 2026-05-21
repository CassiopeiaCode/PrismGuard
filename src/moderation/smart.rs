use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use reqwest::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::timeout as tokio_timeout;

use crate::moderation::bow;
use crate::moderation::fasttext;
use crate::moderation::hashlinear;
use crate::profile::ModerationProfile;
#[cfg(feature = "storage-debug")]
use crate::storage::SampleStorage;

const MAX_LLM_CONCURRENCY: usize = 5;
const CACHE_SIZE: usize = 1024;
const MAX_CACHE_PROFILES: usize = 50;

static LLM_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();
static MODERATION_CACHE: OnceLock<
    Mutex<IndexMap<String, IndexMap<String, SmartModerationResult>>>,
> = OnceLock::new();
#[cfg(feature = "storage-debug")]
static HISTORY_STORAGE_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
static REQUEST_RANDOM_STATE: OnceLock<AtomicU64> = OnceLock::new();
#[cfg(test)]
use std::collections::VecDeque;
#[cfg(test)]
static TEST_RANDOM_VALUES: OnceLock<Mutex<VecDeque<f64>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct SmartModerationResult {
    pub violation: bool,
    pub category: Option<String>,
    pub reason: Option<String>,
    pub source: String,
    pub confidence: Option<f64>,
    #[serde(skip_serializing)]
    pub debug: Option<ModerationDebugInfo>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ModerationDebugInfo {
    pub local_model_source: Option<String>,
    pub local_model_confidence: Option<f64>,
    pub local_model_latency_ms: Option<f64>,
    pub local_model_decision: Option<String>,
    pub llm_reviewed: bool,
    pub llm_result: Option<String>,
    pub llm_latency_ms: Option<f64>,
    pub llm_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalDecision {
    Allow,
    Block,
    Uncertain,
}

#[derive(Debug, Clone)]
struct LocalModelEvaluation {
    source: String,
    probability: f64,
    decision: LocalDecision,
    latency_ms: f64,
}

impl LocalModelEvaluation {
    fn debug_info(&self) -> ModerationDebugInfo {
        let local_model_decision = match self.decision {
            LocalDecision::Allow => "allow",
            LocalDecision::Block => "block",
            LocalDecision::Uncertain => "uncertain",
        };
        ModerationDebugInfo {
            local_model_source: Some(self.source.clone()),
            local_model_confidence: Some(self.probability),
            local_model_latency_ms: Some(self.latency_ms),
            local_model_decision: Some(local_model_decision.to_string()),
            ..Default::default()
        }
    }

    fn direct_result(&self) -> Option<SmartModerationResult> {
        match self.decision {
            LocalDecision::Allow => Some(SmartModerationResult {
                violation: false,
                category: None,
                reason: Some(format!(
                    "{}: low risk (p={:.3})",
                    self.source, self.probability
                )),
                source: self.source.clone(),
                confidence: Some(self.probability),
                debug: None,
            }),
            LocalDecision::Block => Some(SmartModerationResult {
                violation: true,
                category: None,
                reason: Some(format!(
                    "{}: high risk (p={:.3})",
                    self.source, self.probability
                )),
                source: self.source.clone(),
                confidence: Some(self.probability),
                debug: None,
            }),
            LocalDecision::Uncertain => None,
        }
    }

    fn fallback_result(&self, profile: &ModerationProfile) -> SmartModerationResult {
        let threshold = fallback_threshold(profile);
        let violation = self.probability >= threshold;
        SmartModerationResult {
            violation,
            category: None,
            reason: Some(format!(
                "{}: llm concurrency fallback (p={:.3}, threshold={threshold:.3})",
                self.source, self.probability
            )),
            source: "local_fallback".to_string(),
            confidence: Some(self.probability),
            debug: None,
        }
    }
}

#[derive(Debug)]
pub enum SmartModerationError {
    ConcurrencyLimit(String),
    Other(anyhow::Error, Option<ModerationDebugInfo>),
}

impl From<anyhow::Error> for SmartModerationError {
    fn from(value: anyhow::Error) -> Self {
        Self::Other(value, None)
    }
}

pub async fn smart_moderation(
    text: &str,
    cfg: &Value,
    root_dir: &Path,
    http_client: &Client,
    env_map: &HashMap<String, String>,
) -> Result<Option<SmartModerationResult>, SmartModerationError> {
    if !cfg.get("enabled").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(None);
    }

    let profile_name = cfg
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("default");
    let profile = ModerationProfile::load(root_dir, profile_name)?;
    let text = profile.truncate_text(text);

    if let Some(result) = check_cache(profile_name, &text) {
        return Ok(Some(result));
    }

    let result = decide_moderation(&text, &profile, http_client, env_map).await?;
    save_cache(profile_name, &text, &result);
    Ok(Some(result))
}

pub fn llm_semaphore() -> Arc<Semaphore> {
    LLM_SEMAPHORE
        .get_or_init(|| Arc::new(Semaphore::new(MAX_LLM_CONCURRENCY)))
        .clone()
}

pub fn try_acquire_llm_slot() -> Result<OwnedSemaphorePermit, SmartModerationError> {
    llm_semaphore().try_acquire_owned().map_err(|_| {
        SmartModerationError::ConcurrencyLimit(format!(
            "LLM审核并发数超限(max={MAX_LLM_CONCURRENCY})"
        ))
    })
}

async fn decide_moderation(
    text: &str,
    profile: &ModerationProfile,
    http_client: &Client,
    env_map: &HashMap<String, String>,
) -> Result<SmartModerationResult, SmartModerationError> {
    if should_force_ai_review(profile) {
        let (local_evaluation, debug) =
            optional_local_evaluation_for_concurrency_fallback(text, profile);
        let inherited_debug = (!is_empty_debug(&debug)).then_some(debug);
        return run_ai_moderation_with_history(
            text,
            profile,
            http_client,
            env_map,
            local_evaluation,
            inherited_debug,
        )
        .await;
    }

    if profile.local_model_exists() {
        let (evaluation, debug) = evaluate_local_model_state(text, profile);
        if let Some(evaluation) = evaluation {
            if let Some(mut result) = evaluation.direct_result() {
                result.debug = Some(debug);
                return Ok(result);
            }

            return run_ai_moderation_with_history(
                text,
                profile,
                http_client,
                env_map,
                Some(evaluation),
                Some(debug),
            )
            .await;
        }

        return run_ai_moderation_with_history(
            text,
            profile,
            http_client,
            env_map,
            None,
            Some(debug),
        )
        .await;
    }

    run_ai_moderation_with_history(text, profile, http_client, env_map, None, None).await
}

fn should_use_concurrency_limit_fallback(profile: &ModerationProfile) -> bool {
    profile.config.probability.enable_concurrency_limit_fallback
}

fn fallback_threshold(profile: &ModerationProfile) -> f64 {
    let low = profile.config.probability.low_risk_threshold;
    let high = profile.config.probability.high_risk_threshold;
    (low + high) / 2.0
}

fn is_empty_debug(debug: &ModerationDebugInfo) -> bool {
    debug.local_model_source.is_none()
        && debug.local_model_confidence.is_none()
        && debug.local_model_latency_ms.is_none()
        && debug.local_model_decision.is_none()
        && !debug.llm_reviewed
        && debug.llm_result.is_none()
        && debug.llm_latency_ms.is_none()
        && debug.llm_error.is_none()
}

fn optional_local_evaluation_for_concurrency_fallback(
    text: &str,
    profile: &ModerationProfile,
) -> (Option<LocalModelEvaluation>, ModerationDebugInfo) {
    if should_use_concurrency_limit_fallback(profile) && profile.local_model_exists() {
        return evaluate_local_model_state(text, profile);
    }

    (None, ModerationDebugInfo::default())
}

fn evaluate_local_model(
    text: &str,
    profile: &ModerationProfile,
) -> (Option<SmartModerationResult>, ModerationDebugInfo) {
    let (evaluation, debug) = evaluate_local_model_state(text, profile);
    (evaluation.and_then(|item| item.direct_result()), debug)
}

fn evaluate_local_model_state(
    text: &str,
    profile: &ModerationProfile,
) -> (Option<LocalModelEvaluation>, ModerationDebugInfo) {
    let started = Instant::now();
    let (probability, source) = match profile.config.local_model_type.as_str() {
        "fasttext" => match fasttext::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "fasttext_model"),
            Ok(None) | Err(_) => {
                return (
                    None,
                    ModerationDebugInfo {
                        local_model_source: Some("fasttext_model".to_string()),
                        local_model_decision: Some("skipped".to_string()),
                        ..Default::default()
                    },
                )
            }
        },
        "hashlinear" => match hashlinear::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "hashlinear_model"),
            Ok(None) | Err(_) => {
                return (
                    None,
                    ModerationDebugInfo {
                        local_model_source: Some("hashlinear_model".to_string()),
                        local_model_decision: Some("skipped".to_string()),
                        ..Default::default()
                    },
                )
            }
        },
        _ => match bow::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "bow_model"),
            Ok(None) | Err(_) => {
                return (
                    None,
                    ModerationDebugInfo {
                        local_model_source: Some("bow_model".to_string()),
                        local_model_decision: Some("skipped".to_string()),
                        ..Default::default()
                    },
                )
            }
        },
    };
    let low = profile.config.probability.low_risk_threshold;
    let high = profile.config.probability.high_risk_threshold;
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let decision = if probability < low {
        LocalDecision::Allow
    } else if probability > high {
        LocalDecision::Block
    } else {
        LocalDecision::Uncertain
    };
    let evaluation = LocalModelEvaluation {
        source: source.to_string(),
        probability,
        decision,
        latency_ms,
    };
    let debug = evaluation.debug_info();
    (Some(evaluation), debug)
}

fn should_force_ai_review(profile: &ModerationProfile) -> bool {
    let rate = profile.config.probability.ai_review_rate;
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }

    next_random_unit() < rate
}

async fn llm_moderate(
    text: &str,
    profile: &ModerationProfile,
    http_client: &Client,
    env_map: &HashMap<String, String>,
) -> Result<SmartModerationResult, SmartModerationError> {
    let _permit = try_acquire_llm_slot()?;

    let api_key_env = profile.config.ai.api_key_env.as_str();
    let api_key = env_map
        .get(api_key_env)
        .cloned()
        .or_else(|| std::env::var(api_key_env).ok())
        .ok_or_else(|| anyhow!("environment variable {api_key_env} not set"))?;
    let models = profile.config.ai.get_model_candidates();
    if models.is_empty() {
        return Err(anyhow!("AI config model is empty").into());
    }

    let prompt = profile.render_prompt(text);
    let timeout = Duration::from_secs(profile.config.ai.timeout.max(1));
    let max_retries = profile.config.ai.max_retries as usize;
    let endpoint = format!(
        "{}/chat/completions",
        profile.config.ai.base_url.trim_end_matches('/')
    );

    // NOTE:
    // - Always request a streaming response (SSE) so timing out will cancel the in-flight body read,
    //   which causes the underlying HTTP connection to be dropped instead of being left hanging.
    // - Enforce timeout with `tokio::time::timeout` so it cannot "sometimes not timeout".
    async fn parse_llm_response(
        response: reqwest::Response,
    ) -> Result<SmartModerationResult, SmartModerationError> {
        let response = response
            .error_for_status()
            .map_err(|err| SmartModerationError::Other(anyhow!(err), None))?;

        let is_sse = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.starts_with("text/event-stream"))
            .unwrap_or(false);

        if is_sse {
            let content = read_openai_chat_sse_content(response)
                .await
                .map_err(|err| SmartModerationError::Other(err, None))?;
            parse_moderation_content(&content)
        } else {
            let payload = response
                .json::<Value>()
                .await
                .map_err(|err| SmartModerationError::Other(anyhow!(err), None))?;
            parse_openai_moderation_response(payload)
        }
    }

    let mut attempted_models = Vec::new();
    let mut last_error = None;
    let started = Instant::now();
    for _attempt in 0..=max_retries {
        let model = pick_model_for_attempt(&models, &attempted_models);
        attempted_models.push(model.clone());

        let response = tokio_timeout(timeout, async {
            let response = http_client
                .post(&endpoint)
                .bearer_auth(&api_key)
                .json(&json!({
                    "model": &model,
                    "messages": [{
                        "role": "user",
                        "content": prompt
                    }],
                    "temperature": 0,
                    "stream": true
                }))
                .send()
                .await
                .map_err(|err| SmartModerationError::Other(anyhow!(err), None))?;

            parse_llm_response(response).await
        })
        .await;

        match response {
            Ok(Ok(mut parsed)) => {
                parsed.debug = Some(ModerationDebugInfo {
                    llm_reviewed: true,
                    llm_result: Some(if parsed.violation {
                        "block".to_string()
                    } else {
                        "allow".to_string()
                    }),
                    llm_latency_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
                    ..Default::default()
                });
                return Ok(parsed);
            }
            Ok(Err(err)) => {
                let err = match err {
                    SmartModerationError::ConcurrencyLimit(message) => anyhow!(message),
                    SmartModerationError::Other(err, _) => err,
                };
                last_error = Some(err.context("llm moderation request failed"));
            }
            Err(_) => {
                last_error = Some(anyhow!(
                    "llm moderation request timed out after {}s (connection aborted)",
                    timeout.as_secs().max(1)
                ));
            }
        }
    }

    Err(anyhow!(
        "llm moderation request failed: {}",
        last_error
            .map(|err| format!("{err:#}"))
            .unwrap_or_else(|| "unknown error".to_string())
    )
    .into())
}

async fn read_openai_chat_sse_content(response: reqwest::Response) -> Result<String> {
    use futures_util::StreamExt;

    let mut stream = response.bytes_stream();
    let mut buffer: Vec<u8> = Vec::new();
    let mut content = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        buffer.extend_from_slice(&chunk);

        while let Some(newline_idx) = buffer.iter().position(|b| *b == b'\n') {
            let mut line = buffer.drain(..=newline_idx).collect::<Vec<u8>>();
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }

            if line.is_empty() {
                continue;
            }

            let Ok(line) = std::str::from_utf8(&line) else {
                continue;
            };
            let trimmed = line.trim();
            let Some(data) = trimmed.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                return Ok(content);
            }

            let Ok(payload) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(delta) = extract_openai_chat_stream_delta(&payload) {
                content.push_str(&delta);
            }
        }
    }

    Ok(content)
}

fn extract_openai_chat_stream_delta(payload: &Value) -> Option<String> {
    let choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())?;

    if let Some(delta) = choice.get("delta") {
        if let Some(content) = delta.get("content") {
            let text = extract_content_text(content);
            if !text.is_empty() {
                return Some(text);
            }
        }
    }

    // Some upstreams may return a nonstandard streaming shape.
    if let Some(message) = choice.get("message") {
        if let Some(content) = message.get("content") {
            let text = extract_content_text(content);
            if !text.is_empty() {
                return Some(text);
            }
        }
    }

    None
}

async fn run_ai_moderation_with_history(
    text: &str,
    profile: &ModerationProfile,
    http_client: &Client,
    env_map: &HashMap<String, String>,
    local_evaluation: Option<LocalModelEvaluation>,
    inherited_debug: Option<ModerationDebugInfo>,
) -> Result<SmartModerationResult, SmartModerationError> {
    if let Some(mut result) = load_history_result(text, profile).await? {
        let mut debug = inherited_debug.unwrap_or_default();
        debug.llm_reviewed = true;
        debug.llm_result = Some(if result.violation {
            "block".to_string()
        } else {
            "allow".to_string()
        });
        result.debug = Some(debug);
        return Ok(result);
    }
    let mut result = match llm_moderate(text, profile, http_client, env_map).await {
        Ok(result) => result,
        Err(SmartModerationError::ConcurrencyLimit(message))
            if should_use_concurrency_limit_fallback(profile) && local_evaluation.is_some() =>
        {
            let mut result = local_evaluation
                .as_ref()
                .expect("local evaluation checked above")
                .fallback_result(profile);
            let mut debug = inherited_debug.clone().unwrap_or_default();
            debug.llm_reviewed = true;
            debug.llm_result = Some("error".to_string());
            debug.llm_error = Some(truncate_header_value(&message));
            result.debug = Some(debug);
            return Ok(result);
        }
        Err(err) => {
            let mut debug = inherited_debug.clone().unwrap_or_default();
            debug.llm_reviewed = true;
            debug.llm_result = Some("error".to_string());
            debug.llm_error = Some(truncate_header_value(&match &err {
                SmartModerationError::ConcurrencyLimit(message) => message.clone(),
                SmartModerationError::Other(error, _) => format!("{error:#}"),
            }));
            return Err(match err {
                SmartModerationError::ConcurrencyLimit(message) => {
                    SmartModerationError::Other(anyhow!(message), Some(debug))
                }
                SmartModerationError::Other(error, _) => {
                    SmartModerationError::Other(error, Some(debug))
                }
            });
        }
    };
    let mut debug = inherited_debug.unwrap_or_default();
    let latency = result.debug.as_ref().and_then(|value| value.llm_latency_ms);
    debug.llm_reviewed = true;
    debug.llm_result = Some(if result.violation {
        "block".to_string()
    } else {
        "allow".to_string()
    });
    debug.llm_latency_ms = latency;
    result.debug = Some(debug);
    save_history_result(text, profile, &result).await?;
    Ok(result)
}

#[cfg(feature = "storage-debug")]
async fn load_history_result(
    text: &str,
    profile: &ModerationProfile,
) -> Result<Option<SmartModerationResult>, SmartModerationError> {
    let text = text.to_string();
    let profile = profile.clone();
    tokio::task::spawn_blocking(move || {
        let storage_lock = history_storage_lock(&profile.profile_name);
        let _guard = storage_lock.lock().expect("history storage lock");
        let storage = match SampleStorage::open_read_only(profile.history_rocks_path()) {
            Ok(storage) => storage,
            Err(_) => return Ok(None),
        };
        let Some(sample) = storage.find_by_text(&text)? else {
            return Ok(None);
        };

        Ok(Some(SmartModerationResult {
            violation: sample
                .value
                .get("label")
                .and_then(Value::as_i64)
                .unwrap_or_default()
                != 0,
            category: sample
                .value
                .get("category")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            reason: sample
                .value
                .get("created_at")
                .and_then(Value::as_str)
                .map(|created_at| format!("From DB: {created_at}")),
            source: "ai".to_string(),
            confidence: None,
            debug: None,
        }))
    })
    .await
    .map_err(|err| {
        SmartModerationError::Other(anyhow!("history storage read task failed: {err}"), None)
    })?
}

#[cfg(not(feature = "storage-debug"))]
async fn load_history_result(
    _text: &str,
    _profile: &ModerationProfile,
) -> Result<Option<SmartModerationResult>, SmartModerationError> {
    Ok(None)
}

#[cfg(feature = "storage-debug")]
async fn save_history_result(
    text: &str,
    profile: &ModerationProfile,
    result: &SmartModerationResult,
) -> Result<(), SmartModerationError> {
    let text = text.to_string();
    let profile = profile.clone();
    let result = result.clone();
    tokio::task::spawn_blocking(move || {
        let storage_lock = history_storage_lock(&profile.profile_name);
        let _guard = storage_lock.lock().expect("history storage lock");
        let storage = SampleStorage::open_read_write(profile.history_rocks_path())?;
        let label = if result.violation { 1 } else { 0 };
        storage.save_sample(&text, label, result.category.as_deref())?;
        Ok(())
    })
    .await
    .map_err(|err| {
        SmartModerationError::Other(anyhow!("history storage write task failed: {err}"), None)
    })?
}

#[cfg(not(feature = "storage-debug"))]
async fn save_history_result(
    _text: &str,
    _profile: &ModerationProfile,
    _result: &SmartModerationResult,
) -> Result<(), SmartModerationError> {
    Ok(())
}

fn parse_openai_moderation_response(
    payload: Value,
) -> Result<SmartModerationResult, SmartModerationError> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .map(extract_content_text)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("llm moderation response missing choices[0].message.content"))?;

    parse_moderation_content(&content)
}

fn extract_content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.to_string(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    part.get("text")
                        .and_then(Value::as_str)
                        .map(ToString::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_moderation_content(content: &str) -> Result<SmartModerationResult, SmartModerationError> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| anyhow!("llm moderation content is not a JSON object"))?;
    let data: Value = serde_json::from_str(json_text)
        .map_err(|err| anyhow!(err).context("failed to decode moderation JSON"))?;

    Ok(SmartModerationResult {
        violation: data
            .get("violation")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        category: data
            .get("category")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        reason: data
            .get("reason")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        source: "ai".to_string(),
        confidence: data.get("confidence").and_then(Value::as_f64),
        debug: None,
    })
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end > start).then_some(&content[start..=end])
}

fn check_cache(profile_name: &str, text: &str) -> Option<SmartModerationResult> {
    let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(IndexMap::new()));
    let key = cache_key(text);
    let mut guard = cache.lock().expect("moderation cache");
    let profile_cache = guard.get_mut(profile_name)?;
    let mut result = profile_cache.shift_remove(&key)?;
    profile_cache.insert(key, result.clone());
    result.debug = None;
    Some(result)
}

fn save_cache(profile_name: &str, text: &str, result: &SmartModerationResult) {
    let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(IndexMap::new()));
    let key = cache_key(text);
    let mut guard = cache.lock().expect("moderation cache");
    if !guard.contains_key(profile_name) && guard.len() >= MAX_CACHE_PROFILES {
        guard.shift_remove_index(0);
    }
    let profile_cache = guard
        .entry(profile_name.to_string())
        .or_insert_with(IndexMap::new);
    profile_cache.shift_remove(&key);
    let mut cached = result.clone();
    cached.debug = None;
    profile_cache.insert(key, cached);
    while profile_cache.len() > CACHE_SIZE {
        profile_cache.shift_remove_index(0);
    }
}

fn truncate_header_value(value: &str) -> String {
    value.chars().take(160).collect()
}

fn cache_key(text: &str) -> String {
    format!("{:x}", md5::compute(text.as_bytes()))
}

fn next_random_unit() -> f64 {
    #[cfg(test)]
    if let Some(value) = take_test_random_value() {
        return value.clamp(0.0, 1.0 - f64::EPSILON);
    }

    let state = REQUEST_RANDOM_STATE.get_or_init(|| AtomicU64::new(initial_random_seed()));
    let mut current = state.load(Ordering::Relaxed);
    loop {
        let mut next = current;
        next ^= next >> 12;
        next ^= next << 25;
        next ^= next >> 27;
        next = next.wrapping_mul(0x2545_F491_4F6C_DD1D);
        match state.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => return ((next >> 11) as f64) / ((1u64 << 53) as f64),
            Err(observed) => current = observed,
        }
    }
}

fn initial_random_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    nanos ^ 0xA076_1D64_78BD_642F
}

fn pick_model_for_attempt(models: &[String], attempted_models: &[String]) -> String {
    let remaining = models
        .iter()
        .filter(|model| !attempted_models.iter().any(|attempted| attempted == *model))
        .map(String::as_str)
        .collect::<Vec<_>>();
    let pool = if remaining.is_empty() {
        models.iter().map(String::as_str).collect::<Vec<_>>()
    } else {
        remaining
    };
    let index = ((next_random_unit() * pool.len() as f64).floor() as usize).min(pool.len() - 1);
    pool[index].to_string()
}

#[cfg(feature = "storage-debug")]
fn history_storage_lock(profile_name: &str) -> Arc<Mutex<()>> {
    let locks = HISTORY_STORAGE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = locks.lock().expect("history storage locks");
    guard
        .entry(profile_name.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[cfg(test)]
fn take_test_random_value() -> Option<f64> {
    let queue = TEST_RANDOM_VALUES.get_or_init(|| Mutex::new(VecDeque::new()));
    queue.lock().expect("test random values").pop_front()
}

#[cfg(test)]
pub fn set_test_random_values(values: &[f64]) {
    let queue = TEST_RANDOM_VALUES.get_or_init(|| Mutex::new(VecDeque::new()));
    let mut guard = queue.lock().expect("test random values");
    guard.clear();
    guard.extend(values.iter().copied());
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Mutex;

    use indexmap::IndexMap;

    use crate::profile::{ModerationProfile, ProfileConfig};

    use super::{
        evaluate_local_model, fallback_threshold, pick_model_for_attempt, save_cache,
        set_test_random_values, should_force_ai_review, SmartModerationResult, MODERATION_CACHE,
    };

    fn test_profile() -> ModerationProfile {
        ModerationProfile {
            profile_name: "test".to_string(),
            base_dir: PathBuf::from("/tmp/smart-test-profile"),
            config: ProfileConfig::default(),
        }
    }

    fn sample_result() -> SmartModerationResult {
        SmartModerationResult {
            violation: true,
            category: Some("test".to_string()),
            reason: Some("cached".to_string()),
            source: "ai".to_string(),
            confidence: Some(0.9),
            debug: None,
        }
    }

    fn clear_cache() {
        let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(IndexMap::new()));
        cache.lock().expect("moderation cache").clear();
    }

    #[test]
    fn ai_review_rate_uses_fresh_request_time_random_draws() {
        let mut profile = test_profile();
        profile.config.probability.ai_review_rate = 0.5;

        set_test_random_values(&[0.9, 0.1]);

        assert!(!should_force_ai_review(&profile));
        assert!(should_force_ai_review(&profile));
    }

    #[test]
    fn retry_model_selection_prefers_untried_models_before_repeats() {
        let models = vec![
            "model-a".to_string(),
            "model-b".to_string(),
            "model-c".to_string(),
        ];
        let mut attempted = Vec::new();

        set_test_random_values(&[0.60, 0.40, 0.95, 0.10]);

        let first = pick_model_for_attempt(&models, &attempted);
        attempted.push(first.clone());
        let second = pick_model_for_attempt(&models, &attempted);
        attempted.push(second.clone());
        let third = pick_model_for_attempt(&models, &attempted);
        attempted.push(third.clone());
        let fourth = pick_model_for_attempt(&models, &attempted);

        assert_eq!(first, "model-b");
        assert_eq!(second, "model-a");
        assert_eq!(third, "model-c");
        assert_eq!(fourth, "model-a");
    }

    #[test]
    fn moderation_cache_limits_profile_bucket_count() {
        clear_cache();

        for idx in 0..=50 {
            save_cache(&format!("profile-{idx}"), "same text", &sample_result());
        }

        let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(IndexMap::new()));
        let guard = cache.lock().expect("moderation cache");
        let profiles = guard.keys().cloned().collect::<Vec<_>>();
        assert_eq!(profiles.len(), 50);
        assert!(!profiles.iter().any(|name| name == "profile-0"));
        assert!(profiles.iter().any(|name| name == "profile-50"));

        drop(guard);
        clear_cache();
    }

    #[test]
    fn concurrency_limit_fallback_uses_probability_midpoint() {
        let mut profile = test_profile();
        profile.config.probability.low_risk_threshold = 0.2;
        profile.config.probability.high_risk_threshold = 0.8;

        let threshold = fallback_threshold(&profile);
        assert_eq!(threshold, 0.5);
    }

    #[test]
    fn evaluate_local_model_returns_skipped_debug_when_runtime_is_missing() {
        let profile = test_profile();
        let (result, debug) = evaluate_local_model("anything", &profile);

        assert!(result.is_none());
        assert_eq!(debug.local_model_decision.as_deref(), Some("skipped"));
    }
}
