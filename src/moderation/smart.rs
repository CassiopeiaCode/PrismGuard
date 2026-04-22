use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use reqwest::header::CONTENT_TYPE;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::time::timeout as tokio_timeout;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

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
static MODERATION_CACHE: OnceLock<Mutex<IndexMap<String, IndexMap<String, SmartModerationResult>>>> =
    OnceLock::new();
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
}

#[derive(Debug)]
pub enum SmartModerationError {
    ConcurrencyLimit(String),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for SmartModerationError {
    fn from(value: anyhow::Error) -> Self {
        Self::Other(value)
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
        return run_ai_moderation_with_history(text, profile, http_client, env_map).await;
    }

    if profile.local_model_exists() {
        if let Some(result) = run_local_model(text, profile) {
            return Ok(result);
        }
    }

    run_ai_moderation_with_history(text, profile, http_client, env_map).await
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

fn run_local_model(text: &str, profile: &ModerationProfile) -> Option<SmartModerationResult> {
    let (probability, source) = match profile.config.local_model_type.as_str() {
        "fasttext" => match fasttext::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "fasttext_model"),
            Ok(None) | Err(_) => return None,
        },
        "hashlinear" => match hashlinear::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "hashlinear_model"),
            Ok(None) | Err(_) => return None,
        },
        _ => match bow::predict_proba(text, profile) {
            Ok(Some(probability)) => (probability, "bow_model"),
            Ok(None) | Err(_) => return None,
        },
    };
    let low = profile.config.probability.low_risk_threshold;
    let high = profile.config.probability.high_risk_threshold;

    if probability < low {
        return Some(SmartModerationResult {
            violation: false,
            category: None,
            reason: Some(format!("{source}: low risk (p={probability:.3})")),
            source: source.to_string(),
            confidence: Some(probability),
        });
    }
    if probability > high {
        return Some(SmartModerationResult {
            violation: true,
            category: None,
            reason: Some(format!("{source}: high risk (p={probability:.3})")),
            source: source.to_string(),
            confidence: Some(probability),
        });
    }
    None
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
            .map_err(|err| SmartModerationError::Other(anyhow!(err)))?;

        let is_sse = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.starts_with("text/event-stream"))
            .unwrap_or(false);

        if is_sse {
            let content = read_openai_chat_sse_content(response)
                .await
                .map_err(|err| SmartModerationError::Other(err))?;
            parse_moderation_content(&content)
        } else {
            let payload = response
                .json::<Value>()
                .await
                .map_err(|err| SmartModerationError::Other(anyhow!(err)))?;
            parse_openai_moderation_response(payload)
        }
    }

    let mut attempted_models = Vec::new();
    let mut last_error = None;
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
                .map_err(|err| SmartModerationError::Other(anyhow!(err)))?;

            parse_llm_response(response).await
        })
        .await;

        match response {
            Ok(Ok(parsed)) => return Ok(parsed),
            Ok(Err(err)) => {
                let err = match err {
                    SmartModerationError::ConcurrencyLimit(message) => anyhow!(message),
                    SmartModerationError::Other(err) => err,
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

async fn read_openai_chat_sse_content(
    response: reqwest::Response,
) -> Result<String> {
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
) -> Result<SmartModerationResult, SmartModerationError> {
    if let Some(result) = load_history_result(text, profile).await? {
        return Ok(result);
    }
    let result = llm_moderate(text, profile, http_client, env_map).await?;
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
        }))
    })
    .await
    .map_err(|err| SmartModerationError::Other(anyhow!("history storage read task failed: {err}")))?
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
    .map_err(|err| SmartModerationError::Other(anyhow!("history storage write task failed: {err}")))?
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
                    part.get("text").and_then(Value::as_str).map(ToString::to_string)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn parse_moderation_content(
    content: &str,
) -> Result<SmartModerationResult, SmartModerationError> {
    let json_text = extract_json_object(content)
        .ok_or_else(|| anyhow!("llm moderation content is not a JSON object"))?;
    let data: Value = serde_json::from_str(json_text)
        .map_err(|err| anyhow!(err).context("failed to decode moderation JSON"))?;

    Ok(SmartModerationResult {
        violation: data.get("violation").and_then(Value::as_bool).unwrap_or(false),
        category: data.get("category").and_then(Value::as_str).map(ToString::to_string),
        reason: data.get("reason").and_then(Value::as_str).map(ToString::to_string),
        source: "ai".to_string(),
        confidence: data.get("confidence").and_then(Value::as_f64),
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
    let result = profile_cache.shift_remove(&key)?;
    profile_cache.insert(key, result.clone());
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
    profile_cache.insert(key, result.clone());
    while profile_cache.len() > CACHE_SIZE {
        profile_cache.shift_remove_index(0);
    }
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
        pick_model_for_attempt, save_cache, set_test_random_values, should_force_ai_review,
        SmartModerationResult, MODERATION_CACHE,
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
}
