use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use indexmap::IndexMap;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::moderation::hashlinear;
use crate::profile::ModerationProfile;

const MAX_LLM_CONCURRENCY: usize = 5;
const CACHE_SIZE: usize = 1024;

static LLM_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();
static MODERATION_CACHE: OnceLock<Mutex<HashMap<String, IndexMap<String, SmartModerationResult>>>> =
    OnceLock::new();

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
    if should_force_ai_review(text, profile) {
        return llm_moderate(text, profile, http_client, env_map).await;
    }

    if let Some(result) = run_local_model(text, profile) {
        return Ok(result);
    }

    llm_moderate(text, profile, http_client, env_map).await
}

fn should_force_ai_review(text: &str, profile: &ModerationProfile) -> bool {
    let rate = profile.config.probability.ai_review_rate;
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }

    let seed = profile.config.probability.random_seed.to_le_bytes();
    let digest = md5::compute([seed.as_slice(), text.as_bytes()].concat());
    let value = u64::from_le_bytes([
        digest.0[0],
        digest.0[1],
        digest.0[2],
        digest.0[3],
        digest.0[4],
        digest.0[5],
        digest.0[6],
        digest.0[7],
    ]);
    let unit = (value as f64) / (u64::MAX as f64);
    unit < rate
}

fn run_local_model(text: &str, profile: &ModerationProfile) -> Option<SmartModerationResult> {
    if profile.config.local_model_type != "hashlinear" {
        return None;
    }

    let probability = match hashlinear::predict_proba(text, profile) {
        Ok(Some(probability)) => probability,
        Ok(None) | Err(_) => return None,
    };
    let low = profile.config.probability.low_risk_threshold;
    let high = profile.config.probability.high_risk_threshold;

    if probability < low {
        return Some(SmartModerationResult {
            violation: false,
            category: None,
            reason: Some(format!("hashlinear: low risk (p={probability:.3})")),
            source: "hashlinear_model".to_string(),
            confidence: Some(probability),
        });
    }
    if probability > high {
        return Some(SmartModerationResult {
            violation: true,
            category: None,
            reason: Some(format!("hashlinear: high risk (p={probability:.3})")),
            source: "hashlinear_model".to_string(),
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

    let mut last_error = None;
    for attempt in 0..=max_retries {
        let model = &models[attempt % models.len()];
        let response = http_client
            .post(&endpoint)
            .bearer_auth(&api_key)
            .timeout(timeout)
            .json(&json!({
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }],
                "temperature": 0
            }))
            .send()
            .await;

        match response {
            Ok(resp) => {
                match resp.error_for_status() {
                    Ok(resp) => match resp.json::<Value>().await {
                        Ok(payload) => return parse_openai_moderation_response(payload),
                        Err(err) => {
                            last_error = Some(anyhow!(err).context("failed to decode llm moderation response"));
                        }
                    },
                    Err(err) => {
                        last_error = Some(anyhow!(err).context("llm moderation request failed"));
                    }
                }
            }
            Err(err) => last_error = Some(anyhow!(err)),
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

fn parse_openai_moderation_response(payload: Value) -> Result<SmartModerationResult, SmartModerationError> {
    let content = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .map(extract_content_text)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("llm moderation response missing choices[0].message.content"))?;

    Ok(parse_moderation_content(&content))
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

fn parse_moderation_content(content: &str) -> SmartModerationResult {
    let parsed = extract_json_object(content)
        .and_then(|json_text| serde_json::from_str::<Value>(json_text).ok());

    if let Some(data) = parsed {
        return SmartModerationResult {
            violation: data.get("violation").and_then(Value::as_bool).unwrap_or(false),
            category: data.get("category").and_then(Value::as_str).map(ToString::to_string),
            reason: data.get("reason").and_then(Value::as_str).map(ToString::to_string),
            source: "ai".to_string(),
            confidence: data.get("confidence").and_then(Value::as_f64),
        };
    }

    let lowered = content.to_ascii_lowercase();
    SmartModerationResult {
        violation: lowered.contains("违规")
            || lowered.contains("violation")
            || lowered.contains("不当"),
        category: Some("unknown".to_string()),
        reason: Some(content.chars().take(200).collect()),
        source: "ai".to_string(),
        confidence: None,
    }
}

fn extract_json_object(content: &str) -> Option<&str> {
    let start = content.find('{')?;
    let end = content.rfind('}')?;
    (end > start).then_some(&content[start..=end])
}

fn check_cache(profile_name: &str, text: &str) -> Option<SmartModerationResult> {
    let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = cache_key(text);
    let mut guard = cache.lock().expect("moderation cache");
    let profile_cache = guard.get_mut(profile_name)?;
    let result = profile_cache.shift_remove(&key)?;
    profile_cache.insert(key, result.clone());
    Some(result)
}

fn save_cache(profile_name: &str, text: &str, result: &SmartModerationResult) {
    let cache = MODERATION_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = cache_key(text);
    let mut guard = cache.lock().expect("moderation cache");
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
