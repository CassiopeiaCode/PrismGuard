use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use jieba_rs::Jieba;
use serde::Deserialize;

use crate::profile::ModerationProfile;

static RUNTIME_CACHE: OnceLock<Mutex<HashMap<String, CachedRuntime>>> = OnceLock::new();
static JIEBA: OnceLock<Jieba> = OnceLock::new();

#[derive(Clone)]
struct CachedRuntime {
    signature: RuntimeSignature,
    runtime: Arc<FasttextRuntime>,
}

#[derive(Clone, PartialEq, Eq)]
struct RuntimeSignature {
    mtime_secs: u64,
    size: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeMetadata {
    runtime_version: u32,
    intercept: f64,
    classes: Vec<i32>,
    tokenizer: TokenizerConfig,
    weights: Vec<TokenWeight>,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    #[serde(default = "default_true")]
    lowercase: bool,
    #[serde(default = "default_true")]
    split_whitespace: bool,
    #[serde(default)]
    use_jieba: bool,
    #[serde(default)]
    use_tiktoken: bool,
    #[serde(default = "default_tiktoken_model")]
    tiktoken_model: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenWeight {
    token: String,
    weight: f64,
}

#[derive(Clone)]
struct FasttextRuntime {
    intercept: f64,
    tokenizer: TokenizerConfig,
    weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct FasttextScore {
    pub probability: f64,
}

pub fn predict(text: &str, profile: &ModerationProfile) -> Result<Option<FasttextScore>> {
    let path = profile.fasttext_runtime_json_path();
    if !path.exists() {
        return Ok(None);
    }

    let runtime = load_runtime(profile, &path)?;
    Ok(Some(FasttextScore {
        probability: runtime.predict_proba(text),
    }))
}

pub fn predict_proba(text: &str, profile: &ModerationProfile) -> Result<Option<f64>> {
    Ok(predict(text, profile)?.map(|score| score.probability))
}

fn load_runtime(profile: &ModerationProfile, path: &Path) -> Result<Arc<FasttextRuntime>> {
    let signature = runtime_signature(path)?;
    let cache = RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = profile.base_dir.display().to_string();
    {
        let guard = cache.lock().expect("fasttext runtime cache");
        if let Some(cached) = guard.get(&cache_key) {
            if cached.signature == signature {
                return Ok(cached.runtime.clone());
            }
        }
    }

    let runtime = Arc::new(FasttextRuntime::load(path)?);
    let mut guard = cache.lock().expect("fasttext runtime cache");
    guard.insert(
        cache_key,
        CachedRuntime {
            signature,
            runtime: runtime.clone(),
        },
    );
    Ok(runtime)
}

fn runtime_signature(path: &Path) -> Result<RuntimeSignature> {
    Ok(RuntimeSignature {
        mtime_secs: modified_secs(path)?,
        size: fs::metadata(path)?.len(),
    })
}

fn modified_secs(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| anyhow!("mtime before unix epoch for {}: {err}", path.display()))?
        .as_secs())
}

impl FasttextRuntime {
    fn load(path: &Path) -> Result<Self> {
        let metadata: RuntimeMetadata = serde_json::from_str(
            &fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;

        if metadata.runtime_version != 1 {
            return Err(anyhow!(
                "unsupported fasttext runtime version: {}",
                metadata.runtime_version
            ));
        }
        if metadata.classes != [0, 1] {
            return Err(anyhow!(
                "unsupported fasttext runtime classes: {:?}",
                metadata.classes
            ));
        }
        validate_tokenizer(&metadata.tokenizer)?;

        let weights = metadata
            .weights
            .into_iter()
            .map(|item| (item.token, item.weight))
            .collect::<HashMap<_, _>>();

        Ok(Self {
            intercept: metadata.intercept,
            tokenizer: metadata.tokenizer,
            weights,
        })
    }

    fn predict_proba(&self, text: &str) -> f64 {
        let mut score = self.intercept;
        for token in tokenize(text, &self.tokenizer) {
            if let Some(weight) = self.weights.get(&token) {
                score += *weight;
            }
        }
        sigmoid(score)
    }
}

fn validate_tokenizer(cfg: &TokenizerConfig) -> Result<()> {
    if cfg.use_tiktoken {
        return Err(anyhow!(
            "unsupported fasttext tokenizer metadata: use_tiktoken=true (model={})",
            cfg.tiktoken_model
        ));
    }
    Ok(())
}

fn tokenize(text: &str, cfg: &TokenizerConfig) -> Vec<String> {
    let normalized = normalize_text(text, cfg.lowercase);
    if cfg.use_jieba {
        return jieba_tokenize(&normalized);
    }
    if cfg.split_whitespace {
        return normalized
            .split_whitespace()
            .filter(|token| !token.is_empty())
            .map(ToString::to_string)
            .collect();
    }
    vec![normalized]
}

fn normalize_text(text: &str, lowercase: bool) -> String {
    let cleaned = text.replace('\n', " ").replace('\r', " ");
    if lowercase {
        cleaned.to_lowercase()
    } else {
        cleaned
    }
}

fn jieba_tokenize(text: &str) -> Vec<String> {
    jieba()
        .cut(text, false)
        .into_iter()
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let exp = (-x).exp();
        1.0 / (1.0 + exp)
    } else {
        let exp = x.exp();
        exp / (1.0 + exp)
    }
}

fn default_true() -> bool {
    true
}

fn default_tiktoken_model() -> String {
    "cl100k_base".to_string()
}
