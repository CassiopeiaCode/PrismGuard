use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{anyhow, Context, Result};
use jieba_rs::Jieba;
use serde::Deserialize;

use crate::profile::ModerationProfile;

static JIEBA: OnceLock<Jieba> = OnceLock::new();
static RUNTIME_CACHE: OnceLock<Mutex<HashMap<String, CachedRuntime>>> = OnceLock::new();

#[derive(Clone)]
struct CachedRuntime {
    signature: RuntimeSignature,
    runtime: Arc<BowRuntime>,
}

#[derive(Clone, PartialEq, Eq)]
struct RuntimeSignature {
    meta_mtime_secs: u64,
    meta_size: u64,
    coef_mtime_secs: u64,
    coef_size: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeMetadata {
    runtime_version: u32,
    intercept: f64,
    classes: Vec<i32>,
    vocabulary: Vec<String>,
    idf: Vec<f64>,
    tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    #[serde(default = "default_true")]
    lowercase: bool,
    #[serde(default = "default_true")]
    split_whitespace: bool,
    #[serde(default)]
    char_ngram_range: Vec<usize>,
    #[serde(default)]
    use_jieba: bool,
}

#[derive(Clone)]
struct BowRuntime {
    intercept: f64,
    vocabulary: HashMap<String, usize>,
    idf: Vec<f64>,
    coef: Vec<f32>,
    tokenizer: TokenizerConfig,
    reverse_classes: bool,
}

#[derive(Debug, Clone)]
pub struct BowRuntimePaths {
    pub prefix: PathBuf,
    pub metadata_path: PathBuf,
    pub coefficients_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct BowScore {
    pub probability: f64,
}

pub fn runtime_paths(profile: &ModerationProfile) -> BowRuntimePaths {
    let prefix = profile.base_dir.join("bow_runtime");
    BowRuntimePaths {
        metadata_path: prefix.with_extension("json"),
        coefficients_path: prefix.with_extension("coef.f32"),
        prefix,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn runtime_exists(profile: &ModerationProfile) -> bool {
    let paths = runtime_paths(profile);
    paths.metadata_path.exists() && paths.coefficients_path.exists()
}

pub fn predict(text: &str, profile: &ModerationProfile) -> Result<Option<BowScore>> {
    let paths = runtime_paths(profile);
    if !paths.metadata_path.exists() || !paths.coefficients_path.exists() {
        return Ok(None);
    }

    let runtime = load_runtime(profile, &paths.metadata_path, &paths.coefficients_path)?;
    Ok(Some(BowScore {
        probability: runtime.predict_proba(text),
    }))
}

pub fn predict_proba(text: &str, profile: &ModerationProfile) -> Result<Option<f64>> {
    Ok(predict(text, profile)?.map(|score| score.probability))
}

fn load_runtime(
    profile: &ModerationProfile,
    meta_path: &Path,
    coef_path: &Path,
) -> Result<Arc<BowRuntime>> {
    let signature = runtime_signature(meta_path, coef_path)?;
    let cache = RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = runtime_cache_key(profile);
    {
        let guard = cache.lock().expect("bow runtime cache");
        if let Some(cached) = guard.get(&cache_key) {
            if cached.signature == signature {
                return Ok(cached.runtime.clone());
            }
        }
    }

    let runtime = Arc::new(BowRuntime::load(meta_path, coef_path)?);
    let mut guard = cache.lock().expect("bow runtime cache");
    guard.insert(
        cache_key,
        CachedRuntime {
            signature,
            runtime: runtime.clone(),
        },
    );
    Ok(runtime)
}

fn runtime_cache_key(profile: &ModerationProfile) -> String {
    profile.base_dir.display().to_string()
}

fn runtime_signature(meta_path: &Path, coef_path: &Path) -> Result<RuntimeSignature> {
    Ok(RuntimeSignature {
        meta_mtime_secs: modified_secs(meta_path)?,
        meta_size: fs::metadata(meta_path)?.len(),
        coef_mtime_secs: modified_secs(coef_path)?,
        coef_size: fs::metadata(coef_path)?.len(),
    })
}

fn modified_secs(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path)?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|err| anyhow!("mtime before unix epoch for {}: {err}", path.display()))?
        .as_secs())
}

impl BowRuntime {
    fn load(meta_path: &Path, coef_path: &Path) -> Result<Self> {
        let metadata: RuntimeMetadata = serde_json::from_str(
            &fs::read_to_string(meta_path)
                .with_context(|| format!("failed to read {}", meta_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", meta_path.display()))?;
        validate_runtime_metadata(&metadata)?;

        let reverse_classes = match metadata.classes.as_slice() {
            [0, 1] => false,
            [1, 0] => true,
            other => {
                return Err(anyhow!("unsupported bow runtime classes: {:?}", other));
            }
        };

        let coef_bytes =
            fs::read(coef_path).with_context(|| format!("failed to read {}", coef_path.display()))?;
        if coef_bytes.len() % 4 != 0 {
            return Err(anyhow!(
                "invalid bow coef byte length: {}",
                coef_bytes.len()
            ));
        }
        let coef = coef_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if coef.len() != metadata.vocabulary.len() {
            return Err(anyhow!(
                "bow runtime coef length mismatch: {} != {}",
                coef.len(),
                metadata.vocabulary.len()
            ));
        }

        let mut vocabulary = HashMap::with_capacity(metadata.vocabulary.len());
        for (index, token) in metadata.vocabulary.into_iter().enumerate() {
            vocabulary.insert(token, index);
        }

        Ok(Self {
            intercept: metadata.intercept,
            vocabulary,
            idf: metadata.idf,
            coef,
            tokenizer: metadata.tokenizer,
            reverse_classes,
        })
    }

    fn predict_proba(&self, text: &str) -> f64 {
        let mut score = self.intercept;
        let token_counts = tokenize(text, &self.tokenizer);
        for (token, tf) in token_counts {
            if let Some(index) = self.vocabulary.get(&token) {
                let tfidf = tf * self.idf[*index];
                score += f64::from(self.coef[*index]) * tfidf;
            }
        }

        let probability = sigmoid(score);
        if self.reverse_classes {
            1.0 - probability
        } else {
            probability
        }
    }
}

fn validate_runtime_metadata(metadata: &RuntimeMetadata) -> Result<()> {
    if metadata.runtime_version != 1 {
        return Err(anyhow!(
            "unsupported bow runtime version: {}",
            metadata.runtime_version
        ));
    }
    if metadata.vocabulary.len() != metadata.idf.len() {
        return Err(anyhow!(
            "bow runtime vocabulary/idf length mismatch: {} != {}",
            metadata.vocabulary.len(),
            metadata.idf.len()
        ));
    }
    if metadata.vocabulary.is_empty() {
        return Err(anyhow!("bow runtime vocabulary must not be empty"));
    }
    if !metadata.tokenizer.char_ngram_range.is_empty() {
        let _ = ngram_range(&metadata.tokenizer.char_ngram_range)?;
    }
    Ok(())
}

fn tokenize(text: &str, cfg: &TokenizerConfig) -> HashMap<String, f64> {
    let normalized = if cfg.lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };

    let mut counts = HashMap::new();
    if cfg.use_jieba {
        for token in jieba()
            .cut(&normalized, true)
            .into_iter()
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            *counts.entry(token.to_string()).or_insert(0.0) += 1.0;
        }
    } else if cfg.split_whitespace {
        for token in normalized.split_whitespace().filter(|token| !token.is_empty()) {
            *counts.entry(token.to_string()).or_insert(0.0) += 1.0;
        }
    }

    if let Ok((min_n, max_n)) = ngram_range(&cfg.char_ngram_range) {
        let chars = normalized.chars().collect::<Vec<_>>();
        for n in min_n..=max_n {
            if n == 0 || chars.len() < n {
                continue;
            }
            for window in chars.windows(n) {
                let gram = window.iter().collect::<String>();
                *counts.entry(gram).or_insert(0.0) += 1.0;
            }
        }
    }

    counts
}

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

fn ngram_range(range: &[usize]) -> Result<(usize, usize)> {
    match range {
        [] => Err(anyhow!("empty ngram range")),
        [n] => Ok((*n, *n)),
        [start, end, ..] if *start > 0 && *end >= *start => Ok((*start, *end)),
        _ => Err(anyhow!("invalid bow char_ngram_range: {:?}", range)),
    }
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
