use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
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
    runtime: Arc<HashlinearRuntime>,
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
    n_features: usize,
    intercept: f64,
    classes: Vec<i32>,
    cfg: RuntimeConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct RuntimeConfig {
    #[serde(default = "default_analyzer")]
    analyzer: String,
    #[serde(default = "default_ngram_range")]
    ngram_range: Vec<usize>,
    n_features: usize,
    #[serde(default)]
    alternate_sign: bool,
    #[serde(default)]
    norm: Option<String>,
    #[serde(default = "default_true")]
    lowercase: bool,
    #[serde(default)]
    use_jieba: bool,
}

#[derive(Clone)]
struct HashlinearRuntime {
    intercept: f64,
    n_features: usize,
    coef: Vec<f32>,
    cfg: RuntimeConfig,
}

#[derive(Debug, Clone)]
pub struct HashlinearRuntimePaths {
    pub prefix: PathBuf,
    pub metadata_path: PathBuf,
    pub coefficients_path: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct HashlinearScore {
    pub probability: f64,
}

pub fn runtime_paths(profile: &ModerationProfile) -> HashlinearRuntimePaths {
    let prefix = profile.base_dir.join("hashlinear_runtime");
    HashlinearRuntimePaths {
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

pub fn predict(text: &str, profile: &ModerationProfile) -> Result<Option<HashlinearScore>> {
    let paths = runtime_paths(profile);
    if !paths.metadata_path.exists() || !paths.coefficients_path.exists() {
        return Ok(None);
    }

    let runtime = load_runtime(profile, &paths.metadata_path, &paths.coefficients_path)?;
    let clean = text.replace('\r', " ").replace('\n', " ");
    let prepared = maybe_tokenize_for_word_analyzer(&clean, &runtime.cfg);
    Ok(Some(HashlinearScore {
        probability: runtime.predict_proba(&prepared)?,
    }))
}

pub fn predict_proba(text: &str, profile: &ModerationProfile) -> Result<Option<f64>> {
    Ok(predict(text, profile)?.map(|score| score.probability))
}

fn load_runtime(
    profile: &ModerationProfile,
    meta_path: &Path,
    coef_path: &Path,
) -> Result<Arc<HashlinearRuntime>> {
    let signature = runtime_signature(meta_path, coef_path)?;
    let cache = RUNTIME_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let cache_key = runtime_cache_key(profile);
    {
        let guard = cache.lock().expect("hashlinear runtime cache");
        if let Some(cached) = guard.get(&cache_key) {
            if cached.signature == signature {
                return Ok(cached.runtime.clone());
            }
        }
    }

    let runtime = Arc::new(HashlinearRuntime::load(meta_path, coef_path)?);
    let mut guard = cache.lock().expect("hashlinear runtime cache");
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

impl HashlinearRuntime {
    fn load(meta_path: &Path, coef_path: &Path) -> Result<Self> {
        let metadata: RuntimeMetadata = serde_json::from_str(
            &fs::read_to_string(meta_path)
                .with_context(|| format!("failed to read {}", meta_path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", meta_path.display()))?;

        if metadata.runtime_version != 1 {
            return Err(anyhow!(
                "unsupported hashlinear runtime version: {}",
                metadata.runtime_version
            ));
        }
        if metadata.classes != [0, 1] {
            return Err(anyhow!(
                "unsupported hashlinear runtime classes: {:?}",
                metadata.classes
            ));
        }
        if metadata.cfg.n_features != metadata.n_features {
            return Err(anyhow!(
                "hashlinear runtime n_features mismatch: {} != {}",
                metadata.cfg.n_features,
                metadata.n_features
            ));
        }
        validate_runtime_metadata(&metadata)?;

        let coef_bytes =
            fs::read(coef_path).with_context(|| format!("failed to read {}", coef_path.display()))?;
        if coef_bytes.len() % 4 != 0 {
            return Err(anyhow!(
                "invalid hashlinear coef byte length: {}",
                coef_bytes.len()
            ));
        }
        let coef = coef_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if coef.len() != metadata.n_features {
            return Err(anyhow!(
                "hashlinear runtime coef length mismatch: {} != {}",
                coef.len(),
                metadata.n_features
            ));
        }

        Ok(Self {
            intercept: metadata.intercept,
            n_features: metadata.n_features,
            coef,
            cfg: metadata.cfg,
        })
    }

    fn predict_proba(&self, text: &str) -> Result<f64> {
        let features = extract_features(text, &self.cfg, self.n_features)?;
        let mut score = self.intercept;
        for (idx, value) in features {
            score += f64::from(self.coef[idx]) * value;
        }
        Ok(sigmoid(score))
    }
}

fn extract_features(
    text: &str,
    cfg: &RuntimeConfig,
    n_features: usize,
) -> Result<HashMap<usize, f64>> {
    if n_features == 0 {
        return Err(anyhow!("hashlinear runtime n_features must be > 0"));
    }

    let lowered = if cfg.lowercase {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let cleaned = lowered.replace('\r', " ").replace('\n', " ");
    let grams = match cfg.analyzer.as_str() {
        "char" => iter_char_ngrams(&normalize_spaces(&cleaned), ngram_range(&cfg.ngram_range)?),
        "word" => iter_word_ngrams(&cleaned, ngram_range(&cfg.ngram_range)?),
        other => return Err(anyhow!("unsupported hashlinear analyzer: {other}")),
    };

    let mut counts = HashMap::new();
    for gram in grams {
        let hash = murmurhash3_x86_32(gram.as_bytes(), 0);
        let idx = hash.unsigned_abs() as usize % n_features;
        let sign = if cfg.alternate_sign && hash < 0 { -1.0 } else { 1.0 };
        *counts.entry(idx).or_insert(0.0) += sign;
    }

    if cfg.norm.as_deref() == Some("l2") && !counts.is_empty() {
        let norm = counts.values().map(|value| value * value).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in counts.values_mut() {
                *value /= norm;
            }
        }
    }

    Ok(counts)
}

fn ngram_range(range: &[usize]) -> Result<(usize, usize)> {
    match range {
        [min_n, max_n] if *min_n > 0 && *min_n <= *max_n => Ok((*min_n, *max_n)),
        other => Err(anyhow!("invalid hashlinear ngram_range: {:?}", other)),
    }
}

fn validate_runtime_metadata(metadata: &RuntimeMetadata) -> Result<()> {
    if metadata.n_features == 0 {
        return Err(anyhow!("hashlinear runtime n_features must be > 0"));
    }
    Ok(())
}

fn maybe_tokenize_for_word_analyzer(text: &str, cfg: &RuntimeConfig) -> String {
    if cfg.analyzer != "word" || !cfg.use_jieba {
        return text.to_string();
    }

    JIEBA.get_or_init(Jieba::new).cut(text, false).join(" ")
}

fn normalize_spaces(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut previous_was_space = false;
    for ch in input.chars() {
        let is_space = ch.is_whitespace();
        if is_space {
            if !previous_was_space {
                out.push(' ');
            }
        } else {
            out.push(ch);
        }
        previous_was_space = is_space;
    }
    out
}

fn iter_char_ngrams(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut grams = Vec::new();
    let text_len = chars.len();

    let mut current_min = min_n;
    if current_min == 1 {
        grams.extend(chars.iter().map(|ch| ch.to_string()));
        current_min += 1;
    }

    for n in current_min..=max_n.min(text_len) {
        for start in 0..=text_len.saturating_sub(n) {
            grams.push(chars[start..start + n].iter().collect());
        }
    }
    grams
}

fn iter_word_ngrams(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    let mut grams = Vec::new();
    let mut current_min = min_n;
    if current_min == 1 {
        grams.extend(tokens.iter().map(|token| (*token).to_string()));
        current_min += 1;
    }
    for n in current_min..=max_n.min(tokens.len()) {
        for start in 0..=tokens.len().saturating_sub(n) {
            grams.push(tokens[start..start + n].join(" "));
        }
    }
    grams
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn murmurhash3_x86_32(data: &[u8], seed: u32) -> i32 {
    const C1: u32 = 0xCC9E2D51;
    const C2: u32 = 0x1B87_3593;

    let len = data.len();
    let mut h1 = seed;
    let rounded_end = len & !0x3;

    for chunk in data[..rounded_end].chunks_exact(4) {
        let mut k1 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xE654_6B64);
    }

    let tail = &data[rounded_end..];
    let mut k1 = 0u32;
    match tail.len() {
        3 => {
            k1 ^= u32::from(tail[2]) << 16;
            k1 ^= u32::from(tail[1]) << 8;
            k1 ^= u32::from(tail[0]);
        }
        2 => {
            k1 ^= u32::from(tail[1]) << 8;
            k1 ^= u32::from(tail[0]);
        }
        1 => {
            k1 ^= u32::from(tail[0]);
        }
        _ => {}
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85EB_CA6B);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xC2B2_AE35);
    h1 ^= h1 >> 16;
    h1 as i32
}

fn default_analyzer() -> String {
    "char".to_string()
}

fn default_ngram_range() -> Vec<usize> {
    vec![2, 4]
}

fn default_true() -> bool {
    true
}
