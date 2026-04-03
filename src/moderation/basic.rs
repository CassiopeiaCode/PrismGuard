use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::Result;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct BasicModerationBlock {
    pub reason: String,
}

#[derive(Debug, Clone)]
struct KeywordPattern {
    original: String,
    normalized: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KeywordFileSignature {
    modified_at: Option<SystemTime>,
    len: Option<u64>,
}

#[derive(Debug, Clone)]
struct KeywordFilter {
    signature: KeywordFileSignature,
    patterns: Vec<KeywordPattern>,
}

static FILTERS: OnceLock<Mutex<HashMap<String, KeywordFilter>>> = OnceLock::new();

pub fn basic_moderation(text: &str, cfg: &Value) -> Result<Option<BasicModerationBlock>> {
    if !cfg.get("enabled").and_then(Value::as_bool).unwrap_or(false) {
        return Ok(None);
    }

    let keywords_file = cfg
        .get("keywords_file")
        .and_then(Value::as_str)
        .unwrap_or("configs/keywords.txt");
    let patterns = load_keywords(keywords_file)?;
    let error_code = cfg
        .get("error_code")
        .and_then(Value::as_str)
        .unwrap_or("BASIC_MODERATION_BLOCKED");
    let normalized = text.to_ascii_lowercase();

    for kw in patterns {
        if normalized.contains(&kw.normalized) {
            return Ok(Some(BasicModerationBlock {
                reason: format!("[{}] Matched keyword: {}", error_code, kw.original),
            }));
        }
    }

    Ok(None)
}

fn load_keywords(path: &str) -> Result<Vec<KeywordPattern>> {
    let cache = FILTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let signature = keyword_file_signature(Path::new(path));

    let mut guard = cache.lock().expect("keyword filter cache");
    if let Some(entry) = guard.get(path) {
        if entry.signature == signature {
            return Ok(entry.patterns.clone());
        }
    }

    let patterns = if Path::new(path).exists() {
        fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| KeywordPattern {
                original: line.to_string(),
                normalized: line.to_ascii_lowercase(),
            })
            .collect()
    } else {
        Vec::new()
    };

    guard.insert(
        path.to_string(),
        KeywordFilter {
            signature,
            patterns: patterns.clone(),
        },
    );

    Ok(patterns)
}

fn keyword_file_signature(path: &Path) -> KeywordFileSignature {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));

    match fs::metadata(&canonical) {
        Ok(metadata) => KeywordFileSignature {
            modified_at: metadata.modified().ok(),
            len: Some(metadata.len()),
        },
        Err(_) => KeywordFileSignature {
            modified_at: None,
            len: None,
        },
    }
}
