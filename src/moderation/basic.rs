use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct BasicModerationBlock {
    pub reason: String,
}

#[derive(Debug, Clone)]
struct KeywordFilter {
    mtime_secs: u64,
    patterns: Vec<String>,
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
        if normalized.contains(&kw.to_ascii_lowercase()) {
            return Ok(Some(BasicModerationBlock {
                reason: format!("[{}] Matched keyword: {}", error_code, kw),
            }));
        }
    }

    Ok(None)
}

fn load_keywords(path: &str) -> Result<Vec<String>> {
    let cache = FILTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let meta = fs::metadata(path).ok();
    let mtime_secs = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut guard = cache.lock().expect("keyword filter cache");
    if let Some(entry) = guard.get(path) {
        if entry.mtime_secs == mtime_secs {
            return Ok(entry.patterns.clone());
        }
    }

    let patterns = if Path::new(path).exists() {
        fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    guard.insert(
        path.to_string(),
        KeywordFilter {
            mtime_secs,
            patterns: patterns.clone(),
        },
    );

    Ok(patterns)
}
