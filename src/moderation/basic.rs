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

const MINOR_MARKERS: &[&str] = &[
    "minor",
    "child",
    "children",
    "kid",
    "kids",
    "underage",
    "boy",
    "girl",
    "teen",
    "teenager",
    "小孩",
    "孩子",
    "儿童",
    "未成年",
    "男童",
    "女童",
    "男孩",
    "女孩",
    "少年",
    "少女",
    "初中生",
    "小学生",
    "正太",
    "萝莉",
];

const SEXUAL_MARKERS: &[&str] = &[
    "sex",
    "sexual",
    "explicit",
    "nude",
    "naked",
    "penis",
    "vagina",
    "genital",
    "cock",
    "dick",
    "cum",
    "blowjob",
    "handjob",
    "rape",
    "molest",
    "淫",
    "色情",
    "性",
    "性交",
    "强奸",
    "猥亵",
    "轮奸",
    "口交",
    "手交",
    "鸡鸡",
    "阴茎",
    "龟头",
    "生殖器",
    "射精",
];

const ABUSE_CONTROL_MARKERS: &[&str] = &[
    "time stop",
    "freeze time",
    "immobilize",
    "restrain",
    "force",
    "forced",
    "coerce",
    "abuse",
    "exploit",
    "玩弄",
    "时间停止",
    "定住",
    "强制",
    "强迫",
    "迷奸",
    "控制住",
    "侵犯",
];

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

    if let Some(block) = prefilter_moderation(text) {
        return Ok(Some(block));
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

pub fn prefilter_moderation(text: &str) -> Option<BasicModerationBlock> {
    let normalized = text.to_lowercase();

    let mentions_minor = contains_any(&normalized, MINOR_MARKERS) || contains_minor_age(&normalized);
    let mentions_sexual = contains_any(&normalized, SEXUAL_MARKERS);
    let mentions_abuse_or_control = contains_any(&normalized, ABUSE_CONTROL_MARKERS);

    if mentions_minor && mentions_sexual {
        return Some(BasicModerationBlock {
            reason: "[POLICY_MINOR_SEXUAL_CONTENT] Sexual content involving minors is disallowed"
                .to_string(),
        });
    }

    if mentions_minor && mentions_abuse_or_control {
        return Some(BasicModerationBlock {
            reason:
                "[POLICY_MINOR_EXPLOITATION] Exploitative or coercive content involving minors is disallowed"
                    .to_string(),
        });
    }

    None
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

fn contains_any(text: &str, markers: &[&str]) -> bool {
    markers.iter().any(|marker| text.contains(marker))
}

fn contains_minor_age(text: &str) -> bool {
    for age in 0..18 {
        let ascii = age.to_string();
        let full_width = to_full_width_digits(age);
        let patterns = [
            format!("{ascii}岁"),
            format!("{ascii} year old"),
            format!("{ascii}-year-old"),
            format!("age {ascii}"),
            format!("{full_width}岁"),
        ];
        if patterns.iter().any(|pattern| text.contains(pattern)) {
            return true;
        }
    }
    false
}

fn to_full_width_digits(age: u8) -> String {
    age.to_string()
        .chars()
        .map(|ch| match ch {
            '0' => '０',
            '1' => '１',
            '2' => '２',
            '3' => '３',
            '4' => '４',
            '5' => '５',
            '6' => '６',
            '7' => '７',
            '8' => '８',
            '9' => '９',
            _ => ch,
        })
        .collect()
}
