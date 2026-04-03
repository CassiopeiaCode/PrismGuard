#[path = "../src/moderation/basic.rs"]
mod basic;

use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

fn temp_keywords_path(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prismguard-basic-moderation-{label}-{}-{unique}.txt",
        std::process::id()
    ))
}

#[test]
fn basic_moderation_returns_none_when_disabled() {
    let blocked = basic::basic_moderation("blocked phrase", &json!({ "enabled": false }))
        .expect("basic moderation result");

    assert!(blocked.is_none());
}

#[test]
fn basic_moderation_blocks_matching_keyword_case_insensitively() {
    let keyword_file = temp_keywords_path("match");
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let blocked = basic::basic_moderation(
        "This contains BLOCKED PHRASE in mixed case.",
        &json!({
            "enabled": true,
            "keywords_file": keyword_file,
            "error_code": "BASIC_MODERATION_BLOCKED"
        }),
    )
    .expect("basic moderation result");

    let blocked = blocked.expect("keyword should block");
    assert_eq!(
        blocked.reason,
        "[BASIC_MODERATION_BLOCKED] Matched keyword: blocked phrase"
    );

    let _ = std::fs::remove_file(keyword_file);
}

#[test]
fn basic_moderation_reloads_keywords_after_file_changes() {
    let keyword_file = temp_keywords_path("reload");
    std::fs::write(&keyword_file, "blocked phrase\n").expect("write keyword file");

    let cfg = json!({
        "enabled": true,
        "keywords_file": keyword_file,
        "error_code": "BASIC_MODERATION_BLOCKED"
    });

    let first = basic::basic_moderation("blocked phrase appears here", &cfg)
        .expect("first moderation result");
    assert!(first.is_some(), "initial keyword should block");

    std::thread::sleep(Duration::from_millis(5));
    std::fs::write(&keyword_file, "different phrase\n").expect("rewrite keyword file");

    let stale = basic::basic_moderation("blocked phrase appears here", &cfg)
        .expect("stale moderation result");
    assert!(
        stale.is_none(),
        "cache should refresh after keyword file content changes"
    );

    let refreshed = basic::basic_moderation("different phrase appears here", &cfg)
        .expect("refreshed moderation result");
    let refreshed = refreshed.expect("updated keyword should block");
    assert_eq!(
        refreshed.reason,
        "[BASIC_MODERATION_BLOCKED] Matched keyword: different phrase"
    );

    let _ = std::fs::remove_file(keyword_file);
}
