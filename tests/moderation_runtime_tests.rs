#[path = "../src/moderation/basic.rs"]
mod basic;
#[path = "../src/moderation/extract.rs"]
mod extract;

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

#[test]
fn extract_openai_chat_text_joins_string_and_text_parts() {
    let body = json!({
        "messages": [
            {"role": "system", "content": "system guardrails"},
            {"role": "user", "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "text", "text": "blocked phrase"}
            ]},
            {"role": "tool", "content": "tool output should be ignored"}
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "openai_chat");

    assert_eq!(text, "system guardrails\nhello\nblocked phrase");
}

#[test]
fn extract_claude_chat_text_includes_system_blocks_and_message_parts() {
    let body = json!({
        "system": [
            {"type": "text", "text": "system prelude"},
            {"type": "tool_use", "name": "lookup_weather"}
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "first question"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}
            ]},
            {"role": "assistant", "content": "assistant reply"}
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "claude_chat");

    assert_eq!(text, "system prelude\nfirst question\nassistant reply");
}

#[test]
fn extract_openai_responses_text_includes_instructions_and_structured_input() {
    let body = json!({
        "instructions": "system instructions",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_image", "image_url": "https://example.com/cat.png"},
                    {"type": "input_text", "text": "blocked phrase"}
                ]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "assistant context"}
                ]
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "tool result text"
            }
        ]
    });

    let text = extract::extract_text_for_moderation(&body, "openai_responses");

    assert_eq!(
        text,
        "system instructions\nhello\nblocked phrase\nassistant context\ntool result text"
    );
}
