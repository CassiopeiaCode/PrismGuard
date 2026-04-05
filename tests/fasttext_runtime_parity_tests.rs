#![allow(dead_code)]

#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/moderation/fasttext.rs"]
mod fasttext;

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use profile::ModerationProfile;
use serde_json::json;

fn unique_test_root(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prismguard-fasttext-runtime-{label}-{}-{unique}",
        std::process::id()
    ))
}

fn write_profile_into(root_dir: &Path, profile_name: &str) -> ModerationProfile {
    let profile_dir = root_dir
        .join("configs")
        .join("mod_profiles")
        .join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "local_model_type": "fasttext"
        })
        .to_string(),
    )
    .expect("write profile");
    ModerationProfile::load(root_dir, profile_name).expect("load profile")
}

#[test]
fn fasttext_runtime_honors_jieba_tokenizer_metadata_for_unsegmented_chinese_text() {
    let root_dir = unique_test_root("jieba");
    let profile = write_profile_into(&root_dir, "jieba-profile");

    std::fs::write(
        profile.fasttext_runtime_json_path(),
        json!({
            "runtime_version": 1,
            "intercept": -2.0,
            "classes": [0, 1],
            "tokenizer": {
                "lowercase": true,
                "split_whitespace": true,
                "use_jieba": true
            },
            "weights": [
                {"token": "北京", "weight": 4.0}
            ]
        })
        .to_string(),
    )
    .expect("write runtime metadata");

    let probability = fasttext::predict_proba("我喜欢北京", &profile)
        .expect("predict probability result")
        .expect("runtime prediction");

    assert!(
        probability > 0.5,
        "expected jieba tokenization to surface 北京 token, got probability {probability}"
    );
}

#[test]
fn fasttext_runtime_rejects_unsupported_tiktoken_metadata() {
    let root_dir = unique_test_root("tiktoken");
    let profile = write_profile_into(&root_dir, "tiktoken-profile");

    std::fs::write(
        profile.fasttext_runtime_json_path(),
        json!({
            "runtime_version": 1,
            "intercept": -2.0,
            "classes": [0, 1],
            "tokenizer": {
                "lowercase": true,
                "split_whitespace": true,
                "use_tiktoken": true,
                "tiktoken_model": "cl100k_base"
            },
            "weights": []
        })
        .to_string(),
    )
    .expect("write runtime metadata");

    let err = fasttext::predict_proba("北京", &profile).expect_err("tiktoken metadata should fail");
    assert!(
        err.to_string().contains("use_tiktoken"),
        "expected unsupported tiktoken metadata error, got {err:#}"
    );
}
