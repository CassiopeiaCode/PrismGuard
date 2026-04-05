#![allow(dead_code)]

#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/moderation/hashlinear.rs"]
mod hashlinear;

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
        "prismguard-hashlinear-runtime-{label}-{}-{unique}",
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
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "analyzer": "word",
                "ngram_range": [2, 2],
                "n_features": 32,
                "use_jieba": true
            }
        })
        .to_string(),
    )
    .expect("write profile");
    ModerationProfile::load(root_dir, profile_name).expect("load profile")
}

fn write_runtime(profile: &ModerationProfile, cfg: serde_json::Value, coef: &[f32]) {
    std::fs::write(
        profile.hashlinear_runtime_json_path(),
        json!({
            "runtime_version": 1,
            "n_features": coef.len(),
            "intercept": -5.0,
            "classes": [0, 1],
            "cfg": cfg
        })
        .to_string(),
    )
    .expect("write runtime json");

    let mut bytes = Vec::with_capacity(std::mem::size_of_val(coef));
    for weight in coef {
        bytes.extend_from_slice(&weight.to_le_bytes());
    }
    std::fs::write(profile.hashlinear_runtime_coef_path(), bytes).expect("write runtime coef");
}

#[test]
fn hashlinear_runtime_supports_word_analyzer_with_jieba() {
    let root_dir = unique_test_root("jieba");
    let profile = write_profile_into(&root_dir, "default");
    write_runtime(
        &profile,
        json!({
            "analyzer": "word",
            "ngram_range": [2, 2],
            "n_features": 32,
            "alternate_sign": false,
            "norm": "l2",
            "lowercase": true,
            "use_jieba": true
        }),
        &vec![10.0; 32],
    );

    let probability = hashlinear::predict_proba("天气不错", &profile)
        .expect("predict")
        .expect("runtime probability");

    assert!(
        probability > 0.8,
        "expected jieba word-bigram runtime to score high, got {probability}"
    );
}
