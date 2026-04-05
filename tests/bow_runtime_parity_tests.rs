#![allow(dead_code)]

#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/moderation/bow.rs"]
mod bow;

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use profile::ModerationProfile;
use serde_json::json;

fn temp_root(label: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "prismguard-bow-runtime-{label}-{}-{unique}",
        std::process::id()
    ))
}

fn write_profile(root: &PathBuf, profile_name: &str) -> ModerationProfile {
    let profile_dir = root.join("configs").join("mod_profiles").join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(
        profile_dir.join("profile.json"),
        json!({
            "local_model_type": "bow"
        })
        .to_string(),
    )
    .expect("write profile");
    ModerationProfile::load(root, profile_name).expect("load profile")
}

fn write_runtime(
    profile: &ModerationProfile,
    metadata: serde_json::Value,
    coef: &[f32],
) -> (PathBuf, PathBuf) {
    let metadata_path = profile.base_dir.join("bow_runtime.json");
    let coefficients_path = profile.base_dir.join("bow_runtime.coef.f32");
    std::fs::write(&metadata_path, metadata.to_string()).expect("write runtime metadata");

    let mut coef_bytes = Vec::with_capacity(std::mem::size_of_val(coef));
    for weight in coef {
        coef_bytes.extend_from_slice(&weight.to_le_bytes());
    }
    std::fs::write(&coefficients_path, coef_bytes).expect("write runtime coefficients");
    (metadata_path, coefficients_path)
}

#[test]
fn bow_runtime_uses_jieba_tokenizer_metadata_with_char_ngrams() {
    let root = temp_root("jieba");
    let profile = write_profile(&root, "default");
    write_runtime(
        &profile,
        json!({
            "runtime_version": 1,
            "intercept": 0.0,
            "classes": [0, 1],
            "tokenizer": {
                "lowercase": true,
                "split_whitespace": true,
                "use_jieba": true,
                "char_ngram_range": [2, 3]
            },
            "vocabulary": ["天安门广场", "广场欢"],
            "idf": [1.0, 1.0]
        }),
        &[2.0, 0.5],
    );

    let probability = bow::predict_proba("天安门广场欢迎你", &profile)
        .expect("predict")
        .expect("runtime score");

    assert!(
        (probability - 0.9241418199787566).abs() < 1e-9,
        "expected jieba tokens plus char ngrams to contribute, got {probability}"
    );
}

#[test]
fn bow_runtime_flips_probability_when_classes_are_reversed() {
    let root = temp_root("classes");
    let profile = write_profile(&root, "default");
    write_runtime(
        &profile,
        json!({
            "runtime_version": 1,
            "intercept": 0.0,
            "classes": [1, 0],
            "tokenizer": {
                "lowercase": true,
                "split_whitespace": true,
                "char_ngram_range": []
            },
            "vocabulary": ["blocked"],
            "idf": [1.0]
        }),
        &[2.0],
    );

    let probability = bow::predict_proba("blocked", &profile)
        .expect("predict")
        .expect("runtime score");

    assert!(
        (probability - 0.1192029220).abs() < 1e-9,
        "expected class-order handling to invert probability, got {probability}"
    );
}
