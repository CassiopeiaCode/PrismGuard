#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/config.rs"]
mod config;

use profile::ModerationProfile;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

fn env_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn write_profile(profile_name: &str, payload: serde_json::Value) -> ModerationProfile {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let profile_dir = root_dir
        .join("configs")
        .join("mod_profiles")
        .join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(profile_dir.join("profile.json"), payload.to_string()).expect("write profile");
    ModerationProfile::load(&root_dir, profile_name).expect("load profile")
}

#[test]
fn settings_default_scheduler_uses_single_worker_safe_values() {
    let _guard = env_test_lock().lock().expect("env test lock");
    let root_dir = PathBuf::from("/tmp/prismguard-scheduler-defaults");
    std::env::remove_var("TRAINING_SCHEDULER_ENABLED");
    std::env::remove_var("TRAINING_SCHEDULER_INTERVAL_MINUTES");
    std::env::remove_var("TRAINING_SCHEDULER_FAILURE_COOLDOWN_MINUTES");
    std::env::remove_var("TRAINING_SUBPROCESS_ALLOWED_CPUS");

    let settings = config::Settings::load(&root_dir).expect("load settings");

    assert!(settings.training_scheduler_enabled);
    assert_eq!(settings.training_scheduler_interval_minutes, 10);
    assert_eq!(settings.training_scheduler_failure_cooldown_minutes, 30);
    assert_eq!(settings.training_subprocess_allowed_cpus, "0");
}

#[test]
fn profile_training_model_path_matches_local_model_type() {
    let profile = write_profile(
        &format!("scheduler-profile-model-path-{}", std::process::id()),
        json!({"local_model_type": "hashlinear"}),
    );

    assert_eq!(profile.training_model_path(), profile.hashlinear_model_path());
}

#[test]
fn profile_training_max_db_items_matches_local_model_type() {
    let profile = write_profile(
        &format!("scheduler-profile-max-db-items-{}", std::process::id()),
        json!({
            "local_model_type": "fasttext",
            "fasttext_training": {
                "max_db_items": 321
            }
        }),
    );

    assert_eq!(profile.training_max_db_items().expect("max db items"), 321);
}
