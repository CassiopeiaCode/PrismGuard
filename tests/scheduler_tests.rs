#![allow(dead_code)]

#[path = "../src/config.rs"]
mod config;
#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/scheduler.rs"]
mod scheduler;
#[path = "../src/sample_rpc.rs"]
mod sample_rpc;
#[cfg(feature = "storage-debug")]
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/training.rs"]
mod training;

use config::Settings;
use profile::ModerationProfile;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sample_rpc::{
    serve_unix_requests_until_shutdown, SampleRpcRequest, SampleRpcResponse,
};

fn env_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn write_profile(profile_name: &str, payload: serde_json::Value) -> ModerationProfile {
    let root_dir = repo_root();
    let profile_dir = root_dir
        .join("configs")
        .join("mod_profiles")
        .join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(profile_dir.join("profile.json"), payload.to_string()).expect("write profile");
    ModerationProfile::load(&root_dir, profile_name).expect("load profile")
}

fn write_training_status(profile: &ModerationProfile, payload: serde_json::Value) {
    std::fs::write(profile.training_status_path(), payload.to_string()).expect("write training status");
}

fn current_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("current unix time")
        .as_secs()
}

fn test_settings(root_dir: &std::path::Path, socket_path: &std::path::Path) -> Settings {
    Settings {
        host: "127.0.0.1".to_string(),
        port: 0,
        debug: true,
        log_level: "info".to_string(),
        access_log_file: "logs/access.log".to_string(),
        moderation_log_file: "logs/moderation.log".to_string(),
        training_log_file: "logs/training.log".to_string(),
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: socket_path.display().to_string(),
        training_scheduler_enabled: true,
        training_scheduler_interval_minutes: 10,
        training_scheduler_failure_cooldown_minutes: 30,
        training_subprocess_allowed_cpus: "0".to_string(),
        root_dir: root_dir.to_path_buf(),
        env_map: Default::default(),
    }
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

#[test]
fn scheduler_lists_only_profiles_with_profile_json() {
    let profile_name = format!("scheduler-list-valid-{}", std::process::id());
    let ignored_name = format!("scheduler-list-ignored-{}", std::process::id());
    let root_dir = repo_root();
    let valid_dir = root_dir.join("configs").join("mod_profiles").join(&profile_name);
    let ignored_dir = root_dir.join("configs").join("mod_profiles").join(&ignored_name);

    std::fs::create_dir_all(&valid_dir).expect("create valid dir");
    std::fs::write(
        valid_dir.join("profile.json"),
        json!({"local_model_type": "hashlinear"}).to_string(),
    )
    .expect("write valid profile");
    std::fs::create_dir_all(&ignored_dir).expect("create ignored dir");

    let profiles = scheduler::list_profiles(&root_dir).expect("list profiles");

    assert!(profiles.contains(&profile_name));
    assert!(!profiles.contains(&ignored_name));
}

#[test]
fn scheduler_skips_profile_when_failure_cooldown_not_elapsed() {
    let profile = write_profile(
        &format!("scheduler-cooldown-active-{}", std::process::id()),
        json!({"local_model_type": "hashlinear"}),
    );
    let now = current_unix_secs();
    write_training_status(
        &profile,
        json!({
            "status": "failed",
            "message": "boom",
            "timestamp": now,
            "profile": profile.profile_name,
            "model_type": "hashlinear"
        }),
    );

    let decision = scheduler::cooldown_allows_training(&profile, 30, now).expect("cooldown decision");

    assert!(!decision);
}

#[test]
fn scheduler_allows_profile_after_failure_cooldown_elapses() {
    let profile = write_profile(
        &format!("scheduler-cooldown-expired-{}", std::process::id()),
        json!({"local_model_type": "hashlinear"}),
    );
    let now = current_unix_secs();
    let stale = now - Duration::from_secs(31 * 60).as_secs();
    write_training_status(
        &profile,
        json!({
            "status": "failed",
            "message": "old boom",
            "timestamp": stale,
            "profile": profile.profile_name,
            "model_type": "hashlinear"
        }),
    );

    let decision = scheduler::cooldown_allows_training(&profile, 30, now).expect("cooldown decision");

    assert!(decision);
}

#[test]
fn scheduler_skips_profile_when_training_is_already_running() {
    let profile = write_profile(
        &format!("scheduler-training-running-{}", std::process::id()),
        json!({"local_model_type": "hashlinear"}),
    );
    write_training_status(
        &profile,
        json!({
            "status": "running",
            "message": "training in progress",
            "timestamp": current_unix_secs(),
            "profile": profile.profile_name,
            "model_type": "hashlinear"
        }),
    );

    let decision = scheduler::training_launch_allowed(&profile).expect("training launch decision");

    assert!(!decision);
}

#[test]
fn scheduler_spawn_detached_command_returns_before_child_exits() {
    let started = Instant::now();
    let pid = scheduler::spawn_detached_command(
        "/bin/sh",
        &["-c".to_string(), "sleep 1".to_string()],
        "detached-test",
    )
    .expect("spawn detached command");

    assert!(pid > 0);
    assert!(started.elapsed() < Duration::from_millis(500));

    std::thread::sleep(Duration::from_millis(1200));
}

#[test]
fn scheduler_command_uses_single_core_systemd_scope() {
    let root_dir = repo_root();
    let cmd = scheduler::build_training_subprocess_command(
        root_dir.to_str().expect("root dir string"),
        "default",
        "0",
    )
    .expect("build command");

    assert_eq!(cmd.program, "systemd-run");
    assert!(cmd.args.iter().any(|arg| arg == "--scope"));
    assert!(cmd.args.iter().any(|arg| arg == "AllowedCPUs=0"));
    assert!(cmd.args.iter().any(|arg| arg == "--working-directory"));
    assert!(cmd.args.windows(2).any(|window| {
        window[0] == "--working-directory" && window[1] == root_dir.display().to_string()
    }));
    assert!(
        cmd.args
            .windows(2)
            .any(|window| window[0] == "train-profile" && window[1] == "default")
    );
}

#[tokio::test]
async fn scheduler_run_once_trains_only_eligible_profiles() {
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-scheduler-run-once-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let eligible_dir = root_dir.join("configs/mod_profiles/eligible");
    std::fs::create_dir_all(&eligible_dir).expect("create eligible dir");
    std::fs::write(
        eligible_dir.join("profile.json"),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10
            }
        })
        .to_string(),
    )
    .expect("write eligible profile");

    let cooldown_dir = root_dir.join("configs/mod_profiles/cooldown");
    std::fs::create_dir_all(&cooldown_dir).expect("create cooldown dir");
    std::fs::write(
        cooldown_dir.join("profile.json"),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10
            }
        })
        .to_string(),
    )
    .expect("write cooldown profile");
    std::fs::write(
        cooldown_dir.join(".train_status.json"),
        json!({
            "status": "failed",
            "message": "previous run failed",
            "timestamp": current_unix_secs(),
            "profile": "cooldown",
            "model_type": "hashlinear",
            "sample_count": 20
        })
        .to_string(),
    )
    .expect("write cooldown status");

    let socket_path = root_dir.join("run/sample-store.sock");
    let settings = test_settings(&root_dir, &socket_path);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn({
        let response_socket = socket_path.clone();
        async move {
            serve_unix_requests_until_shutdown(
                &response_socket,
                move |request| match request {
                    SampleRpcRequest::GetSampleCount { profile, .. } if profile == "eligible" => {
                        SampleRpcResponse::ok(json!({ "sample_count": 20 }))
                    }
                    SampleRpcRequest::GetSampleCount { profile, .. } if profile == "cooldown" => {
                        SampleRpcResponse::ok(json!({ "sample_count": 20 }))
                    }
                    other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
                },
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        }
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let actions = scheduler::plan_training_round(&root_dir, &settings, current_unix_secs())
        .await
        .expect("plan round");

    assert!(actions.iter().any(|item| item.profile_name == "eligible"));
    assert!(!actions.iter().any(|item| item.profile_name == "cooldown"));

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&root_dir).expect("cleanup root");
}

#[tokio::test]
async fn scheduler_plan_round_skips_profiles_with_unreadable_history_storage() {
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-scheduler-corrupt-history-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let eligible_dir = root_dir.join("configs/mod_profiles/eligible");
    std::fs::create_dir_all(&eligible_dir).expect("create eligible dir");
    std::fs::write(
        eligible_dir.join("profile.json"),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10
            }
        })
        .to_string(),
    )
    .expect("write eligible profile");

    let corrupt_dir = root_dir.join("configs/mod_profiles/corrupt");
    std::fs::create_dir_all(&corrupt_dir).expect("create corrupt dir");
    std::fs::write(
        corrupt_dir.join("profile.json"),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10
            }
        })
        .to_string(),
    )
    .expect("write corrupt profile");

    let socket_path = root_dir.join("run/sample-store.sock");
    let settings = test_settings(&root_dir, &socket_path);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn({
        let response_socket = socket_path.clone();
        async move {
            serve_unix_requests_until_shutdown(
                &response_socket,
                move |request| match request {
                    SampleRpcRequest::GetSampleCount { profile, .. } if profile == "eligible" => {
                        SampleRpcResponse::ok(json!({ "sample_count": 20 }))
                    }
                    SampleRpcRequest::GetSampleCount { profile, .. } if profile == "corrupt" => {
                        SampleRpcResponse::err(
                            "failed to open RocksDB /tmp/corrupt/history.rocks: Corruption: Corrupt or unsupported format_version: 6"
                        )
                    }
                    other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
                },
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        }
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let actions = scheduler::plan_training_round(&root_dir, &settings, current_unix_secs())
        .await
        .expect("plan round");

    assert!(actions.iter().any(|item| item.profile_name == "eligible"));
    assert!(!actions.iter().any(|item| item.profile_name == "corrupt"));

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&root_dir).expect("cleanup root");
}
