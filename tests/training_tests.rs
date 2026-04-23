#![allow(dead_code)]

#[path = "../src/profile.rs"]
mod profile;
#[path = "../src/config.rs"]
mod config;
#[cfg(feature = "storage-debug")]
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/sample_rpc.rs"]
mod sample_rpc;
#[path = "../src/training.rs"]
mod training;

use config::Settings;
use profile::ModerationProfile;
#[cfg(feature = "storage-debug")]
use rocksdb::{DBWithThreadMode, MultiThreaded, Options};
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime};
use sample_rpc::{
    decode_request_line, decode_response_line, dispatch_request, encode_request_line,
    encode_response_line, send_unix_request, serve_one_unix_request,
    serve_unix_requests_until_shutdown, SampleRpcConfig,
    SampleRpcRequest, SampleRpcResponse, SampleRpcTransport,
};
#[cfg(feature = "storage-debug")]
use sample_rpc::serve_storage_sample_rpc_until_shutdown;
use training::{
    build_training_sample_request, cleanup_training_samples_via_rpc, evaluate_training_need,
    fetch_training_sample_count_via_rpc, fetch_training_samples_via_rpc,
    fetch_training_samples_via_unix_socket, run_profile_training,
    run_training_subprocess_from_args, train_bow_runtime, train_fasttext_runtime,
    train_hashlinear_runtime, write_training_status, TrainingSample,
};

fn write_profile(profile_name: &str, payload: serde_json::Value) -> ModerationProfile {
    let profile_dir = PathBuf::from("/services/apps/Prismguand-Rust")
        .join("configs")
        .join("mod_profiles")
        .join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(profile_dir.join("profile.json"), payload.to_string()).expect("write profile");
    ModerationProfile::load("/services/apps/Prismguand-Rust", profile_name).expect("load profile")
}

fn write_profile_into(
    root_dir: &std::path::Path,
    profile_name: &str,
    payload: serde_json::Value,
) -> ModerationProfile {
    let profile_dir = root_dir
        .join("configs")
        .join("mod_profiles")
        .join(profile_name);
    std::fs::create_dir_all(&profile_dir).expect("create profile dir");
    std::fs::write(profile_dir.join("profile.json"), payload.to_string()).expect("write profile");
    ModerationProfile::load(root_dir, profile_name).expect("load profile")
}

fn current_dir_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(feature = "storage-debug")]
fn encode_rocksdict_string(value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + 1);
    out.push(0x02);
    out.extend_from_slice(value.as_bytes());
    out
}

#[cfg(feature = "storage-debug")]
fn put_rocks_string(db: &DBWithThreadMode<MultiThreaded>, key: &str, value: &str) {
    db.put(encode_rocksdict_string(key), encode_rocksdict_string(value))
        .expect("put rocks string");
}

#[cfg(feature = "storage-debug")]
fn seed_sample_db(rocks_dir: &PathBuf, samples: &[serde_json::Value]) {
    if rocks_dir.exists() {
        std::fs::remove_dir_all(rocks_dir).expect("cleanup old rocks dir");
    }

    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, rocks_dir).expect("open rocksdb");

    for sample in samples {
        let id = sample["id"].as_u64().expect("sample id");
        let key = format!("sample:{id:020}");
        put_rocks_string(&db, &key, &sample.to_string());
    }

    let total = samples.len();
    let pass_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(0))
        .count();
    let violation_count = total - pass_count;

    put_rocks_string(&db, "meta:next_id", &(total as u64 + 1).to_string());
    put_rocks_string(&db, "meta:count", &total.to_string());
    put_rocks_string(&db, "meta:count:0", &pass_count.to_string());
    put_rocks_string(&db, "meta:count:1", &violation_count.to_string());
    drop(db);
}

#[test]
fn training_need_requires_minimum_samples() {
    let profile = write_profile(
        &format!("training-min-samples-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10,
                "retrain_interval_minutes": 60
            }
        }),
    );

    let decision = evaluate_training_need(&profile, 9, None, SystemTime::UNIX_EPOCH)
        .expect("training decision");

    assert!(!decision.should_train);
    assert_eq!(decision.reason, "insufficient_samples");
    assert_eq!(decision.min_samples, 10);
    assert_eq!(decision.sample_count, 9);
}

#[test]
fn training_need_requests_initial_training_when_model_missing() {
    let profile = write_profile(
        &format!("training-missing-model-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10,
                "retrain_interval_minutes": 60
            }
        }),
    );

    let decision = evaluate_training_need(&profile, 10, None, SystemTime::UNIX_EPOCH)
        .expect("training decision");

    assert!(decision.should_train);
    assert_eq!(decision.reason, "model_missing");
}

#[test]
fn training_need_respects_retrain_interval() {
    let profile = write_profile(
        &format!("training-retrain-interval-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "min_samples": 10,
                "retrain_interval_minutes": 60
            }
        }),
    );

    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);
    let fresh_model = now - Duration::from_secs(30 * 60);
    let stale_model = now - Duration::from_secs(90 * 60);

    let fresh = evaluate_training_need(&profile, 20, Some(fresh_model), now)
        .expect("fresh training decision");
    assert!(!fresh.should_train);
    assert_eq!(fresh.reason, "retrain_interval_not_elapsed");

    let stale = evaluate_training_need(&profile, 20, Some(stale_model), now)
        .expect("stale training decision");
    assert!(stale.should_train);
    assert_eq!(stale.reason, "retrain_interval_elapsed");
}

#[test]
fn training_builds_sample_request_from_profile_strategy() {
    let profile = write_profile(
        &format!("training-request-strategy-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_samples": 32,
                "sample_loading": "latest_full"
            }
        }),
    );

    let request = build_training_sample_request(&profile).expect("build sample request");
    assert_eq!(
        request,
        SampleRpcRequest::LoadBalancedLatestSamples {
            profile: profile.profile_name.clone(),
            db_path: profile.history_rocks_path().display().to_string(),
            max_samples: 32,
        }
    );
}

#[test]
fn training_builds_sample_request_from_random_duplicate_strategy() {
    let profile = write_profile(
        &format!("training-request-random-duplicate-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_samples": 32,
                "sample_loading": "random_duplicate"
            }
        }),
    );

    let request = build_training_sample_request(&profile).expect("build sample request");
    assert_eq!(
        request,
        SampleRpcRequest::LoadBalancedRandomDuplicateSamples {
            profile: profile.profile_name.clone(),
            db_path: profile.history_rocks_path().display().to_string(),
            max_samples: 32,
        }
    );
}

#[tokio::test]
async fn training_fetches_samples_over_unix_socket() {
    let profile = write_profile(
        &format!("training-fetch-samples-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_samples": 4,
                "sample_loading": "balanced_undersample"
            }
        }),
    );

    let socket_dir = PathBuf::from(format!("/tmp/prismguard-train-fetch-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let expected_request = SampleRpcRequest::LoadBalancedSamples {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path().display().to_string(),
        max_samples: 4,
    };
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_one_unix_request(&server_socket, |request| {
            assert_eq!(request, expected_request);
            Ok(SampleRpcResponse {
                ok: true,
                result: Some(json!({
                    "samples": [
                        {"id": 11, "text": "safe text", "label": 0, "category": null, "created_at": "2026-04-03 12:00:00"},
                        {"id": 12, "text": "unsafe text", "label": 1, "category": "violence", "created_at": "2026-04-03 12:01:00"}
                    ]
                })),
                error: None,
            })
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let samples =
        fetch_training_samples_via_unix_socket(&socket_path, &profile).await.expect("fetch samples");
    assert_eq!(
        samples,
        vec![
            TrainingSample {
                id: Some(11),
                text: "safe text".to_string(),
                label: 0,
                category: None,
                created_at: Some("2026-04-03 12:00:00".to_string()),
            },
            TrainingSample {
                id: Some(12),
                text: "unsafe text".to_string(),
                label: 1,
                category: Some("violence".to_string()),
                created_at: Some("2026-04-03 12:01:00".to_string()),
            },
        ]
    );

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn training_fetches_samples_via_sample_rpc_config() {
    let profile = write_profile(
        &format!("training-fetch-rpc-config-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_samples": 2,
                "sample_loading": "latest_full"
            }
        }),
    );

    let socket_dir = PathBuf::from(format!("/tmp/prismguard-train-rpc-config-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let expected_request = SampleRpcRequest::LoadBalancedLatestSamples {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path().display().to_string(),
        max_samples: 2,
    };
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_one_unix_request(&server_socket, |request| {
            assert_eq!(request, expected_request);
            Ok(SampleRpcResponse::ok(json!({
                "samples": [
                    {"id": 21, "text": "safe", "label": 0}
                ]
            })))
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let samples = fetch_training_samples_via_rpc(&rpc, &profile)
        .await
        .expect("fetch samples via rpc config");
    assert_eq!(
        samples,
        vec![TrainingSample {
            id: Some(21),
            text: "safe".to_string(),
            label: 0,
            category: None,
            created_at: None,
        }]
    );

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn training_fetches_sample_count_via_sample_rpc_config() {
    let profile = write_profile(
        &format!("training-fetch-count-rpc-config-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear"
        }),
    );

    let socket_dir =
        PathBuf::from(format!("/tmp/prismguard-train-count-rpc-config-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let expected_request = SampleRpcRequest::GetSampleCount {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path().display().to_string(),
    };
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_one_unix_request(&server_socket, |request| {
            assert_eq!(request, expected_request);
            Ok(SampleRpcResponse::ok(json!({
                "sample_count": 23
            })))
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let sample_count = fetch_training_sample_count_via_rpc(&rpc, &profile)
        .await
        .expect("fetch sample count via rpc config");
    assert_eq!(sample_count, 23);

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn training_cleans_up_samples_via_sample_rpc_config() {
    let profile = write_profile(
        &format!("training-cleanup-rpc-config-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_db_items": 32
            }
        }),
    );

    let socket_dir =
        PathBuf::from(format!("/tmp/prismguard-train-cleanup-rpc-config-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let expected_request = SampleRpcRequest::CleanupExcessSamples {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path().display().to_string(),
        max_items: 32,
    };
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_one_unix_request(&server_socket, |request| {
            assert_eq!(request, expected_request);
            Ok(SampleRpcResponse::ok(json!({
                "removed": 5
            })))
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let removed = cleanup_training_samples_via_rpc(&rpc, &profile)
        .await
        .expect("cleanup training samples via rpc config");
    assert_eq!(removed, 5);

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[test]
fn training_status_writer_persists_python_style_fields() {
    let profile = write_profile(
        &format!("training-status-shape-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear"
        }),
    );

    write_training_status(
        &profile,
        &json!({
            "status": "failed",
            "message": "previous run failed",
            "timestamp": 1_744_000_000u64,
            "profile": profile.profile_name,
            "model_type": "hashlinear",
            "sample_count": 19
        }),
    )
    .expect("write status");

    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "failed");
    assert_eq!(status["message"], "previous run failed");
    assert_eq!(status["sample_count"], 19);
}

#[tokio::test]
async fn run_profile_training_cleans_samples_then_trains_hashlinear_runtime() {
    let profile = write_profile(
        &format!("training-run-profile-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "n_features": 32,
                "epochs": 2,
                "batch_size": 2,
                "alpha": 0.0001,
                "max_seconds": 10
            }
        }),
    );

    let runtime_json = profile.hashlinear_runtime_json_path();
    let runtime_coef = profile.hashlinear_runtime_coef_path();
    let model_marker = profile.hashlinear_model_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&runtime_coef);
    let _ = std::fs::remove_file(&model_marker);

    let socket_dir = PathBuf::from(format!("/tmp/prismguard-run-profile-rpc-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let profile_name = profile.profile_name.clone();
    let db_path = profile.history_rocks_path().display().to_string();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_socket = socket_path.clone();
    let server_requests = Arc::clone(&requests);
    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let response_profile_name = profile_name.clone();
        let response_db_path = db_path.clone();
        let recorded_requests = Arc::clone(&server_requests);
        let server = tokio::spawn(async move {
            serve_unix_requests_until_shutdown(
                &server_socket,
                move |request| {
                    recorded_requests.lock().expect("lock requests").push(request.clone());
                    match request {
                        SampleRpcRequest::CleanupExcessSamples {
                            profile,
                            db_path,
                            max_items,
                        } => {
                            assert_eq!(profile, response_profile_name);
                            assert_eq!(db_path, response_db_path);
                            assert_eq!(max_items, 12);
                            SampleRpcResponse::ok(json!({ "removed": 3 }))
                        }
                        SampleRpcRequest::LoadBalancedSamples {
                            profile,
                            db_path,
                            max_samples,
                        } => {
                            assert_eq!(profile, response_profile_name);
                            assert_eq!(db_path, response_db_path);
                            assert_eq!(max_samples, 4);
                            SampleRpcResponse::ok(json!({
                                "samples": [
                                    {"id": 1, "text": "safe text", "label": 0},
                                    {"id": 2, "text": "hello world", "label": 0},
                                    {"id": 3, "text": "kill threat", "label": 1, "category": "violence"},
                                    {"id": 4, "text": "bomb attack", "label": 1, "category": "violence"}
                                ]
                            }))
                        }
                        other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
                    }
                },
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(150)).await;
        let _ = shutdown_tx.send(());
        let result = server.await.expect("join nested server");
        result
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let output = run_profile_training(&rpc, &profile)
        .await
        .expect("run profile training");
    assert_eq!(output.sample_count, 4);
    assert_eq!(output.pass_count, 2);
    assert_eq!(output.violation_count, 2);
    assert!(runtime_json.exists());
    assert!(runtime_coef.exists());
    assert!(model_marker.exists());

    server.await.expect("join server").expect("server result");
    let requests = requests.lock().expect("lock requests");
    assert_eq!(requests.len(), 2);
    assert!(matches!(
        requests.first(),
        Some(SampleRpcRequest::CleanupExcessSamples { .. })
    ));
    assert!(matches!(
        requests.get(1),
        Some(SampleRpcRequest::LoadBalancedSamples { .. })
    ));

    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn run_profile_training_cleans_samples_then_trains_bow_runtime() {
    let profile = write_profile(
        &format!("training-run-profile-bow-{}", std::process::id()),
        json!({
            "local_model_type": "bow",
            "bow_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "max_features": 16
            }
        }),
    );

    let runtime_json = profile.bow_runtime_json_path();
    let runtime_coef = profile.bow_runtime_coef_path();
    let model_marker = profile.bow_model_path();
    let vectorizer_marker = profile.bow_vectorizer_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&runtime_coef);
    let _ = std::fs::remove_file(&model_marker);
    let _ = std::fs::remove_file(&vectorizer_marker);

    let socket_dir = PathBuf::from(format!("/tmp/prismguard-run-profile-bow-rpc-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let profile_name = profile.profile_name.clone();
    let db_path = profile.history_rocks_path().display().to_string();
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let response_profile_name = profile_name.clone();
        let response_db_path = db_path.clone();
        let server = tokio::spawn(async move {
            serve_unix_requests_until_shutdown(
                &server_socket,
                move |request| match request {
                    SampleRpcRequest::CleanupExcessSamples {
                        profile,
                        db_path,
                        max_items,
                    } => {
                        assert_eq!(profile, response_profile_name);
                        assert_eq!(db_path, response_db_path);
                        assert_eq!(max_items, 12);
                        SampleRpcResponse::ok(json!({ "removed": 1 }))
                    }
                    SampleRpcRequest::LoadBalancedSamples {
                        profile,
                        db_path,
                        max_samples,
                    } => {
                        assert_eq!(profile, response_profile_name);
                        assert_eq!(db_path, response_db_path);
                        assert_eq!(max_samples, 4);
                        SampleRpcResponse::ok(json!({
                            "samples": [
                                {"id": 1, "text": "hello friend", "label": 0},
                                {"id": 2, "text": "safe weather", "label": 0},
                                {"id": 3, "text": "blocked phrase", "label": 1},
                                {"id": 4, "text": "blocked threat", "label": 1}
                            ]
                        }))
                    }
                    other => panic!("unexpected request: {other:?}"),
                },
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        server.await.expect("join inner server")
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let output = run_profile_training(&rpc, &profile)
        .await
        .expect("run profile training");
    assert_eq!(output.sample_count, 4);
    assert_eq!(output.pass_count, 2);
    assert_eq!(output.violation_count, 2);
    assert_eq!(output.runtime_json_path, runtime_json);
    assert_eq!(output.runtime_coef_path, runtime_coef);
    assert_eq!(output.model_marker_path, model_marker);

    assert!(runtime_json.exists());
    assert!(runtime_coef.exists());
    assert!(model_marker.exists());
    assert!(vectorizer_marker.exists());

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn run_profile_training_cleans_samples_then_trains_fasttext_runtime() {
    let profile = write_profile(
        &format!("training-run-profile-fasttext-{}", std::process::id()),
        json!({
            "local_model_type": "fasttext",
            "fasttext_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4
            }
        }),
    );

    let runtime_json = profile.fasttext_runtime_json_path();
    let model_marker = profile.fasttext_model_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&model_marker);

    let socket_dir = PathBuf::from(format!("/tmp/prismguard-run-profile-fasttext-rpc-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: socket_path.clone(),
    };

    let profile_name = profile.profile_name.clone();
    let db_path = profile.history_rocks_path().display().to_string();
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
        let response_profile_name = profile_name.clone();
        let response_db_path = db_path.clone();
        let server = tokio::spawn(async move {
            serve_unix_requests_until_shutdown(
                &server_socket,
                move |request| match request {
                    SampleRpcRequest::CleanupExcessSamples {
                        profile,
                        db_path,
                        max_items,
                    } => {
                        assert_eq!(profile, response_profile_name);
                        assert_eq!(db_path, response_db_path);
                        assert_eq!(max_items, 12);
                        SampleRpcResponse::ok(json!({ "removed": 1 }))
                    }
                    SampleRpcRequest::LoadBalancedSamples {
                        profile,
                        db_path,
                        max_samples,
                    } => {
                        assert_eq!(profile, response_profile_name);
                        assert_eq!(db_path, response_db_path);
                        assert_eq!(max_samples, 4);
                        SampleRpcResponse::ok(json!({
                            "samples": [
                                {"id": 1, "text": "hello friend", "label": 0},
                                {"id": 2, "text": "safe weather", "label": 0},
                                {"id": 3, "text": "blocked phrase", "label": 1},
                                {"id": 4, "text": "blocked threat", "label": 1}
                            ]
                        }))
                    }
                    other => panic!("unexpected request: {other:?}"),
                },
                async move {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        server.await.expect("join inner server")
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let output = run_profile_training(&rpc, &profile)
        .await
        .expect("run profile training");
    assert_eq!(output.sample_count, 4);
    assert_eq!(output.pass_count, 2);
    assert_eq!(output.violation_count, 2);
    assert_eq!(output.runtime_json_path, runtime_json);
    assert_eq!(output.model_marker_path, model_marker);

    assert!(runtime_json.exists());
    assert!(model_marker.exists());

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn training_subprocess_marks_success_after_runtime_write() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-success-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "n_features": 32,
                "epochs": 2,
                "batch_size": 2,
                "alpha": 0.0001,
                "max_seconds": 10
            }
        }),
    );

    let socket_path = root_dir.join("run/sample-store.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let requests = Arc::new(Mutex::new(Vec::new()));
    let server_socket = socket_path.clone();
    let expected_profile_name = profile.profile_name.clone();
    let expected_db_path = profile.history_rocks_path().display().to_string();
    let server_requests = Arc::clone(&requests);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            move |request| {
                server_requests.lock().expect("lock requests").push(request.clone());
                match request {
                    SampleRpcRequest::GetSampleCount {
                        profile: request_profile,
                        db_path: request_db_path,
                    } => {
                        assert_eq!(request_profile, expected_profile_name);
                        assert_eq!(request_db_path, expected_db_path);
                        SampleRpcResponse::ok(json!({ "sample_count": 20 }))
                    }
                    SampleRpcRequest::CleanupExcessSamples {
                        profile: request_profile,
                        db_path: request_db_path,
                        max_items,
                    } => {
                        assert_eq!(request_profile, expected_profile_name);
                        assert_eq!(request_db_path, expected_db_path);
                        assert_eq!(max_items, 12);
                        SampleRpcResponse::ok(json!({ "removed": 3 }))
                    }
                    SampleRpcRequest::LoadBalancedSamples {
                        profile: request_profile,
                        db_path: request_db_path,
                        max_samples,
                    } => {
                        assert_eq!(request_profile, expected_profile_name);
                        assert_eq!(request_db_path, expected_db_path);
                        assert_eq!(max_samples, 4);
                        SampleRpcResponse::ok(json!({
                            "samples": [
                                {"id": 1, "text": "safe text", "label": 0},
                                {"id": 2, "text": "hello world", "label": 0},
                                {"id": 3, "text": "kill threat", "label": 1, "category": "violence"},
                                {"id": 4, "text": "bomb attack", "label": 1, "category": "violence"}
                            ]
                        }))
                    }
                    other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
                }
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");

    result.expect("training subprocess result");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "success");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "hashlinear");
    assert_eq!(status["sample_count"], 4);
    assert_eq!(status["pass_count"], 2);
    assert_eq!(status["violation_count"], 2);
    assert_eq!(
        status["runtime_json_path"],
        serde_json::json!(profile.hashlinear_runtime_json_path())
    );
    assert_eq!(
        status["runtime_coef_path"],
        serde_json::json!(profile.hashlinear_runtime_coef_path())
    );
    assert_eq!(
        status["model_marker_path"],
        serde_json::json!(profile.hashlinear_model_path())
    );
    assert!(profile.hashlinear_runtime_json_path().exists());
    assert!(profile.hashlinear_runtime_coef_path().exists());
    assert!(profile.hashlinear_model_path().exists());

    let requests = requests.lock().expect("requests lock");
    assert!(requests.iter().any(|req| matches!(req, SampleRpcRequest::GetSampleCount { .. })));
    assert!(requests.iter().any(|req| matches!(req, SampleRpcRequest::CleanupExcessSamples { .. })));
    assert!(requests.iter().any(|req| matches!(req, SampleRpcRequest::LoadBalancedSamples { .. })));

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[tokio::test]
async fn training_subprocess_marks_success_after_bow_runtime_write() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-bow-success-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "bow",
            "bow_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "max_features": 16
            }
        }),
    );

    let socket_path = root_dir.join("run/sample-store.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let server_socket = socket_path.clone();
    let expected_profile_name = profile.profile_name.clone();
    let expected_db_path = profile.history_rocks_path().display().to_string();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            move |request| match request {
                SampleRpcRequest::GetSampleCount {
                    profile: request_profile,
                    db_path: request_db_path,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    SampleRpcResponse::ok(json!({ "sample_count": 4 }))
                }
                SampleRpcRequest::CleanupExcessSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_items,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_items, 12);
                    SampleRpcResponse::ok(json!({ "removed": 0 }))
                }
                SampleRpcRequest::LoadBalancedSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_samples,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_samples, 4);
                    SampleRpcResponse::ok(json!({
                        "samples": [
                            {"id": 1, "text": "hello friend", "label": 0},
                            {"id": 2, "text": "safe weather", "label": 0},
                            {"id": 3, "text": "blocked phrase", "label": 1},
                            {"id": 4, "text": "blocked threat", "label": 1}
                        ]
                    }))
                }
                other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");

    result.expect("training subprocess result");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "success");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "bow");
    assert_eq!(status["sample_count"], 4);
    assert_eq!(status["pass_count"], 2);
    assert_eq!(status["violation_count"], 2);
    assert_eq!(
        status["runtime_json_path"],
        serde_json::json!(profile.bow_runtime_json_path())
    );
    assert_eq!(
        status["runtime_coef_path"],
        serde_json::json!(profile.bow_runtime_coef_path())
    );
    assert_eq!(
        status["model_marker_path"],
        serde_json::json!(profile.bow_model_path())
    );
    assert!(profile.bow_runtime_json_path().exists());
    assert!(profile.bow_runtime_coef_path().exists());
    assert!(profile.bow_model_path().exists());
    assert!(profile.bow_vectorizer_path().exists());

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[tokio::test]
async fn training_subprocess_marks_success_after_fasttext_runtime_write() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-fasttext-success-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "fasttext",
            "fasttext_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4
            }
        }),
    );

    let socket_path = root_dir.join("run/sample-store.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let server_socket = socket_path.clone();
    let expected_profile_name = profile.profile_name.clone();
    let expected_db_path = profile.history_rocks_path().display().to_string();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            move |request| match request {
                SampleRpcRequest::GetSampleCount {
                    profile: request_profile,
                    db_path: request_db_path,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    SampleRpcResponse::ok(json!({ "sample_count": 4 }))
                }
                SampleRpcRequest::CleanupExcessSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_items,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_items, 12);
                    SampleRpcResponse::ok(json!({ "removed": 0 }))
                }
                SampleRpcRequest::LoadBalancedSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_samples,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_samples, 4);
                    SampleRpcResponse::ok(json!({
                        "samples": [
                            {"id": 1, "text": "hello friend", "label": 0},
                            {"id": 2, "text": "safe weather", "label": 0},
                            {"id": 3, "text": "blocked phrase", "label": 1},
                            {"id": 4, "text": "blocked threat", "label": 1}
                        ]
                    }))
                }
                other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");

    result.expect("training subprocess result");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "success");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "fasttext");
    assert_eq!(status["sample_count"], 4);
    assert_eq!(status["pass_count"], 2);
    assert_eq!(status["violation_count"], 2);
    assert_eq!(
        status["runtime_json_path"],
        serde_json::json!(profile.fasttext_runtime_json_path())
    );
    assert_eq!(
        status["runtime_coef_path"],
        serde_json::json!(profile.fasttext_runtime_json_path())
    );
    assert_eq!(
        status["model_marker_path"],
        serde_json::json!(profile.fasttext_model_path())
    );
    assert!(profile.fasttext_runtime_json_path().exists());
    assert!(profile.fasttext_model_path().exists());

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[tokio::test]
async fn training_subprocess_marks_failed_when_rpc_unavailable() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-failed-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "n_features": 32
            }
        }),
    );

    let socket_path = root_dir.join("run/missing.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    assert!(result.is_err(), "expected subprocess training to fail without RPC");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "failed");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "hashlinear");
    assert_eq!(status["sample_count"], 0);
    assert!(
        status["message"]
            .as_str()
            .expect("error message")
            .contains("sample RPC")
            || status["message"]
                .as_str()
                .expect("error message")
                .contains("connect")
    );

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[tokio::test]
async fn training_subprocess_marks_failed_when_bow_samples_are_single_label() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-bow-failed-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "bow",
            "bow_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4,
                "max_features": 16
            }
        }),
    );

    let socket_path = root_dir.join("run/sample-store.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let server_socket = socket_path.clone();
    let expected_profile_name = profile.profile_name.clone();
    let expected_db_path = profile.history_rocks_path().display().to_string();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            move |request| match request {
                SampleRpcRequest::GetSampleCount {
                    profile: request_profile,
                    db_path: request_db_path,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    SampleRpcResponse::ok(json!({ "sample_count": 4 }))
                }
                SampleRpcRequest::CleanupExcessSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_items,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_items, 12);
                    SampleRpcResponse::ok(json!({ "removed": 0 }))
                }
                SampleRpcRequest::LoadBalancedSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_samples,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_samples, 4);
                    SampleRpcResponse::ok(json!({
                        "samples": [
                            {"id": 1, "text": "safe text", "label": 0},
                            {"id": 2, "text": "more safe text", "label": 0}
                        ]
                    }))
                }
                other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");

    assert!(result.is_err(), "single-label bow training should fail");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "failed");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "bow");
    assert_eq!(status["sample_count"], 4);
    assert!(
        status["message"]
            .as_str()
            .expect("error message")
            .contains("both labels")
    );

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[tokio::test]
async fn training_subprocess_marks_failed_when_fasttext_samples_are_single_label() {
    let _guard = current_dir_test_lock().lock().expect("current dir lock");
    let root_dir = PathBuf::from(format!(
        "/tmp/prismguard-training-subprocess-fasttext-failed-{}",
        std::process::id()
    ));
    if root_dir.exists() {
        std::fs::remove_dir_all(&root_dir).expect("cleanup old root");
    }
    std::fs::create_dir_all(root_dir.join("configs/mod_profiles")).expect("create profile root");
    std::fs::create_dir_all(root_dir.join("run")).expect("create run dir");

    let profile = write_profile_into(
        &root_dir,
        "default",
        json!({
            "local_model_type": "fasttext",
            "fasttext_training": {
                "max_db_items": 12,
                "sample_loading": "balanced_undersample",
                "max_samples": 4
            }
        }),
    );

    let socket_path = root_dir.join("run/sample-store.sock");
    std::fs::write(
        root_dir.join(".env"),
        format!(
            "TRAINING_DATA_RPC_ENABLED=1\nTRAINING_DATA_RPC_TRANSPORT=unix\nTRAINING_DATA_RPC_UNIX_SOCKET={}\n",
            socket_path.display()
        ),
    )
    .expect("write env");

    let server_socket = socket_path.clone();
    let expected_profile_name = profile.profile_name.clone();
    let expected_db_path = profile.history_rocks_path().display().to_string();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            move |request| match request {
                SampleRpcRequest::GetSampleCount {
                    profile: request_profile,
                    db_path: request_db_path,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    SampleRpcResponse::ok(json!({ "sample_count": 4 }))
                }
                SampleRpcRequest::CleanupExcessSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_items,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_items, 12);
                    SampleRpcResponse::ok(json!({ "removed": 0 }))
                }
                SampleRpcRequest::LoadBalancedSamples {
                    profile: request_profile,
                    db_path: request_db_path,
                    max_samples,
                } => {
                    assert_eq!(request_profile, expected_profile_name);
                    assert_eq!(request_db_path, expected_db_path);
                    assert_eq!(max_samples, 4);
                    SampleRpcResponse::ok(json!({
                        "samples": [
                            {"id": 1, "text": "blocked phrase", "label": 1},
                            {"id": 2, "text": "blocked threat", "label": 1}
                        ]
                    }))
                }
                other => SampleRpcResponse::err(format!("unexpected request: {other:?}")),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let previous_dir = std::env::current_dir().expect("current dir");
    std::env::set_current_dir(&root_dir).expect("set current dir");
    let result = run_training_subprocess_from_args(&[
        "prismguard-rust".to_string(),
        "train-profile".to_string(),
        "default".to_string(),
    ])
    .await;
    std::env::set_current_dir(previous_dir).expect("restore current dir");

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");

    assert!(result.is_err(), "single-label fasttext training should fail");
    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "failed");
    assert_eq!(status["profile"], "default");
    assert_eq!(status["model_type"], "fasttext");
    assert_eq!(status["sample_count"], 4);
    assert!(
        status["message"]
            .as_str()
            .expect("error message")
            .contains("both labels")
    );

    std::fs::remove_dir_all(&root_dir).expect("cleanup root dir");
}

#[test]
fn training_writes_hashlinear_runtime_files() {
    let profile = write_profile(
        &format!("training-runtime-export-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear",
            "hashlinear_training": {
                "n_features": 64,
                "epochs": 3,
                "batch_size": 2,
                "alpha": 0.0001,
                "max_seconds": 10,
                "sample_loading": "balanced_undersample",
                "max_samples": 8
            }
        }),
    );

    let runtime_json = profile.base_dir.join("hashlinear_runtime.json");
    let runtime_coef = profile.base_dir.join("hashlinear_runtime.coef.f32");
    let model_marker = profile.hashlinear_model_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&runtime_coef);
    let _ = std::fs::remove_file(&model_marker);

    train_hashlinear_runtime(
        &profile,
        &[
            TrainingSample {
                id: Some(1),
                text: "hello friend".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(2),
                text: "nice weather".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(3),
                text: "kill threat".to_string(),
                label: 1,
                category: Some("violence".to_string()),
                created_at: None,
            },
            TrainingSample {
                id: Some(4),
                text: "bomb attack".to_string(),
                label: 1,
                category: Some("violence".to_string()),
                created_at: None,
            },
        ],
    )
    .expect("train hashlinear runtime");

    assert!(runtime_json.exists());
    assert!(runtime_coef.exists());
    assert!(profile.local_model_exists());

    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&runtime_json).expect("read runtime json"))
            .expect("parse runtime json");
    assert_eq!(metadata["runtime_version"], 1);
    assert_eq!(metadata["classes"], json!([0, 1]));
    assert_eq!(metadata["n_features"], 64);
    assert_eq!(metadata["source_model"], json!(profile.hashlinear_model_path()));
    assert_eq!(metadata["cfg"]["lowercase"], true);

    let coef_bytes = std::fs::read(&runtime_coef).expect("read runtime coef");
    assert_eq!(coef_bytes.len(), 64 * std::mem::size_of::<f32>());

    let marker: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&model_marker).expect("read model marker"))
            .expect("parse model marker");
    assert_eq!(marker["kind"], "hashlinear_runtime_marker");
    assert_eq!(marker["runtime_json"], json!(runtime_json));
    assert_eq!(marker["runtime_coef"], json!(runtime_coef));
    assert!(marker["trained_at"].as_u64().is_some());
}

#[test]
fn training_writes_bow_runtime_files() {
    let profile = write_profile(
        &format!("training-bow-runtime-export-{}", std::process::id()),
        json!({
            "local_model_type": "bow",
            "bow_training": {
                "max_features": 16,
                "sample_loading": "balanced_undersample",
                "max_samples": 8
            }
        }),
    );

    let runtime_json = profile.base_dir.join("bow_runtime.json");
    let runtime_coef = profile.base_dir.join("bow_runtime.coef.f32");
    let model_marker = profile.bow_model_path();
    let vectorizer_marker = profile.bow_vectorizer_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&runtime_coef);
    let _ = std::fs::remove_file(&model_marker);
    let _ = std::fs::remove_file(&vectorizer_marker);

    train_bow_runtime(
        &profile,
        &[
            TrainingSample {
                id: Some(1),
                text: "hello friend".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(2),
                text: "calm weather".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(3),
                text: "blocked phrase".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
            TrainingSample {
                id: Some(4),
                text: "blocked threat".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
        ],
    )
    .expect("train bow runtime");

    assert!(runtime_json.exists());
    assert!(runtime_coef.exists());
    assert!(model_marker.exists());
    assert!(vectorizer_marker.exists());

    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&runtime_json).expect("read runtime json"))
            .expect("parse runtime json");
    assert_eq!(metadata["runtime_version"], 1);
    assert_eq!(metadata["classes"], json!([0, 1]));
    assert_eq!(metadata["source_model"], json!(profile.bow_model_path()));
    assert_eq!(
        metadata["source_vectorizer"],
        json!(profile.bow_vectorizer_path())
    );
    assert!(
        metadata["vocabulary"]
            .as_array()
            .expect("vocabulary array")
            .len()
            >= 2
    );
    assert_eq!(metadata["tokenizer"]["lowercase"], true);
    assert_eq!(metadata["tokenizer"]["split_whitespace"], true);

    let coef_bytes = std::fs::read(&runtime_coef).expect("read runtime coef");
    assert_eq!(
        coef_bytes.len(),
        metadata["vocabulary"].as_array().expect("vocabulary array").len() * std::mem::size_of::<f32>()
    );

    let model_marker_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&model_marker).expect("read bow model marker"),
    )
    .expect("parse bow model marker");
    assert_eq!(model_marker_payload["kind"], "bow_runtime_marker");
    assert_eq!(model_marker_payload["runtime_json"], json!(runtime_json));
    assert_eq!(model_marker_payload["runtime_coef"], json!(runtime_coef));
    assert!(model_marker_payload["trained_at"].as_u64().is_some());

    let vectorizer_marker_payload: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&vectorizer_marker).expect("read bow vectorizer marker"),
    )
    .expect("parse bow vectorizer marker");
    assert_eq!(
        vectorizer_marker_payload["kind"],
        "bow_vectorizer_runtime_marker"
    );
    assert_eq!(vectorizer_marker_payload["runtime_json"], json!(runtime_json));
    assert!(vectorizer_marker_payload["trained_at"].as_u64().is_some());
}

#[test]
fn training_writes_fasttext_runtime_file() {
    let profile = write_profile(
        &format!("training-fasttext-runtime-export-{}", std::process::id()),
        json!({
            "local_model_type": "fasttext",
            "fasttext_training": {
                "sample_loading": "balanced_undersample",
                "max_samples": 8
            }
        }),
    );

    let runtime_json = profile.base_dir.join("fasttext_runtime.json");
    let model_marker = profile.fasttext_model_path();
    let _ = std::fs::remove_file(&runtime_json);
    let _ = std::fs::remove_file(&model_marker);

    train_fasttext_runtime(
        &profile,
        &[
            TrainingSample {
                id: Some(1),
                text: "hello friend".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(2),
                text: "safe weather".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(3),
                text: "blocked phrase".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
            TrainingSample {
                id: Some(4),
                text: "blocked threat".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
        ],
    )
    .expect("train fasttext runtime");

    assert!(runtime_json.exists());
    assert!(model_marker.exists());

    let metadata: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&runtime_json).expect("read runtime json"))
            .expect("parse runtime json");
    assert_eq!(metadata["runtime_version"], 1);
    assert_eq!(metadata["classes"], json!([0, 1]));
    assert_eq!(metadata["source_model"], json!(profile.fasttext_model_path()));
    assert_eq!(metadata["tokenizer"]["lowercase"], true);
    assert_eq!(metadata["tokenizer"]["split_whitespace"], true);
    assert!(
        metadata["weights"]
            .as_array()
            .expect("weights array")
            .len()
            >= 2
    );
    assert!(
        metadata["weights"]
            .as_array()
            .expect("weights array")
            .iter()
            .all(|item| item.get("token").is_some() && item.get("weight").is_some())
    );

    let marker: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&model_marker).expect("read fasttext marker"),
    )
    .expect("parse fasttext marker");
    assert_eq!(marker["kind"], "fasttext_runtime_marker");
    assert_eq!(marker["runtime_json"], json!(runtime_json));
    assert!(marker["trained_at"].as_u64().is_some());
}

#[test]
fn training_rejects_bow_runtime_when_only_one_label_exists() {
    let profile = write_profile(
        &format!("training-bow-single-label-{}", std::process::id()),
        json!({
            "local_model_type": "bow",
            "bow_training": {
                "max_features": 16
            }
        }),
    );

    let err = train_bow_runtime(
        &profile,
        &[
            TrainingSample {
                id: Some(1),
                text: "safe text".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
            TrainingSample {
                id: Some(2),
                text: "more safe text".to_string(),
                label: 0,
                category: None,
                created_at: None,
            },
        ],
    )
    .expect_err("single-label bow training should fail");

    assert!(err.to_string().contains("both labels"));
}

#[test]
fn training_rejects_fasttext_runtime_when_only_one_label_exists() {
    let profile = write_profile(
        &format!("training-fasttext-single-label-{}", std::process::id()),
        json!({
            "local_model_type": "fasttext"
        }),
    );

    let err = train_fasttext_runtime(
        &profile,
        &[
            TrainingSample {
                id: Some(1),
                text: "blocked phrase".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
            TrainingSample {
                id: Some(2),
                text: "blocked threat".to_string(),
                label: 1,
                category: Some("policy".to_string()),
                created_at: None,
            },
        ],
    )
    .expect_err("single-label fasttext training should fail");

    assert!(err.to_string().contains("both labels"));
}

#[test]
fn profile_hashlinear_runtime_paths_use_shared_prefix() {
    let profile = write_profile(
        &format!("training-runtime-paths-{}", std::process::id()),
        json!({
            "local_model_type": "hashlinear"
        }),
    );

    assert_eq!(
        profile.hashlinear_runtime_json_path(),
        profile.hashlinear_runtime_prefix().with_extension("json")
    );
    assert_eq!(
        profile.hashlinear_runtime_coef_path(),
        profile.hashlinear_runtime_prefix().with_extension("coef.f32")
    );
}

#[test]
fn sample_rpc_defaults_to_unix_socket_runtime_dir() {
    let settings = Settings {
        host: "127.0.0.1".to_string(),
        port: 0,
        debug: true,
        log_level: "info".to_string(),
        access_log_file: "logs/access.log".to_string(),
        moderation_log_file: "logs/moderation.log".to_string(),
        training_log_file: "logs/training.log".to_string(),
        training_data_rpc_enabled: true,
        training_data_rpc_transport: "unix".to_string(),
        training_data_rpc_unix_socket: "/tmp/prismguard-rpc-test/run/sample.sock".to_string(),
        training_scheduler_enabled: true,
        training_scheduler_interval_minutes: 10,
        training_scheduler_failure_cooldown_minutes: 30,
        training_subprocess_allowed_cpus: "0".to_string(),
        root_dir: PathBuf::from("/services/apps/Prismguand-Rust"),
        env_map: Default::default(),
    };

    let rpc = SampleRpcConfig::from_settings(&settings).expect("rpc config");
    assert!(rpc.enabled);
    assert_eq!(rpc.transport, SampleRpcTransport::Unix);
    assert_eq!(rpc.unix_socket, PathBuf::from("/tmp/prismguard-rpc-test/run/sample.sock"));
}

#[test]
fn sample_rpc_prepare_runtime_creates_parent_dir() {
    let base = PathBuf::from(format!("/tmp/prismguard-rpc-{}", std::process::id()));
    if base.exists() {
        std::fs::remove_dir_all(&base).expect("cleanup old temp dir");
    }

    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Unix,
        unix_socket: base.join("nested/sample.sock"),
    };

    rpc.prepare_runtime().expect("prepare runtime");
    assert!(base.join("nested").is_dir());

    std::fs::remove_dir_all(&base).expect("cleanup temp dir");
}

#[test]
fn sample_rpc_unix_socket_path_rejects_non_unix_transport() {
    let rpc = SampleRpcConfig {
        enabled: true,
        transport: SampleRpcTransport::Tcp,
        unix_socket: PathBuf::from("/tmp/ignored.sock"),
    };

    let err = rpc.unix_socket_path().expect_err("tcp transport should not expose unix socket");
    assert!(err.to_string().contains("unix"));
}

#[test]
fn sample_rpc_request_line_codec_roundtrips() {
    let request = SampleRpcRequest::LoadBalancedLatestSamples {
        profile: "default".to_string(),
        db_path: "/tmp/history.rocks".to_string(),
        max_samples: 2048,
    };

    let encoded = encode_request_line(&request).expect("encode request");
    assert_eq!(encoded.last().copied(), Some(b'\n'));

    let decoded = decode_request_line(&encoded).expect("decode request");
    assert_eq!(decoded, request);
}

#[test]
fn sample_rpc_response_line_codec_roundtrips() {
    let response = SampleRpcResponse {
        ok: true,
        result: Some(json!({
            "sample_count": 42
        })),
        error: None,
    };

    let encoded = encode_response_line(&response).expect("encode response");
    assert_eq!(encoded.last().copied(), Some(b'\n'));

    let decoded = decode_response_line(&encoded).expect("decode response");
    assert_eq!(decoded, response);
}

#[tokio::test]
async fn sample_rpc_unix_client_server_roundtrip() {
    let socket_dir = PathBuf::from(format!("/tmp/prismguard-rpc-roundtrip-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");

    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_one_unix_request(&server_socket, |request| {
            assert_eq!(
                request,
                SampleRpcRequest::GetSampleCount {
                    profile: "default".to_string(),
                    db_path: "/tmp/history.rocks".to_string(),
                }
            );
            Ok(SampleRpcResponse {
                ok: true,
                result: Some(json!({ "sample_count": 7 })),
                error: None,
            })
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let response = send_unix_request(
        &socket_path,
        &SampleRpcRequest::GetSampleCount {
            profile: "default".to_string(),
            db_path: "/tmp/history.rocks".to_string(),
        },
    )
    .await
    .expect("send unix request");

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "sample_count": 7 })),
            error: None,
        }
    );

    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[tokio::test]
async fn sample_rpc_unix_server_handles_multiple_requests() {
    let socket_dir = PathBuf::from(format!("/tmp/prismguard-rpc-loop-{}", std::process::id()));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");
    let socket_path = socket_dir.join("sample.sock");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_unix_requests_until_shutdown(
            &server_socket,
            |request| match request {
                SampleRpcRequest::GetSampleCount { .. } => SampleRpcResponse {
                    ok: true,
                    result: Some(json!({ "sample_count": 3 })),
                    error: None,
                },
                _ => SampleRpcResponse::err("unexpected"),
            },
            async move {
                let _ = shutdown_rx.await;
            },
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    for _ in 0..2 {
        let response = send_unix_request(
            &socket_path,
            &SampleRpcRequest::GetSampleCount {
                profile: "default".to_string(),
                db_path: "/tmp/history.rocks".to_string(),
            },
        )
        .await
        .expect("send unix request");

        assert_eq!(
            response,
            SampleRpcResponse {
                ok: true,
                result: Some(json!({ "sample_count": 3 })),
                error: None,
            }
        );
    }

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[test]
fn sample_rpc_dispatches_get_sample_count() {
    let response = dispatch_request(
        SampleRpcRequest::GetSampleCount {
            profile: "default".to_string(),
            db_path: "/tmp/history.rocks".to_string(),
        },
        |request| {
            assert_eq!(
                request,
                &SampleRpcRequest::GetSampleCount {
                    profile: "default".to_string(),
                    db_path: "/tmp/history.rocks".to_string(),
                }
            );
            Ok(json!({ "sample_count": 12 }))
        },
    );

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "sample_count": 12 })),
            error: None,
        }
    );
}

#[test]
fn sample_rpc_dispatches_balanced_latest_samples() {
    let response = dispatch_request(
        SampleRpcRequest::LoadBalancedLatestSamples {
            profile: "default".to_string(),
            db_path: "/tmp/history.rocks".to_string(),
            max_samples: 4,
        },
        |request| {
            assert_eq!(
                request,
                &SampleRpcRequest::LoadBalancedLatestSamples {
                    profile: "default".to_string(),
                    db_path: "/tmp/history.rocks".to_string(),
                    max_samples: 4,
                }
            );
            Ok(json!({
                "samples": [
                    {"id": 4, "text": "pass-3", "label": 0},
                    {"id": 6, "text": "violation-3", "label": 1},
                    {"id": 2, "text": "pass-2", "label": 0},
                    {"id": 5, "text": "violation-2", "label": 1}
                ]
            }))
        },
    );

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({
                "samples": [
                    {"id": 4, "text": "pass-3", "label": 0},
                    {"id": 6, "text": "violation-3", "label": 1},
                    {"id": 2, "text": "pass-2", "label": 0},
                    {"id": 5, "text": "violation-2", "label": 1}
                ]
            })),
            error: None,
        }
    );
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_get_sample_count_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-storage-{}",
        std::process::id()
    ));
    if rocks_dir.exists() {
        std::fs::remove_dir_all(&rocks_dir).expect("cleanup old rocks dir");
    }

    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, &rocks_dir).expect("open rocksdb");
    db.put(vec![0x02, b'm', b'e', b't', b'a', b':', b'c', b'o', b'u', b'n', b't'], vec![0x02, b'7'])
        .expect("put meta count");
    drop(db);

    let response = sample_rpc::dispatch_request_with_storage(SampleRpcRequest::GetSampleCount {
        profile: "default".to_string(),
        db_path: rocks_dir.display().to_string(),
    });

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "sample_count": 7 })),
            error: None,
        }
    );

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_cleanup_excess_samples_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-cleanup-{}",
        std::process::id()
    ));
    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "pass-4", "label": 0, "category": null, "created_at": "2026-04-03 10:03:00"}),
            json!({"id": 5, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:04:00"}),
            json!({"id": 6, "text": "violation-2", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:05:00"}),
            json!({"id": 7, "text": "violation-3", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:06:00"}),
            json!({"id": 8, "text": "violation-4", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:07:00"}),
        ],
    );

    let response = sample_rpc::dispatch_request_with_storage(
        SampleRpcRequest::CleanupExcessSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_items: 4,
        },
    );

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "removed": 4 })),
            error: None,
        }
    );

    let count_response = sample_rpc::dispatch_request_with_storage(SampleRpcRequest::GetSampleCount {
        profile: "default".to_string(),
        db_path: rocks_dir.display().to_string(),
    });
    assert_eq!(
        count_response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "sample_count": 4 })),
            error: None,
        }
    );

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_balanced_latest_samples_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-balanced-latest-{}",
        std::process::id()
    ));
    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:03:00"}),
            json!({"id": 5, "text": "violation-2", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:04:00"}),
            json!({"id": 6, "text": "violation-3", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:05:00"}),
        ],
    );

    let response = sample_rpc::dispatch_request_with_storage(SampleRpcRequest::LoadBalancedLatestSamples {
        profile: "default".to_string(),
        db_path: rocks_dir.display().to_string(),
        max_samples: 4,
    });

    assert!(response.ok, "unexpected response: {response:?}");
    let samples = response.result.expect("response result")["samples"]
        .as_array()
        .cloned()
        .expect("samples array");
    assert_eq!(samples.len(), 4);

    let mut ids = samples
        .iter()
        .map(|sample| sample["id"].as_u64().expect("sample id"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, vec![2, 4, 5, 6]);

    let pass_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(0))
        .count();
    let violation_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(1))
        .count();
    assert_eq!(pass_count, 2);
    assert_eq!(violation_count, 2);

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_balanced_samples_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-balanced-{}",
        std::process::id()
    ));
    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "pass-4", "label": 0, "category": null, "created_at": "2026-04-03 10:03:00"}),
            json!({"id": 5, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:04:00"}),
            json!({"id": 6, "text": "violation-2", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:05:00"}),
        ],
    );

    let response = sample_rpc::dispatch_request_with_storage(SampleRpcRequest::LoadBalancedSamples {
        profile: "default".to_string(),
        db_path: rocks_dir.display().to_string(),
        max_samples: 6,
    });

    assert!(response.ok, "unexpected response: {response:?}");
    let samples = response.result.expect("response result")["samples"]
        .as_array()
        .cloned()
        .expect("samples array");
    assert_eq!(samples.len(), 4);

    let pass_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(0))
        .count();
    let violation_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(1))
        .count();
    assert_eq!(pass_count, 2);
    assert_eq!(violation_count, 2);

    let mut ids = samples
        .iter()
        .map(|sample| sample["id"].as_u64().expect("sample id"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 4);

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_balanced_random_samples_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-balanced-random-{}",
        std::process::id()
    ));
    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "pass-4", "label": 0, "category": null, "created_at": "2026-04-03 10:03:00"}),
            json!({"id": 5, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:04:00"}),
            json!({"id": 6, "text": "violation-2", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:05:00"}),
            json!({"id": 7, "text": "violation-3", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:06:00"}),
            json!({"id": 8, "text": "violation-4", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:07:00"}),
        ],
    );

    let response =
        sample_rpc::dispatch_request_with_storage(SampleRpcRequest::LoadBalancedRandomSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_samples: 6,
        });

    assert!(response.ok, "unexpected response: {response:?}");
    let samples = response.result.expect("response result")["samples"]
        .as_array()
        .cloned()
        .expect("samples array");
    assert_eq!(samples.len(), 6);

    let pass_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(0))
        .count();
    let violation_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(1))
        .count();
    assert_eq!(pass_count, 3);
    assert_eq!(violation_count, 3);

    let mut ids = samples
        .iter()
        .map(|sample| sample["id"].as_u64().expect("sample id"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 6);

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_dispatches_balanced_random_duplicate_samples_with_real_storage() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-random-duplicate-{}",
        std::process::id()
    ));
    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:03:00"}),
        ],
    );

    let response = sample_rpc::dispatch_request_with_storage(
        SampleRpcRequest::LoadBalancedRandomDuplicateSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_samples: 6,
        },
    );

    assert!(response.ok, "unexpected response: {response:?}");
    let samples = response.result.expect("response result")["samples"]
        .as_array()
        .cloned()
        .expect("samples array");
    assert_eq!(samples.len(), 6);

    let pass_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(0))
        .count();
    let violation_count = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(1))
        .count();
    assert_eq!(pass_count, 3);
    assert_eq!(violation_count, 3);

    let violation_ids = samples
        .iter()
        .filter(|sample| sample["label"].as_i64() == Some(1))
        .map(|sample| sample["id"].as_u64().expect("sample id"))
        .collect::<Vec<_>>();
    assert_eq!(violation_ids.len(), 3);
    assert!(violation_ids.iter().all(|id| *id == 4));

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[tokio::test]
async fn sample_rpc_storage_server_serves_real_sample_count_over_unix_socket() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-server-{}",
        std::process::id()
    ));
    let socket_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-server-sock-{}",
        std::process::id()
    ));
    if rocks_dir.exists() {
        std::fs::remove_dir_all(&rocks_dir).expect("cleanup old rocks dir");
    }
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");

    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, &rocks_dir).expect("open rocksdb");
    db.put(vec![0x02, b'm', b'e', b't', b'a', b':', b'c', b'o', b'u', b'n', b't'], vec![0x02, b'9'])
        .expect("put meta count");
    drop(db);

    let socket_path = socket_dir.join("sample.sock");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_storage_sample_rpc_until_shutdown(&server_socket, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let response = send_unix_request(
        &socket_path,
        &SampleRpcRequest::GetSampleCount {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
        },
    )
    .await
    .expect("send unix request");

    assert_eq!(
        response,
        SampleRpcResponse {
            ok: true,
            result: Some(json!({ "sample_count": 9 })),
            error: None,
        }
    );

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}

#[cfg(feature = "storage-debug")]
#[tokio::test]
async fn sample_rpc_storage_server_serves_balanced_latest_samples_over_unix_socket() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-latest-server-{}",
        std::process::id()
    ));
    let socket_dir = PathBuf::from(format!(
        "/tmp/prismguard-sample-rpc-latest-server-sock-{}",
        std::process::id()
    ));
    if socket_dir.exists() {
        std::fs::remove_dir_all(&socket_dir).expect("cleanup old socket dir");
    }
    std::fs::create_dir_all(&socket_dir).expect("create socket dir");

    seed_sample_db(
        &rocks_dir,
        &[
            json!({"id": 1, "text": "pass-1", "label": 0, "category": null, "created_at": "2026-04-03 10:00:00"}),
            json!({"id": 2, "text": "pass-2", "label": 0, "category": null, "created_at": "2026-04-03 10:01:00"}),
            json!({"id": 3, "text": "violation-1", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:02:00"}),
            json!({"id": 4, "text": "pass-3", "label": 0, "category": null, "created_at": "2026-04-03 10:03:00"}),
            json!({"id": 5, "text": "violation-2", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:04:00"}),
            json!({"id": 6, "text": "violation-3", "label": 1, "category": "unsafe", "created_at": "2026-04-03 10:05:00"}),
        ],
    );

    let socket_path = socket_dir.join("sample.sock");
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server_socket = socket_path.clone();
    let server = tokio::spawn(async move {
        serve_storage_sample_rpc_until_shutdown(&server_socket, async move {
            let _ = shutdown_rx.await;
        })
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    let response = send_unix_request(
        &socket_path,
        &SampleRpcRequest::LoadBalancedLatestSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_samples: 4,
        },
    )
    .await
    .expect("send unix request");

    assert!(response.ok, "unexpected response: {response:?}");
    let samples = response.result.expect("response result")["samples"]
        .as_array()
        .cloned()
        .expect("samples array");
    assert_eq!(samples.len(), 4);

    let mut ids = samples
        .iter()
        .map(|sample| sample["id"].as_u64().expect("sample id"))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, vec![2, 4, 5, 6]);

    let _ = shutdown_tx.send(());
    server.await.expect("join server").expect("server result");
    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
    std::fs::remove_dir_all(&socket_dir).expect("cleanup socket dir");
}
