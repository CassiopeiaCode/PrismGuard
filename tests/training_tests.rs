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
    build_training_sample_request, evaluate_training_need, fetch_training_samples_via_rpc,
    fetch_training_samples_via_unix_socket, train_hashlinear_runtime, TrainingSample,
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
    assert_eq!(metadata["n_features"], 64);
    assert_eq!(metadata["source_model"], json!(profile.hashlinear_model_path()));

    let coef_bytes = std::fs::read(&runtime_coef).expect("read runtime coef");
    assert_eq!(coef_bytes.len(), 64 * std::mem::size_of::<f32>());
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
