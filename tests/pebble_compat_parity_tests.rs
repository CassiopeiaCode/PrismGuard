#![allow(dead_code)]

#[cfg(feature = "storage-debug")]
#[path = "../src/storage.rs"]
mod storage;
#[path = "../src/config.rs"]
mod config;
#[cfg(feature = "storage-debug")]
#[path = "../src/sample_rpc.rs"]
mod sample_rpc;

#[cfg(feature = "storage-debug")]
use rocksdb::{DBWithThreadMode, MultiThreaded, Options};
#[cfg(feature = "storage-debug")]
use serde_json::json;
#[cfg(feature = "storage-debug")]
use std::path::PathBuf;

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
fn get_rocks_string(db: &DBWithThreadMode<MultiThreaded>, key: &str) -> Option<String> {
    db.get(encode_rocksdict_string(key))
        .expect("read rocks string")
        .map(|raw| {
            let bytes = raw.strip_prefix(&[0x02]).unwrap_or(raw.as_ref());
            String::from_utf8(bytes.to_vec()).expect("decode rocks string")
        })
}

#[cfg(feature = "storage-debug")]
fn seed_storage(path: &PathBuf, samples: &[serde_json::Value]) {
    if path.exists() {
        std::fs::remove_dir_all(path).expect("cleanup old rocks dir");
    }
    let storage = storage::SampleStorage::open_read_write(path).expect("open writeable storage");
    for sample in samples {
        storage
            .save_sample(
                sample["text"].as_str().expect("text"),
                sample["label"].as_i64().expect("label"),
                sample["category"].as_str(),
            )
            .expect("save sample");
    }
}

#[cfg(feature = "storage-debug")]
#[test]
fn storage_find_by_text_falls_back_when_pointer_is_invalid() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-storage-invalid-pointer-{}",
        std::process::id()
    ));
    seed_storage(
        &rocks_dir,
        &[
            json!({"text": "needle", "label": 0, "category": null}),
            json!({"text": "other", "label": 1, "category": "unsafe"}),
        ],
    );

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, &rocks_dir).expect("open rocksdb");
    put_rocks_string(&db, "text_latest:4bf84babe76dabda6c4da4d25354704d", "not-a-number");
    drop(db);

    let storage = storage::SampleStorage::open_read_only(&rocks_dir).expect("open storage");
    let sample = storage
        .find_by_text("needle")
        .expect("find sample")
        .expect("sample exists");
    assert_eq!(sample.id, Some(1));
    assert_eq!(sample.value["text"].as_str(), Some("needle"));

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}

#[cfg(feature = "storage-debug")]
#[test]
fn sample_rpc_cleanup_excess_samples_balances_labels_and_refreshes_pointer() {
    let rocks_dir = PathBuf::from(format!(
        "/tmp/prismguard-storage-cleanup-balance-{}",
        std::process::id()
    ));
    seed_storage(
        &rocks_dir,
        &[
            json!({"text": "b", "label": 1, "category": "unsafe"}),
            json!({"text": "b", "label": 1, "category": "unsafe"}),
            json!({"text": "b", "label": 1, "category": "unsafe"}),
            json!({"text": "pass-1", "label": 0, "category": null}),
            json!({"text": "pass-2", "label": 0, "category": null}),
        ],
    );

    let response = sample_rpc::dispatch_request_with_storage(
        sample_rpc::SampleRpcRequest::CleanupExcessSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_items: 4,
        },
    );
    assert!(response.ok, "unexpected response: {response:?}");
    assert_eq!(response.result, Some(json!({ "removed": 1 })));

    let storage = storage::SampleStorage::open_read_only(&rocks_dir).expect("open storage");
    assert_eq!(storage.sample_count(), Some(4));
    assert_eq!(storage.label_counts(), (2, 2));
    let latest = storage
        .find_by_text("b")
        .expect("find duplicate")
        .expect("duplicate exists");
    let latest_id = latest.id.expect("latest duplicate id");
    assert!(
        latest_id == 2 || latest_id == 3,
        "expected surviving latest duplicate id to be 2 or 3, got {latest_id}"
    );
    drop(storage);

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, &rocks_dir).expect("open rocksdb");
    assert_eq!(
        get_rocks_string(&db, "text_latest:92eb5ffee6ae2fec3ad71c777531578f"),
        Some(latest_id.to_string())
    );
    drop(db);

    std::fs::remove_dir_all(&rocks_dir).expect("cleanup rocks dir");
}
