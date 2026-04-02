use std::path::{Path, PathBuf};
use std::cmp::Ordering;

use anyhow::{Context, Result};
use rocksdb::{DBWithThreadMode, IteratorMode, MultiThreaded, Options};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct StorageMeta {
    pub rocks_path: String,
    pub column_families: Vec<String>,
    pub raw_preview_keys: Vec<String>,
    pub next_id: Option<u64>,
    pub count: Option<u64>,
    pub count_0: Option<u64>,
    pub count_1: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleRecord {
    pub id: Option<u64>,
    pub key: String,
    pub value: Value,
}

pub struct SampleStorage {
    db: DBWithThreadMode<MultiThreaded>,
    rocks_path: PathBuf,
    column_families: Vec<String>,
}

impl SampleStorage {
    pub fn open_read_only(rocks_path: impl AsRef<Path>) -> Result<Self> {
        let rocks_path = rocks_path.as_ref().to_path_buf();
        let mut options = Options::default();
        options.create_if_missing(false);
        options.set_comparator("rocksdict", Box::new(bytewise_compare));
        let column_families = DBWithThreadMode::<MultiThreaded>::list_cf(&options, &rocks_path)
            .unwrap_or_else(|_| vec!["default".to_string()]);
        let db = if column_families.is_empty() || column_families == ["default"] {
            DBWithThreadMode::<MultiThreaded>::open_for_read_only(&options, &rocks_path, false)
        } else {
            DBWithThreadMode::<MultiThreaded>::open_cf_for_read_only(
                &options,
                &rocks_path,
                column_families.iter(),
                false,
            )
        }
        .with_context(|| format!("failed to open RocksDB {}", rocks_path.display()))?;

        Ok(Self {
            db,
            rocks_path,
            column_families,
        })
    }

    pub fn metadata(&self) -> StorageMeta {
        StorageMeta {
            rocks_path: self.rocks_path.display().to_string(),
            column_families: self.column_families.clone(),
            raw_preview_keys: self.raw_keys(8),
            next_id: self.get_u64("meta:next_id"),
            count: self.get_u64("meta:count"),
            count_0: self.get_u64("meta:count:0"),
            count_1: self.get_u64("meta:count:1"),
        }
    }

    pub fn sample_by_id(&self, id: u64) -> Result<Option<SampleRecord>> {
        let key = format!("sample:{id:020}");
        self.sample_by_key(&key)
    }

    pub fn find_by_text(&self, text: &str) -> Result<Option<SampleRecord>> {
        let text_hash = format!("{:x}", md5::compute(text.as_bytes()));
        let pointer_key = format!("text_latest:{text_hash}");
        let Some(sample_id) = self.get_u64(&pointer_key) else {
            return Ok(None);
        };
        self.sample_by_id(sample_id)
    }

    pub fn first_samples(&self, limit: usize) -> Vec<SampleRecord> {
        let mut out = Vec::new();
        for entry in self.db.iterator(IteratorMode::Start) {
            let Ok((key, value)) = entry else {
                continue;
            };
            let key = String::from_utf8_lossy(&key).to_string();
            if !key.starts_with("sample:") {
                continue;
            }

            if let Ok(json) = serde_json::from_slice::<Value>(&value) {
                out.push(SampleRecord {
                    id: json.get("id").and_then(Value::as_u64),
                    key,
                    value: json,
                });
            }

            if out.len() >= limit {
                break;
            }
        }
        out
    }

    fn sample_by_key(&self, key: &str) -> Result<Option<SampleRecord>> {
        let Some(raw) = self
            .db
            .get(encode_rocksdict_string(key))
            .with_context(|| format!("failed to read key {key}"))?
        else {
            return Ok(None);
        };

        let value: Value = serde_json::from_str(&decode_rocksdict_string(&raw)?)
            .with_context(|| format!("failed to decode sample {key} as JSON"))?;
        Ok(Some(SampleRecord {
            id: value.get("id").and_then(Value::as_u64),
            key: key.to_string(),
            value,
        }))
    }

    fn get_u64(&self, key: &str) -> Option<u64> {
        self.db
            .get(encode_rocksdict_string(key))
            .ok()
            .flatten()
            .and_then(|raw| decode_rocksdict_string(&raw).ok())
            .and_then(|s| s.parse::<u64>().ok())
    }

    fn raw_keys(&self, limit: usize) -> Vec<String> {
        let mut out = Vec::new();
        for entry in self.db.iterator(IteratorMode::Start) {
            let Ok((key, _)) = entry else {
                continue;
            };
            out.push(decode_rocksdict_string_lossy(&key));
            if out.len() >= limit {
                break;
            }
        }
        out
    }
}

fn bytewise_compare(lhs: &[u8], rhs: &[u8]) -> Ordering {
    lhs.cmp(rhs)
}

fn encode_rocksdict_string(value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + 1);
    out.push(0x02);
    out.extend_from_slice(value.as_bytes());
    out
}

fn decode_rocksdict_string(raw: &[u8]) -> Result<String> {
    let bytes = strip_rocksdict_prefix(raw);
    String::from_utf8(bytes.to_vec()).context("invalid UTF-8 in rocksdict string")
}

fn decode_rocksdict_string_lossy(raw: &[u8]) -> String {
    let bytes = strip_rocksdict_prefix(raw);
    String::from_utf8_lossy(bytes).to_string()
}

fn strip_rocksdict_prefix(raw: &[u8]) -> &[u8] {
    if raw.first() == Some(&0x02) {
        &raw[1..]
    } else {
        raw
    }
}
