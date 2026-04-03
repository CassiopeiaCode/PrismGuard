use std::cmp::Ordering;
use std::ffi::CStr;
use std::path::{Path, PathBuf};

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
    pub fn open_read_write(rocks_path: impl AsRef<Path>) -> Result<Self> {
        let rocks_path = rocks_path.as_ref().to_path_buf();
        if let Some(parent) = rocks_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create RocksDB parent dir {}", parent.display()))?;
        }

        let mut options = Options::default();
        options.create_if_missing(true);
        options.set_comparator("rocksdict", Box::new(bytewise_compare));
        let db = DBWithThreadMode::<MultiThreaded>::open(&options, &rocks_path)
            .with_context(|| format!("failed to open RocksDB {}", rocks_path.display()))?;

        let storage = Self {
            db,
            rocks_path,
            column_families: vec!["default".to_string()],
        };
        storage.ensure_metadata_defaults()?;
        Ok(storage)
    }

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

    pub fn sample_count(&self) -> Option<u64> {
        self.get_u64("meta:count")
    }

    pub fn label_counts(&self) -> (u64, u64) {
        (
            self.get_u64("meta:count:0").unwrap_or(0),
            self.get_u64("meta:count:1").unwrap_or(0),
        )
    }

    pub fn load_balanced_latest_samples(&self, max_samples: usize) -> Vec<SampleRecord> {
        if max_samples <= 0 {
            return Vec::new();
        }

        let (pass_count, violation_count) = self.label_counts();
        let target_per_label = max_samples / 2;
        if target_per_label == 0 {
            return Vec::new();
        }

        let pass_samples = self.load_samples_by_label(0, usize::min(pass_count as usize, target_per_label));
        let violation_samples =
            self.load_samples_by_label(1, usize::min(violation_count as usize, target_per_label));
        self.interleave_deterministically(pass_samples, violation_samples)
    }

    pub fn load_balanced_random_samples(&self, max_samples: usize) -> Vec<SampleRecord> {
        if max_samples <= 0 {
            return Vec::new();
        }

        let (pass_count, violation_count) = self.label_counts();
        let target_per_label = max_samples / 2;
        if target_per_label == 0 {
            return Vec::new();
        }

        let pass_take = usize::min(pass_count as usize, target_per_label);
        let violation_take = usize::min(violation_count as usize, target_per_label);

        let pass_samples = self.select_pseudo_random(self.load_samples_by_label(0, pass_count as usize), pass_take);
        let violation_samples =
            self.select_pseudo_random(self.load_samples_by_label(1, violation_count as usize), violation_take);
        self.interleave_deterministically(pass_samples, violation_samples)
    }

    pub fn load_balanced_samples(&self, max_samples: usize) -> Vec<SampleRecord> {
        if max_samples <= 0 {
            return Vec::new();
        }

        let (pass_count, violation_count) = self.label_counts();
        if pass_count == 0 || violation_count == 0 {
            return Vec::new();
        }

        let target_per_label = max_samples / 2;
        if target_per_label == 0 {
            return Vec::new();
        }

        let balanced_count = usize::min(usize::min(pass_count as usize, violation_count as usize), target_per_label);
        if balanced_count == 0 {
            return Vec::new();
        }

        let pass_samples =
            self.select_pseudo_random(self.load_samples_by_label(0, pass_count as usize), balanced_count);
        let violation_samples =
            self.select_pseudo_random(self.load_samples_by_label(1, violation_count as usize), balanced_count);
        self.interleave_deterministically(pass_samples, violation_samples)
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

    pub fn save_sample(&self, text: &str, label: i64, category: Option<&str>) -> Result<()> {
        let sample_id = self.next_id();
        let created_at = current_local_timestamp()?;
        let text_hash = format!("{:x}", md5::compute(text.as_bytes()));
        let sample = serde_json::json!({
            "id": sample_id,
            "text": text,
            "label": label,
            "category": category,
            "created_at": created_at,
        });

        self.put_string(&format!("sample:{sample_id:020}"), &sample.to_string())?;
        self.put_string(&format!("text_latest:{text_hash}"), &sample_id.to_string())?;

        let mut count_0 = self.get_u64("meta:count:0").unwrap_or(0);
        let mut count_1 = self.get_u64("meta:count:1").unwrap_or(0);
        if label == 0 {
            count_0 += 1;
        } else {
            count_1 += 1;
        }
        self.put_string("meta:count", &(count_0 + count_1).to_string())?;
        self.put_string("meta:count:0", &count_0.to_string())?;
        self.put_string("meta:count:1", &count_1.to_string())?;
        self.put_string("meta:next_id", &(sample_id + 1).to_string())?;
        Ok(())
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

    fn next_id(&self) -> u64 {
        self.get_u64("meta:next_id").unwrap_or(1)
    }

    fn ensure_metadata_defaults(&self) -> Result<()> {
        if self.get_u64("meta:next_id").is_none() {
            self.put_string("meta:next_id", "1")?;
        }
        if self.get_u64("meta:count").is_none() {
            self.put_string("meta:count", "0")?;
        }
        if self.get_u64("meta:count:0").is_none() {
            self.put_string("meta:count:0", "0")?;
        }
        if self.get_u64("meta:count:1").is_none() {
            self.put_string("meta:count:1", "0")?;
        }
        Ok(())
    }

    fn put_string(&self, key: &str, value: &str) -> Result<()> {
        self.db
            .put(encode_rocksdict_string(key), encode_rocksdict_string(value))
            .with_context(|| format!("failed to write key {key}"))
    }

    fn load_samples_by_label(&self, label: i64, limit: usize) -> Vec<SampleRecord> {
        if limit == 0 {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut sample_id = self.next_id().saturating_sub(1);
        while sample_id > 0 && out.len() < limit {
            if let Ok(Some(sample)) = self.sample_by_id(sample_id) {
                if sample.value.get("label").and_then(Value::as_i64) == Some(label) {
                    out.push(sample);
                }
            }
            sample_id -= 1;
        }
        out
    }

    fn select_pseudo_random(&self, mut samples: Vec<SampleRecord>, take: usize) -> Vec<SampleRecord> {
        if samples.len() <= take {
            return samples;
        }

        samples.sort_by(|lhs, rhs| {
            let lhs_key = pseudo_random_key(lhs);
            let rhs_key = pseudo_random_key(rhs);
            lhs_key
                .cmp(&rhs_key)
                .then_with(|| lhs.id.cmp(&rhs.id))
                .then_with(|| lhs.key.cmp(&rhs.key))
        });
        samples.truncate(take);
        samples
    }

    fn interleave_deterministically(
        &self,
        pass_samples: Vec<SampleRecord>,
        violation_samples: Vec<SampleRecord>,
    ) -> Vec<SampleRecord> {
        let mut combined = Vec::with_capacity(pass_samples.len() + violation_samples.len());
        let mut pass_iter = pass_samples.into_iter();
        let mut violation_iter = violation_samples.into_iter();

        loop {
            let mut pushed = false;
            if let Some(sample) = violation_iter.next() {
                combined.push(sample);
                pushed = true;
            }
            if let Some(sample) = pass_iter.next() {
                combined.push(sample);
                pushed = true;
            }
            if !pushed {
                break;
            }
        }

        combined
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

fn pseudo_random_key(sample: &SampleRecord) -> [u8; 16] {
    let mut seed = sample.id.unwrap_or(0).to_string();
    seed.push(':');
    seed.push_str(sample.value.get("text").and_then(Value::as_str).unwrap_or(""));
    md5::compute(seed.as_bytes()).0
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

fn current_local_timestamp() -> Result<String> {
    let mut now = libc::time_t::default();
    unsafe {
        libc::time(&mut now as *mut libc::time_t);
    }

    let mut local_tm = libc::tm {
        tm_sec: 0,
        tm_min: 0,
        tm_hour: 0,
        tm_mday: 0,
        tm_mon: 0,
        tm_year: 0,
        tm_wday: 0,
        tm_yday: 0,
        tm_isdst: 0,
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "emscripten",
            target_os = "freebsd",
            target_os = "fuchsia",
            target_os = "haiku",
            target_os = "illumos",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "solaris"
        ))]
        tm_gmtoff: 0,
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "emscripten",
            target_os = "freebsd",
            target_os = "fuchsia",
            target_os = "haiku",
            target_os = "illumos",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "solaris"
        ))]
        tm_zone: std::ptr::null(),
    };
    let local_ptr = unsafe { libc::localtime_r(&now, &mut local_tm as *mut libc::tm) };
    if local_ptr.is_null() {
        return Err(anyhow::anyhow!("failed to compute local timestamp"));
    }

    let mut buffer = [0i8; 20];
    let written = unsafe {
        const FORMAT: &[u8] = b"%Y-%m-%d %H:%M:%S\0";
        libc::strftime(
            buffer.as_mut_ptr(),
            buffer.len(),
            FORMAT.as_ptr().cast(),
            &local_tm as *const libc::tm,
        )
    };
    if written == 0 {
        return Err(anyhow::anyhow!("failed to format local timestamp"));
    }

    Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}
