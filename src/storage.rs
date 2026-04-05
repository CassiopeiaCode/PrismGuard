use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rocksdb::{DBWithThreadMode, IteratorMode, MultiThreaded, Options, WriteBatch};
use rusqlite::Connection;
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
        migrate_legacy_sqlite_if_needed(&rocks_path)?;
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
        migrate_legacy_sqlite_if_needed(&rocks_path)?;
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
        self.shuffle_combined(pass_samples, violation_samples)
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

        let pass_samples = self.select_random(self.load_samples_by_label(0, pass_count as usize), pass_take);
        let violation_samples =
            self.select_random(self.load_samples_by_label(1, violation_count as usize), violation_take);
        self.shuffle_combined(pass_samples, violation_samples)
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
            self.select_random(self.load_samples_by_label(0, pass_count as usize), balanced_count);
        let violation_samples =
            self.select_random(self.load_samples_by_label(1, violation_count as usize), balanced_count);
        self.shuffle_combined(pass_samples, violation_samples)
    }

    pub fn sample_by_id(&self, id: u64) -> Result<Option<SampleRecord>> {
        let key = format!("sample:{id:020}");
        self.sample_by_key(&key)
    }

    pub fn find_by_text(&self, text: &str) -> Result<Option<SampleRecord>> {
        let text_hash = format!("{:x}", md5::compute(text.as_bytes()));
        let pointer_key = format!("text_latest:{text_hash}");
        let Some(pointer_raw) = self
            .db
            .get(encode_rocksdict_string(&pointer_key))
            .with_context(|| format!("failed to read key {pointer_key}"))?
        else {
            return Ok(None);
        };

        if let Ok(sample_id) = decode_rocksdict_string(&pointer_raw)?.parse::<u64>() {
            if let Some(sample) = self.sample_by_id(sample_id)? {
                if sample.value.get("text").and_then(Value::as_str) == Some(text) {
                    return Ok(Some(sample));
                }
            }
        }

        let mut candidate_id = self.next_id().saturating_sub(1);
        while candidate_id > 0 {
            if let Some(sample) = self.sample_by_id(candidate_id)? {
                if sample.value.get("text").and_then(Value::as_str) == Some(text) {
                    return Ok(Some(sample));
                }
            }
            candidate_id -= 1;
        }

        Ok(None)
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
            "text_hash": text_hash,
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

    pub fn cleanup_excess_samples(&self, max_items: usize) -> Result<usize> {
        let total = self.sample_count().unwrap_or_default() as usize;
        if total <= max_items {
            return Ok(0);
        }

        let (pass_count, violation_count) = self.label_counts();
        let target_per_label = max_items / 2;

        let mut removed = 0usize;
        if pass_count as usize > target_per_label {
            removed += self.delete_samples_for_label(0, pass_count as usize - target_per_label)?;
        }
        if violation_count as usize > target_per_label {
            removed += self.delete_samples_for_label(1, violation_count as usize - target_per_label)?;
        }

        Ok(removed)
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

    fn delete_samples_for_label(&self, label: i64, count: usize) -> Result<usize> {
        if count == 0 {
            return Ok(0);
        }

        let available = self.load_samples_by_label(label, usize::MAX);
        if available.is_empty() {
            return Ok(0);
        }

        let to_delete = self.select_random(available, count);
        if to_delete.is_empty() {
            return Ok(0);
        }

        let mut batch = WriteBatch::default();
        let mut deleted_ids = HashSet::with_capacity(to_delete.len());
        let mut affected_hashes = HashSet::new();
        let mut removed = 0usize;

        for sample in &to_delete {
            let Some(sample_id) = sample.id else {
                continue;
            };
            deleted_ids.insert(sample_id);
            batch.delete(encode_rocksdict_string(&sample.key));
            removed += 1;

            if let Some(text_hash) = sample_text_hash(sample) {
                let pointer_key = format!("text_latest:{text_hash}");
                if self.get_u64(&pointer_key) == Some(sample_id) {
                    affected_hashes.insert(text_hash);
                }
            }
        }

        let total_count = self.get_u64("meta:count").unwrap_or_default();
        let count_0 = self.get_u64("meta:count:0").unwrap_or_default();
        let count_1 = self.get_u64("meta:count:1").unwrap_or_default();

        let next_total = total_count.saturating_sub(removed as u64);
        let next_count_0 = if label == 0 {
            count_0.saturating_sub(removed as u64)
        } else {
            count_0
        };
        let next_count_1 = if label == 1 {
            count_1.saturating_sub(removed as u64)
        } else {
            count_1
        };

        batch.put(
            encode_rocksdict_string("meta:count"),
            encode_rocksdict_string(&next_total.to_string()),
        );
        batch.put(
            encode_rocksdict_string("meta:count:0"),
            encode_rocksdict_string(&next_count_0.to_string()),
        );
        batch.put(
            encode_rocksdict_string("meta:count:1"),
            encode_rocksdict_string(&next_count_1.to_string()),
        );

        let refreshed = self.refresh_text_latest_after_deletes(&deleted_ids, &affected_hashes);
        for (text_hash, sample_id) in refreshed {
            let pointer_key = format!("text_latest:{text_hash}");
            match sample_id {
                Some(sample_id) => batch.put(
                    encode_rocksdict_string(&pointer_key),
                    encode_rocksdict_string(&sample_id.to_string()),
                ),
                None => batch.delete(encode_rocksdict_string(&pointer_key)),
            }
        }

        self.db
            .write(batch)
            .context("failed to write cleanup batch to RocksDB")?;
        Ok(removed)
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

    fn select_random(&self, mut samples: Vec<SampleRecord>, take: usize) -> Vec<SampleRecord> {
        if samples.len() <= take {
            return samples;
        }

        shuffle_in_place(&mut samples);
        samples.truncate(take);
        samples
    }

    fn shuffle_combined(
        &self,
        pass_samples: Vec<SampleRecord>,
        violation_samples: Vec<SampleRecord>,
    ) -> Vec<SampleRecord> {
        let mut combined = pass_samples;
        combined.extend(violation_samples);
        shuffle_in_place(&mut combined);
        combined
    }

    fn refresh_text_latest_after_deletes(
        &self,
        deleted_ids: &HashSet<u64>,
        affected_hashes: &HashSet<String>,
    ) -> HashMap<String, Option<u64>> {
        let mut resolved = HashMap::new();
        if affected_hashes.is_empty() {
            return resolved;
        }

        let mut remaining = affected_hashes.clone();
        let mut sample_id = self.next_id().saturating_sub(1);
        while sample_id > 0 && !remaining.is_empty() {
            if deleted_ids.contains(&sample_id) {
                sample_id -= 1;
                continue;
            }

            if let Ok(Some(sample)) = self.sample_by_id(sample_id) {
                if let Some(text_hash) = sample_text_hash(&sample) {
                    if remaining.remove(&text_hash) {
                        resolved.insert(text_hash, Some(sample_id));
                    }
                }
            }

            sample_id -= 1;
        }

        for text_hash in remaining {
            resolved.insert(text_hash, None);
        }

        resolved
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

fn migrate_legacy_sqlite_if_needed(rocks_path: &Path) -> Result<()> {
    if rocks_path.exists() {
        return Ok(());
    }

    let sqlite_path = legacy_sqlite_path(rocks_path);
    if !sqlite_path.exists() {
        return Ok(());
    }

    if let Some(parent) = rocks_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create RocksDB parent dir {}", parent.display()))?;
    }

    let temp_rocks = rocks_path.with_extension("rocks.migrating");
    if temp_rocks.exists() {
        std::fs::remove_dir_all(&temp_rocks).with_context(|| {
            format!(
                "failed to remove stale temporary RocksDB dir {}",
                temp_rocks.display()
            )
        })?;
    }

    let connection = Connection::open(&sqlite_path)
        .with_context(|| format!("failed to open legacy SQLite {}", sqlite_path.display()))?;
    let mut statement = connection.prepare(
        "SELECT id, text, label, category, created_at FROM samples ORDER BY id ASC",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(LegacySqliteSample {
            id: row.get::<_, i64>(0)?,
            text: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            label: row.get::<_, Option<i64>>(2)?.unwrap_or_default(),
            category: row.get::<_, Option<String>>(3)?,
            created_at: row.get::<_, Option<String>>(4)?,
        })
    })?;

    let mut options = Options::default();
    options.create_if_missing(true);
    options.set_comparator("rocksdict", Box::new(bytewise_compare));
    let temp_db = DBWithThreadMode::<MultiThreaded>::open(&options, &temp_rocks)
        .with_context(|| format!("failed to create temporary RocksDB {}", temp_rocks.display()))?;

    let mut batch = WriteBatch::default();
    let mut next_id = 1u64;
    let mut count_0 = 0u64;
    let mut count_1 = 0u64;
    for row in rows {
        let sample = row.with_context(|| {
            format!(
                "failed to read row from legacy SQLite {}",
                sqlite_path.display()
            )
        })?;
        let id = u64::try_from(sample.id).context("legacy SQLite sample id must be non-negative")?;
        let text_hash = format!("{:x}", md5::compute(sample.text.as_bytes()));
        let created_at = sample
            .created_at
            .unwrap_or_else(|| "1970-01-01 00:00:00".to_string());
        let payload = serde_json::json!({
            "id": id,
            "text": sample.text,
            "label": sample.label,
            "category": sample.category,
            "created_at": created_at,
            "text_hash": text_hash,
        })
        .to_string();
        batch.put(
            encode_rocksdict_string(&format!("sample:{id:020}")),
            encode_rocksdict_string(&payload),
        );
        batch.put(
            encode_rocksdict_string(&format!("text_latest:{text_hash}")),
            encode_rocksdict_string(&id.to_string()),
        );
        if sample.label == 0 {
            count_0 += 1;
        } else {
            count_1 += 1;
        }
        next_id = next_id.max(id.saturating_add(1));
    }

    batch.put(
        encode_rocksdict_string("meta:next_id"),
        encode_rocksdict_string(&next_id.to_string()),
    );
    batch.put(
        encode_rocksdict_string("meta:count"),
        encode_rocksdict_string(&(count_0 + count_1).to_string()),
    );
    batch.put(
        encode_rocksdict_string("meta:count:0"),
        encode_rocksdict_string(&count_0.to_string()),
    );
    batch.put(
        encode_rocksdict_string("meta:count:1"),
        encode_rocksdict_string(&count_1.to_string()),
    );
    temp_db
        .write(batch)
        .with_context(|| format!("failed to populate temporary RocksDB {}", temp_rocks.display()))?;
    drop(temp_db);

    std::fs::rename(&temp_rocks, rocks_path).with_context(|| {
        format!(
            "failed to move temporary RocksDB {} to {}",
            temp_rocks.display(),
            rocks_path.display()
        )
    })?;

    let sqlite_backup = sqlite_path.with_extension("db.bak");
    let _ = std::fs::rename(&sqlite_path, &sqlite_backup);
    let _ = std::fs::remove_file(sqlite_path.with_file_name(format!(
        "{}-shm",
        sqlite_path.file_name().and_then(|name| name.to_str()).unwrap_or("history.db")
    )));
    let _ = std::fs::remove_file(sqlite_path.with_file_name(format!(
        "{}-wal",
        sqlite_path.file_name().and_then(|name| name.to_str()).unwrap_or("history.db")
    )));

    Ok(())
}

fn legacy_sqlite_path(rocks_path: &Path) -> PathBuf {
    match rocks_path.file_name().and_then(|name| name.to_str()) {
        Some(name) if name.ends_with(".rocks") => rocks_path.with_file_name(format!(
            "{}.db",
            name.trim_end_matches(".rocks")
        )),
        Some(name) => rocks_path.with_file_name(format!("{name}.db")),
        None => rocks_path.with_extension("db"),
    }
}

struct LegacySqliteSample {
    id: i64,
    text: String,
    label: i64,
    category: Option<String>,
    created_at: Option<String>,
}

fn bytewise_compare(lhs: &[u8], rhs: &[u8]) -> Ordering {
    lhs.cmp(rhs)
}

fn shuffle_in_place(samples: &mut [SampleRecord]) {
    if samples.len() <= 1 {
        return;
    }
    let mut rng = XorShift64::new(initial_shuffle_seed());
    for idx in (1..samples.len()).rev() {
        let swap_idx = (rng.next_u64() as usize) % (idx + 1);
        samples.swap(idx, swap_idx);
    }
}

fn initial_shuffle_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
        ^ 0xA076_1D64_78BD_642F
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut next = self.state;
        next ^= next >> 12;
        next ^= next << 25;
        next ^= next >> 27;
        next = next.wrapping_mul(0x2545_F491_4F6C_DD1D);
        self.state = next;
        next
    }
}

fn sample_text_hash(sample: &SampleRecord) -> Option<String> {
    sample
        .value
        .get("text_hash")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            sample
                .value
                .get("text")
                .and_then(Value::as_str)
                .map(|text| format!("{:x}", md5::compute(text.as_bytes())))
        })
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
