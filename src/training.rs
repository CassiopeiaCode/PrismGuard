use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::profile::ModerationProfile;
use crate::sample_rpc::{send_unix_request, SampleRpcConfig, SampleRpcRequest};

#[derive(Debug, Clone, Serialize)]
pub struct TrainingDecision {
    pub should_train: bool,
    pub reason: &'static str,
    pub model_path: PathBuf,
    pub sample_count: usize,
    pub min_samples: usize,
    pub retrain_interval_minutes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TrainingSample {
    #[serde(default)]
    pub id: Option<u64>,
    pub text: String,
    pub label: i32,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HashlinearTrainingOutput {
    pub sample_count: usize,
    pub pass_count: usize,
    pub violation_count: usize,
    pub runtime_json_path: PathBuf,
    pub runtime_coef_path: PathBuf,
    pub model_marker_path: PathBuf,
}

#[derive(Debug, Clone)]
struct HashlinearTrainingSpec {
    analyzer: String,
    ngram_range: Vec<usize>,
    n_features: usize,
    alternate_sign: bool,
    norm: Option<String>,
    epochs: usize,
    batch_size: usize,
    alpha: f64,
    random_seed: u64,
    max_seconds: usize,
    use_jieba: bool,
}

pub fn evaluate_training_need(
    profile: &ModerationProfile,
    sample_count: usize,
    model_mtime: Option<SystemTime>,
    now: SystemTime,
) -> Result<TrainingDecision> {
    let (model_path, min_samples, retrain_interval_minutes) = training_config(profile)?;

    if sample_count < min_samples {
        return Ok(TrainingDecision {
            should_train: false,
            reason: "insufficient_samples",
            model_path,
            sample_count,
            min_samples,
            retrain_interval_minutes,
        });
    }

    let Some(model_mtime) = model_mtime else {
        return Ok(TrainingDecision {
            should_train: true,
            reason: "model_missing",
            model_path,
            sample_count,
            min_samples,
            retrain_interval_minutes,
        });
    };

    let elapsed = now
        .duration_since(model_mtime)
        .unwrap_or_else(|_| Duration::ZERO);
    let interval = Duration::from_secs((retrain_interval_minutes as u64) * 60);
    let (should_train, reason) = if elapsed > interval {
        (true, "retrain_interval_elapsed")
    } else {
        (false, "retrain_interval_not_elapsed")
    };

    Ok(TrainingDecision {
        should_train,
        reason,
        model_path,
        sample_count,
        min_samples,
        retrain_interval_minutes,
    })
}

pub fn build_training_sample_request(profile: &ModerationProfile) -> Result<SampleRpcRequest> {
    let (sample_loading, max_samples) = sample_loading_config(profile)?;
    let profile_name = profile.profile_name.clone();
    let db_path = profile.history_rocks_path_string();

    let request = match sample_loading {
        "latest_full" => SampleRpcRequest::LoadBalancedLatestSamples {
            profile: profile_name,
            db_path,
            max_samples,
        },
        "random_full" => SampleRpcRequest::LoadBalancedRandomSamples {
            profile: profile_name,
            db_path,
            max_samples,
        },
        _ => SampleRpcRequest::LoadBalancedSamples {
            profile: profile_name,
            db_path,
            max_samples,
        },
    };
    Ok(request)
}

pub async fn fetch_training_samples_via_rpc(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<Vec<TrainingSample>> {
    fetch_training_samples_via_unix_socket(rpc.unix_socket_path()?, profile).await
}

pub async fn fetch_training_sample_count_via_rpc(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<usize> {
    let request = SampleRpcRequest::GetSampleCount {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path_string(),
    };
    let response = send_unix_request(rpc.unix_socket_path()?, &request).await?;
    if !response.ok {
        return Err(anyhow!(
            "sample RPC request failed: {}",
            response.error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    parse_usize_result(
        response
            .result
            .as_ref()
            .ok_or_else(|| anyhow!("sample RPC response missing result payload"))?,
        "sample_count",
    )
}

pub async fn cleanup_training_samples_via_rpc(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<usize> {
    let request = SampleRpcRequest::CleanupExcessSamples {
        profile: profile.profile_name.clone(),
        db_path: profile.history_rocks_path_string(),
        max_items: training_max_db_items(profile)?,
    };
    let response = send_unix_request(rpc.unix_socket_path()?, &request).await?;
    if !response.ok {
        return Err(anyhow!(
            "sample RPC request failed: {}",
            response.error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    parse_usize_result(
        response
            .result
            .as_ref()
            .ok_or_else(|| anyhow!("sample RPC response missing result payload"))?,
        "removed",
    )
}

pub async fn fetch_training_samples_via_unix_socket(
    socket_path: &Path,
    profile: &ModerationProfile,
) -> Result<Vec<TrainingSample>> {
    let request = build_training_sample_request(profile)?;
    let response = send_unix_request(socket_path, &request).await?;
    if !response.ok {
        return Err(anyhow!(
            "sample RPC request failed: {}",
            response.error.unwrap_or_else(|| "unknown error".to_string())
        ));
    }

    parse_training_samples(
        response
            .result
            .as_ref()
            .ok_or_else(|| anyhow!("sample RPC response missing result payload"))?,
    )
}

pub fn write_training_status(profile: &ModerationProfile, status: &Value) -> Result<()> {
    let status_path = profile.training_status_path();
    let tmp_path = status_path.with_extension("json.tmp");
    let payload = serde_json::to_vec_pretty(status).context("failed to encode training status")?;
    fs::write(&tmp_path, payload)
        .with_context(|| format!("failed to write training status tmp {}", tmp_path.display()))?;
    fs::rename(&tmp_path, &status_path).with_context(|| {
        format!(
            "failed to replace training status {} with {}",
            status_path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

pub async fn run_profile_training(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<HashlinearTrainingOutput> {
    let _ = cleanup_training_samples_via_rpc(rpc, profile).await?;
    let samples = fetch_training_samples_via_rpc(rpc, profile).await?;
    train_hashlinear_runtime(profile, &samples)
}

pub fn train_hashlinear_runtime(
    profile: &ModerationProfile,
    samples: &[TrainingSample],
) -> Result<HashlinearTrainingOutput> {
    if profile.config.local_model_type != "hashlinear" {
        return Err(anyhow!(
            "train_hashlinear_runtime requires local_model_type=hashlinear, got {}",
            profile.config.local_model_type
        ));
    }
    if samples.is_empty() {
        return Err(anyhow!("cannot train hashlinear runtime with empty samples"));
    }

    let pass_count = samples.iter().filter(|sample| sample.label == 0).count();
    let violation_count = samples.iter().filter(|sample| sample.label == 1).count();
    if pass_count == 0 || violation_count == 0 {
        return Err(anyhow!(
            "hashlinear runtime training requires both labels, got pass={} violation={}",
            pass_count,
            violation_count
        ));
    }

    let spec = hashlinear_training_spec(profile);
    let (mut weights, mut intercept) = initialize_hashlinear_model(&spec, samples);
    let mut order = (0..samples.len()).collect::<Vec<_>>();
    let mut rng = XorShift64::new(spec.random_seed.max(1));
    let start = Instant::now();

    for _epoch in 0..spec.epochs.max(1) {
        shuffle_indices(&mut order, &mut rng);
        for chunk in order.chunks(spec.batch_size.max(1)) {
            if start.elapsed() >= Duration::from_secs(spec.max_seconds as u64) {
                break;
            }
            for &sample_index in chunk {
                train_one_hashlinear_sample(
                    &samples[sample_index],
                    &spec,
                    &mut weights,
                    &mut intercept,
                )?;
            }
        }

        if start.elapsed() >= Duration::from_secs(spec.max_seconds as u64) {
            break;
        }
    }

    write_hashlinear_runtime(profile, &spec, intercept as f32, &weights)?;

    let runtime_json_path = profile.hashlinear_runtime_json_path();
    let runtime_coef_path = profile.hashlinear_runtime_coef_path();
    let model_marker_path = profile.hashlinear_model_path();

    Ok(HashlinearTrainingOutput {
        sample_count: samples.len(),
        pass_count,
        violation_count,
        runtime_json_path,
        runtime_coef_path,
        model_marker_path,
    })
}

fn sample_loading_config(profile: &ModerationProfile) -> Result<(&str, usize)> {
    match profile.config.local_model_type.as_str() {
        "fasttext" => Ok((
            profile.config.fasttext_training.sample_loading.as_str(),
            profile.config.fasttext_training.max_samples,
        )),
        "hashlinear" => Ok((
            profile.config.hashlinear_training.sample_loading.as_str(),
            profile.config.hashlinear_training.max_samples,
        )),
        "bow" => Ok((
            profile.config.bow_training.sample_loading.as_str(),
            profile.config.bow_training.max_samples,
        )),
        other => Err(anyhow!("unsupported local_model_type: {other}")),
    }
}

fn training_config(profile: &ModerationProfile) -> Result<(PathBuf, usize, usize)> {
    match profile.config.local_model_type.as_str() {
        "fasttext" => Ok((
            profile.fasttext_model_path(),
            profile.config.fasttext_training.min_samples,
            profile.config.fasttext_training.retrain_interval_minutes,
        )),
        "hashlinear" => Ok((
            profile.hashlinear_model_path(),
            profile.config.hashlinear_training.min_samples,
            profile.config.hashlinear_training.retrain_interval_minutes,
        )),
        "bow" => Ok((
            profile.bow_model_path(),
            profile.config.bow_training.min_samples,
            profile.config.bow_training.retrain_interval_minutes,
        )),
        other => Err(anyhow!("unsupported local_model_type: {other}")),
    }
}

fn parse_training_samples(result: &Value) -> Result<Vec<TrainingSample>> {
    let samples = result
        .get("samples")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    samples
        .into_iter()
        .map(|sample| serde_json::from_value(sample).context("failed to decode training sample"))
        .collect()
}

fn parse_usize_result(result: &Value, field: &str) -> Result<usize> {
    let value = result
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("sample RPC response missing numeric {field}"))?;
    usize::try_from(value).context("sample RPC numeric result overflowed usize")
}

fn training_max_db_items(profile: &ModerationProfile) -> Result<usize> {
    match profile.config.local_model_type.as_str() {
        "fasttext" => Ok(profile.config.fasttext_training.max_db_items),
        "hashlinear" => Ok(profile.config.hashlinear_training.max_db_items),
        "bow" => Ok(profile.config.bow_training.max_db_items),
        other => Err(anyhow!("unsupported local_model_type: {other}")),
    }
}

fn hashlinear_training_spec(profile: &ModerationProfile) -> HashlinearTrainingSpec {
    let cfg = &profile.config.hashlinear_training;
    HashlinearTrainingSpec {
        analyzer: cfg.analyzer.clone(),
        ngram_range: cfg.ngram_range.clone(),
        n_features: cfg.n_features.max(1),
        alternate_sign: cfg.alternate_sign,
        norm: cfg.norm.clone(),
        epochs: cfg.epochs.max(1),
        batch_size: cfg.batch_size.max(1),
        alpha: cfg.alpha.max(0.0),
        random_seed: cfg.random_seed,
        max_seconds: cfg.max_seconds.max(1),
        use_jieba: cfg.use_jieba,
    }
}

fn initialize_hashlinear_model(
    spec: &HashlinearTrainingSpec,
    samples: &[TrainingSample],
) -> (Vec<f32>, f64) {
    let pass_count = samples.iter().filter(|sample| sample.label == 0).count() as f64;
    let violation_count = samples.iter().filter(|sample| sample.label == 1).count() as f64;
    let prior = ((violation_count + 1.0) / (pass_count + 1.0)).ln();
    (vec![0.0; spec.n_features], prior)
}

fn train_one_hashlinear_sample(
    sample: &TrainingSample,
    spec: &HashlinearTrainingSpec,
    weights: &mut [f32],
    intercept: &mut f64,
) -> Result<()> {
    let features = extract_features(&sample.text, spec)?;
    let mut score = *intercept;
    for (idx, value) in &features {
        score += f64::from(weights[*idx]) * *value;
    }

    let probability = sigmoid(score);
    let label = if sample.label == 0 { 0.0 } else { 1.0 };
    let error = label - probability;
    let learning_rate = 0.35;

    *intercept += learning_rate * error;
    for (idx, value) in features {
        let current = f64::from(weights[idx]);
        let gradient = error * value - spec.alpha * current;
        weights[idx] = (current + learning_rate * gradient) as f32;
    }

    Ok(())
}

fn write_hashlinear_runtime(
    profile: &ModerationProfile,
    spec: &HashlinearTrainingSpec,
    intercept: f32,
    weights: &[f32],
) -> Result<()> {
    fs::create_dir_all(&profile.base_dir)
        .with_context(|| format!("failed to create {}", profile.base_dir.display()))?;

    let runtime_json_path = profile.hashlinear_runtime_json_path();
    let runtime_coef_path = profile.hashlinear_runtime_coef_path();
    let model_marker_path = profile.hashlinear_model_path();

    let runtime_json_tmp = runtime_json_path.with_extension("json.tmp");
    let runtime_coef_tmp = runtime_coef_path.with_extension("f32.tmp");
    let model_marker_tmp = model_marker_path.with_extension("pkl.tmp");

    let metadata = serde_json::json!({
        "runtime_version": 1,
        "source_model": model_marker_path,
        "n_features": spec.n_features,
        "intercept": intercept,
        "classes": [0, 1],
        "cfg": {
            "analyzer": spec.analyzer,
            "ngram_range": spec.ngram_range,
            "n_features": spec.n_features,
            "alternate_sign": spec.alternate_sign,
            "norm": spec.norm,
            "lowercase": true,
            "use_jieba": spec.use_jieba
        }
    });
    fs::write(&runtime_json_tmp, metadata.to_string())
        .with_context(|| format!("failed to write {}", runtime_json_tmp.display()))?;

    let mut coef_bytes = Vec::with_capacity(weights.len() * std::mem::size_of::<f32>());
    for weight in weights {
        coef_bytes.extend_from_slice(&weight.to_le_bytes());
    }
    fs::write(&runtime_coef_tmp, coef_bytes)
        .with_context(|| format!("failed to write {}", runtime_coef_tmp.display()))?;

    let marker = serde_json::json!({
        "kind": "hashlinear_runtime_marker",
        "runtime_json": runtime_json_path,
        "runtime_coef": runtime_coef_path,
        "trained_at": SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_else(|_| Duration::ZERO)
            .as_secs()
    });
    fs::write(&model_marker_tmp, marker.to_string())
        .with_context(|| format!("failed to write {}", model_marker_tmp.display()))?;

    fs::rename(&runtime_json_tmp, &runtime_json_path)
        .with_context(|| format!("failed to rename {}", runtime_json_path.display()))?;
    fs::rename(&runtime_coef_tmp, &runtime_coef_path)
        .with_context(|| format!("failed to rename {}", runtime_coef_path.display()))?;
    fs::rename(&model_marker_tmp, &model_marker_path)
        .with_context(|| format!("failed to rename {}", model_marker_path.display()))?;
    Ok(())
}

fn extract_features(text: &str, spec: &HashlinearTrainingSpec) -> Result<HashMap<usize, f64>> {
    let lowered = text.to_lowercase();
    let cleaned = lowered.replace('\r', " ").replace('\n', " ");
    let grams = match spec.analyzer.as_str() {
        "char" => iter_char_ngrams(&normalize_spaces(&cleaned), ngram_range(&spec.ngram_range)?),
        "word" => iter_word_ngrams(&cleaned, ngram_range(&spec.ngram_range)?),
        other => return Err(anyhow!("unsupported hashlinear analyzer: {other}")),
    };

    let mut counts = HashMap::new();
    for gram in grams {
        let hash = murmurhash3_x86_32(gram.as_bytes(), 0);
        let idx = hash.unsigned_abs() as usize % spec.n_features;
        let sign = if spec.alternate_sign && hash < 0 { -1.0 } else { 1.0 };
        *counts.entry(idx).or_insert(0.0) += sign;
    }

    if spec.norm.as_deref() == Some("l2") && !counts.is_empty() {
        let norm = counts.values().map(|value| value * value).sum::<f64>().sqrt();
        if norm > 0.0 {
            for value in counts.values_mut() {
                *value /= norm;
            }
        }
    }

    Ok(counts)
}

fn ngram_range(range: &[usize]) -> Result<(usize, usize)> {
    match range {
        [min_n, max_n] => Ok((*min_n, *max_n)),
        other => Err(anyhow!("invalid hashlinear ngram_range: {:?}", other)),
    }
}

fn normalize_spaces(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut previous_was_space = false;
    for ch in input.chars() {
        let is_space = ch.is_whitespace();
        if is_space {
            if !previous_was_space {
                out.push(' ');
            }
        } else {
            out.push(ch);
        }
        previous_was_space = is_space;
    }
    out
}

fn iter_char_ngrams(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let chars = text.chars().collect::<Vec<_>>();
    let mut grams = Vec::new();
    let text_len = chars.len();

    let mut current_min = min_n;
    if current_min == 1 {
        grams.extend(chars.iter().map(|ch| ch.to_string()));
        current_min += 1;
    }

    for n in current_min..=max_n.min(text_len) {
        for start in 0..=text_len.saturating_sub(n) {
            grams.push(chars[start..start + n].iter().collect());
        }
    }
    grams
}

fn iter_word_ngrams(text: &str, (min_n, max_n): (usize, usize)) -> Vec<String> {
    let tokens = text.split_whitespace().collect::<Vec<_>>();
    let mut grams = Vec::new();
    let mut current_min = min_n;
    if current_min == 1 {
        grams.extend(tokens.iter().map(|token| (*token).to_string()));
        current_min += 1;
    }
    for n in current_min..=max_n.min(tokens.len()) {
        for start in 0..=tokens.len().saturating_sub(n) {
            grams.push(tokens[start..start + n].join(" "));
        }
    }
    grams
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn murmurhash3_x86_32(data: &[u8], seed: u32) -> i32 {
    const C1: u32 = 0xCC9E2D51;
    const C2: u32 = 0x1B87_3593;

    let len = data.len();
    let mut h1 = seed;
    let rounded_end = len & !0x3;

    for chunk in data[..rounded_end].chunks_exact(4) {
        let mut k1 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);

        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xE654_6B64);
    }

    let tail = &data[rounded_end..];
    let mut k1 = 0u32;
    match tail.len() {
        3 => {
            k1 ^= u32::from(tail[2]) << 16;
            k1 ^= u32::from(tail[1]) << 8;
            k1 ^= u32::from(tail[0]);
        }
        2 => {
            k1 ^= u32::from(tail[1]) << 8;
            k1 ^= u32::from(tail[0]);
        }
        1 => {
            k1 ^= u32::from(tail[0]);
        }
        _ => {}
    }
    if !tail.is_empty() {
        k1 = k1.wrapping_mul(C1);
        k1 = k1.rotate_left(15);
        k1 = k1.wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85EB_CA6B);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xC2B2_AE35);
    h1 ^= h1 >> 16;
    h1 as i32
}

fn shuffle_indices(indices: &mut [usize], rng: &mut XorShift64) {
    if indices.len() < 2 {
        return;
    }

    for idx in (1..indices.len()).rev() {
        let swap_idx = (rng.next_u64() as usize) % (idx + 1);
        indices.swap(idx, swap_idx);
    }
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}
