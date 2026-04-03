use std::fs;
use std::future::Future;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

use crate::config::Settings;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub enum SampleRpcTransport {
    Unix,
    Tcp,
}

#[derive(Debug, Clone, Serialize)]
pub struct SampleRpcConfig {
    pub enabled: bool,
    pub transport: SampleRpcTransport,
    pub unix_socket: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "method", rename_all = "snake_case")]
pub enum SampleRpcRequest {
    GetSampleCount {
        profile: String,
        db_path: String,
    },
    CleanupExcessSamples {
        profile: String,
        db_path: String,
        max_items: usize,
    },
    LoadBalancedSamples {
        profile: String,
        db_path: String,
        max_samples: usize,
    },
    LoadBalancedLatestSamples {
        profile: String,
        db_path: String,
        max_samples: usize,
    },
    LoadBalancedRandomSamples {
        profile: String,
        db_path: String,
        max_samples: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SampleRpcResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SampleRpcResponse {
    pub fn ok(result: Value) -> Self {
        Self {
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(message: impl Into<String>) -> Self {
        Self {
            ok: false,
            result: None,
            error: Some(message.into()),
        }
    }
}

impl SampleRpcConfig {
    pub fn from_settings(settings: &Settings) -> Result<Self> {
        let transport = match settings.training_data_rpc_transport.trim().to_ascii_lowercase().as_str() {
            "unix" => SampleRpcTransport::Unix,
            "tcp" => SampleRpcTransport::Tcp,
            other => return Err(anyhow!("unsupported TRAINING_DATA_RPC_TRANSPORT: {other}")),
        };

        Ok(Self {
            enabled: settings.training_data_rpc_enabled,
            transport,
            unix_socket: PathBuf::from(&settings.training_data_rpc_unix_socket),
        })
    }

    pub fn prepare_runtime(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        if self.transport == SampleRpcTransport::Unix {
            let Some(parent) = self.unix_socket.parent() else {
                return Err(anyhow!(
                    "TRAINING_DATA_RPC_UNIX_SOCKET has no parent directory: {}",
                    self.unix_socket.display()
                ));
            };
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create RPC socket dir {}", parent.display()))?;
        }

        Ok(())
    }

    pub fn unix_socket_path(&self) -> Result<&Path> {
        if !self.enabled {
            return Err(anyhow!("sample RPC is disabled"));
        }
        if self.transport != SampleRpcTransport::Unix {
            return Err(anyhow!(
                "sample RPC transport {:?} does not expose a unix socket",
                self.transport
            ));
        }
        Ok(self.unix_socket.as_path())
    }
}

pub fn cleanup_stale_unix_socket(path: &Path) -> Result<()> {
    if path.exists() {
        fs::remove_file(path)
            .with_context(|| format!("failed to remove stale unix socket {}", path.display()))?;
    }
    Ok(())
}

pub fn dispatch_request<F>(request: SampleRpcRequest, mut handler: F) -> SampleRpcResponse
where
    F: FnMut(&SampleRpcRequest) -> Result<Value>,
{
    match handler(&request) {
        Ok(result) => SampleRpcResponse::ok(result),
        Err(err) => SampleRpcResponse::err(format!("{err:#}")),
    }
}

#[cfg(feature = "storage-debug")]
pub fn dispatch_request_with_storage(request: SampleRpcRequest) -> SampleRpcResponse {
    dispatch_request(request, |request| match request {
        SampleRpcRequest::GetSampleCount { db_path, .. } => {
            let storage = crate::storage::SampleStorage::open_read_only(db_path)?;
            let sample_count = storage
                .sample_count()
                .ok_or_else(|| anyhow!("sample count metadata not found"))?;
            Ok(serde_json::json!({
                "sample_count": sample_count
            }))
        }
        SampleRpcRequest::LoadBalancedSamples { db_path, max_samples, .. } => {
            let storage = crate::storage::SampleStorage::open_read_only(db_path)?;
            Ok(serde_json::json!({
                "samples": storage
                    .load_balanced_samples(*max_samples)
                    .into_iter()
                    .map(|sample| sample.value)
                    .collect::<Vec<_>>()
            }))
        }
        SampleRpcRequest::LoadBalancedLatestSamples { db_path, max_samples, .. } => {
            let storage = crate::storage::SampleStorage::open_read_only(db_path)?;
            Ok(serde_json::json!({
                "samples": storage
                    .load_balanced_latest_samples(*max_samples)
                    .into_iter()
                    .map(|sample| sample.value)
                    .collect::<Vec<_>>()
            }))
        }
        SampleRpcRequest::LoadBalancedRandomSamples { db_path, max_samples, .. } => {
            let storage = crate::storage::SampleStorage::open_read_only(db_path)?;
            Ok(serde_json::json!({
                "samples": storage
                    .load_balanced_random_samples(*max_samples)
                    .into_iter()
                    .map(|sample| sample.value)
                    .collect::<Vec<_>>()
            }))
        }
        SampleRpcRequest::CleanupExcessSamples {
            db_path, max_items, ..
        } => {
            let removed = cleanup_excess_samples_in_rocksdb(db_path, *max_items)?;
            Ok(serde_json::json!({
                "removed": removed
            }))
        }
    })
}

#[cfg(feature = "storage-debug")]
fn cleanup_excess_samples_in_rocksdb(db_path: &str, max_items: usize) -> Result<usize> {
    use rocksdb::{DBWithThreadMode, IteratorMode, MultiThreaded, Options, WriteBatch};

    let mut options = Options::default();
    options.create_if_missing(false);
    options.set_comparator("rocksdict", Box::new(|lhs, rhs| lhs.cmp(rhs)));
    let db = DBWithThreadMode::<MultiThreaded>::open(&options, db_path)
        .with_context(|| format!("failed to open RocksDB {db_path}"))?;

    let mut samples = Vec::new();
    for entry in db.iterator(IteratorMode::Start) {
        let Ok((raw_key, raw_value)) = entry else {
            continue;
        };
        let key = decode_rocksdict_string(&raw_key)?;
        if !key.starts_with("sample:") {
            continue;
        }

        let value_str = decode_rocksdict_string(&raw_value)?;
        let value: Value = serde_json::from_str(&value_str)
            .with_context(|| format!("failed to decode sample JSON for key {key}"))?;
        samples.push((key, value));
    }

    if samples.len() <= max_items {
        return Ok(0);
    }

    samples.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    let removed_samples = &samples[..samples.len() - max_items];
    let removed = removed_samples.len();

    let mut removed_pass = 0usize;
    let mut removed_violation = 0usize;
    let mut batch = WriteBatch::default();
    for (key, value) in removed_samples {
        let label = value.get("label").and_then(Value::as_i64).unwrap_or_default();
        match label {
            0 => removed_pass += 1,
            1 => removed_violation += 1,
            _ => {}
        }
        batch.delete(encode_rocksdict_string(key));
    }

    let total_count = read_rocks_u64(&db, "meta:count").unwrap_or(samples.len() as u64);
    let pass_count = read_rocks_u64(&db, "meta:count:0").unwrap_or_default();
    let violation_count = read_rocks_u64(&db, "meta:count:1").unwrap_or_default();
    let next_total = total_count.saturating_sub(removed as u64);
    let next_pass = pass_count.saturating_sub(removed_pass as u64);
    let next_violation = violation_count.saturating_sub(removed_violation as u64);

    batch.put(
        encode_rocksdict_string("meta:count"),
        encode_rocksdict_string(&next_total.to_string()),
    );
    batch.put(
        encode_rocksdict_string("meta:count:0"),
        encode_rocksdict_string(&next_pass.to_string()),
    );
    batch.put(
        encode_rocksdict_string("meta:count:1"),
        encode_rocksdict_string(&next_violation.to_string()),
    );

    db.write(batch)
        .with_context(|| format!("failed to cleanup excess samples in {db_path}"))?;
    Ok(removed)
}

#[cfg(feature = "storage-debug")]
fn read_rocks_u64(
    db: &rocksdb::DBWithThreadMode<rocksdb::MultiThreaded>,
    key: &str,
) -> Option<u64> {
    db.get(encode_rocksdict_string(key))
        .ok()
        .flatten()
        .and_then(|raw| decode_rocksdict_string(&raw).ok())
        .and_then(|value| value.parse::<u64>().ok())
}

#[cfg(feature = "storage-debug")]
fn encode_rocksdict_string(value: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(value.len() + 1);
    out.push(0x02);
    out.extend_from_slice(value.as_bytes());
    out
}

#[cfg(feature = "storage-debug")]
fn decode_rocksdict_string(raw: &[u8]) -> Result<String> {
    let payload = raw
        .strip_prefix(&[0x02])
        .ok_or_else(|| anyhow!("invalid rocksdict string prefix"))?;
    String::from_utf8(payload.to_vec()).context("invalid rocksdict utf-8 payload")
}

pub fn encode_request_line(request: &SampleRpcRequest) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(request).context("failed to encode sample RPC request")?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn decode_request_line(line: &[u8]) -> Result<SampleRpcRequest> {
    serde_json::from_slice(trim_ascii_whitespace_end(line))
        .context("failed to decode sample RPC request")
}

pub fn encode_response_line(response: &SampleRpcResponse) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(response).context("failed to encode sample RPC response")?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn decode_response_line(line: &[u8]) -> Result<SampleRpcResponse> {
    serde_json::from_slice(trim_ascii_whitespace_end(line))
        .context("failed to decode sample RPC response")
}

fn trim_ascii_whitespace_end(line: &[u8]) -> &[u8] {
    let end = line
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|idx| idx + 1)
        .unwrap_or(0);
    &line[..end]
}

pub async fn send_unix_request(
    socket_path: &Path,
    request: &SampleRpcRequest,
) -> Result<SampleRpcResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .with_context(|| format!("failed to connect sample RPC socket {}", socket_path.display()))?;
    let bytes = encode_request_line(request)?;
    stream
        .write_all(&bytes)
        .await
        .context("failed to write sample RPC request")?;
    stream.flush().await.context("failed to flush sample RPC request")?;

    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    let read = reader
        .read_until(b'\n', &mut line)
        .await
        .context("failed to read sample RPC response")?;
    if read == 0 {
        return Err(anyhow!("sample RPC server closed connection without response"));
    }
    decode_response_line(&line)
}

pub async fn serve_one_unix_request<F>(
    socket_path: &Path,
    handler: F,
) -> Result<()>
where
    F: FnOnce(SampleRpcRequest) -> Result<SampleRpcResponse>,
{
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind sample RPC socket {}", socket_path.display()))?;
    let (stream, _) = listener.accept().await.context("failed to accept sample RPC connection")?;
    handle_unix_stream(stream, handler).await
}

pub async fn serve_unix_requests_until_shutdown<F, Fut>(
    socket_path: &Path,
    handler: F,
    shutdown: Fut,
) -> Result<()>
where
    F: Fn(SampleRpcRequest) -> SampleRpcResponse + Clone + Send + Sync + 'static,
    Fut: Future<Output = ()>,
{
    let listener = UnixListener::bind(socket_path)
        .with_context(|| format!("failed to bind sample RPC socket {}", socket_path.display()))?;
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            _ = &mut shutdown => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted.context("failed to accept sample RPC connection")?;
                let handler = handler.clone();
                tokio::spawn(async move {
                    let _ = handle_unix_stream(stream, move |request| Ok(handler(request))).await;
                });
            }
        }
    }

    Ok(())
}

#[cfg(feature = "storage-debug")]
pub async fn serve_storage_sample_rpc_until_shutdown<Fut>(
    socket_path: &Path,
    shutdown: Fut,
) -> Result<()>
where
    Fut: Future<Output = ()>,
{
    serve_unix_requests_until_shutdown(socket_path, dispatch_request_with_storage, shutdown).await
}

async fn handle_unix_stream<F>(stream: UnixStream, handler: F) -> Result<()>
where
    F: FnOnce(SampleRpcRequest) -> Result<SampleRpcResponse>,
{
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    let read = reader
        .read_until(b'\n', &mut line)
        .await
        .context("failed to read sample RPC request")?;
    if read == 0 {
        return Err(anyhow!("sample RPC client closed connection without request"));
    }

    let request = decode_request_line(&line)?;
    let response = handler(request)?;
    let bytes = encode_response_line(&response)?;
    let stream = reader.get_mut();
    stream
        .write_all(&bytes)
        .await
        .context("failed to write sample RPC response")?;
    stream.flush().await.context("failed to flush sample RPC response")?;
    Ok(())
}
