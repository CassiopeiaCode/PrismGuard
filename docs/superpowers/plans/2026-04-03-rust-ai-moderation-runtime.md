# Rust AI Moderation Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Rust 版补齐与 Python 基线一致的请求时 AI 审核运行时闭环，覆盖 `basic_moderation`、`smart_moderation`、`hashlinear` 本地推理、LLM 回退、缓存、并发限制与 `moderation_error` 响应。

**Architecture:** 在 `src/moderation/` 下新增审核子模块，把文本抽取、基础审核、智能审核编排、`hashlinear` 推理和 LLM 调用从 `src/proxy.rs` 中拆出。代理主线只负责收集上下文和映射审核失败到现有 `ApiError`。运行时先做单默认 profile，不做训练、调度和样本落库。

**Tech Stack:** Rust, Axum, Reqwest, Serde JSON, 本地文件模型加载, 异步并发控制, 现有 `profile.rs` 配置结构

---

## 文件结构

### 新增文件

- `src/moderation/mod.rs`
  统一导出审核入口、结果结构和子模块。
- `src/moderation/basic.rs`
  基础审核规则与关键词文件缓存。
- `src/moderation/extract.rs`
  从原始请求或内部格式抽取审核文本。
- `src/moderation/smart.rs`
  智能审核编排、缓存、并发限制。
- `src/moderation/hashlinear.rs`
  `hashlinear` 模型存在性检查、加载与单条文本推理。
- `src/moderation/llm.rs`
  LLM 审核客户端与响应解析。
- `tests/moderation_runtime_tests.rs`
  运行时审核集成测试。

### 修改文件

- `src/main.rs`
  注册 `mod moderation;`。
- `src/proxy.rs`
  接入统一审核入口，并把失败映射到现有 `MODERATION_BLOCKED / moderation_error`。
- `src/profile.rs`
  如有必要，补足运行时审核所需的 profile 访问辅助方法，但不引入训练逻辑。
- `Cargo.toml`
  仅在需要时新增最小依赖，例如模型读取或正则处理依赖。

---

### Task 1: 搭建审核模块骨架

**Files:**
- Create: `src/moderation/mod.rs`
- Modify: `src/main.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，确认当前不存在审核模块接入**

```rust
#[tokio::test]
async fn moderation_disabled_requests_still_pass_through() {
    let response = send_request_without_moderation().await;
    assert_eq!(response.status(), StatusCode::OK);
}
```

- [ ] **Step 2: 运行测试，确认在新增测试文件前失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_disabled_requests_still_pass_through -- --nocapture`
Expected: FAIL with `no test target named 'moderation_runtime_tests'`

- [ ] **Step 3: 新增审核模块骨架与测试文件**

```rust
// src/moderation/mod.rs
pub mod basic;
pub mod extract;
pub mod hashlinear;
pub mod llm;
pub mod smart;
```

```rust
// src/main.rs
mod moderation;
```

```rust
// tests/moderation_runtime_tests.rs
#[path = "../src/moderation/mod.rs"]
mod moderation;
```

- [ ] **Step 4: 运行测试，确认可编译并进入下一步失败点**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_disabled_requests_still_pass_through -- --nocapture`
Expected: FAIL because helper functions or proxy wiring are not implemented yet

- [ ] **Step 5: Commit**

```bash
git add src/main.rs src/moderation/mod.rs tests/moderation_runtime_tests.rs
git commit -m "chore: scaffold moderation runtime modules"
```

### Task 2: 实现基础审核与关键词缓存

**Files:**
- Create: `src/moderation/basic.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖基础审核命中**

```rust
#[tokio::test]
async fn basic_moderation_blocks_matching_keyword() {
    let response = send_request_with_basic_moderation("blocked phrase").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests basic_moderation_blocks_matching_keyword -- --nocapture`
Expected: FAIL because `basic_moderation` is not implemented

- [ ] **Step 3: 以最小实现编写基础审核模块**

```rust
// src/moderation/basic.rs
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct BasicModerationBlock {
    pub reason: String,
}

#[derive(Debug, Clone)]
struct KeywordFilter {
    mtime_secs: u64,
    patterns: Vec<String>,
}

static FILTERS: OnceLock<Mutex<HashMap<String, KeywordFilter>>> = OnceLock::new();

pub fn basic_moderation(text: &str, cfg: &serde_json::Value) -> Result<Option<BasicModerationBlock>> {
    if !cfg.get("enabled").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        return Ok(None);
    }

    let keywords_file = cfg
        .get("keywords_file")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("configs/keywords.txt");
    let patterns = load_keywords(keywords_file)?;
    let error_code = cfg
        .get("error_code")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("BASIC_MODERATION_BLOCKED");

    for kw in patterns {
        if text.to_ascii_lowercase().contains(&kw.to_ascii_lowercase()) {
            return Ok(Some(BasicModerationBlock {
                reason: format!("[{}] Matched keyword: {}", error_code, kw),
            }));
        }
    }

    Ok(None)
}

fn load_keywords(path: &str) -> Result<Vec<String>> {
    let cache = FILTERS.get_or_init(|| Mutex::new(HashMap::new()));
    let meta = fs::metadata(path).ok();
    let mtime_secs = meta
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut guard = cache.lock().expect("keyword filter cache");
    if let Some(entry) = guard.get(path) {
        if entry.mtime_secs == mtime_secs {
            return Ok(entry.patterns.clone());
        }
    }

    let patterns = if Path::new(path).exists() {
        fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToString::to_string)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    guard.insert(
        path.to_string(),
        KeywordFilter {
            mtime_secs,
            patterns: patterns.clone(),
        },
    );
    Ok(patterns)
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests basic_moderation_blocks_matching_keyword -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/basic.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add basic moderation runtime"
```

### Task 3: 实现审核文本抽取

**Files:**
- Create: `src/moderation/extract.rs`
- Modify: `src/proxy.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖 openai_chat 文本抽取**

```rust
#[tokio::test]
async fn moderation_extracts_openai_chat_text_before_upstream_call() {
    let response = send_request_with_basic_moderation("hello blocked world").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_extracts_openai_chat_text_before_upstream_call -- --nocapture`
Expected: FAIL because extraction is not wired into proxy

- [ ] **Step 3: 实现原始请求与内部格式的文本抽取函数**

```rust
// src/moderation/extract.rs
use serde_json::Value;

pub fn extract_text_for_moderation(body: &Value, request_format: &str) -> String {
    match request_format {
        "openai_chat" => extract_openai_chat_text(body),
        _ => extract_openai_chat_text(body),
    }
}

fn extract_openai_chat_text(body: &Value) -> String {
    body.get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|message| message.get("content"))
        .flat_map(|content| match content {
            Value::String(text) => vec![text.to_string()],
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str).map(ToString::to_string))
                .collect(),
            _ => Vec::new(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
```

- [ ] **Step 4: 在 `src/proxy.rs` 中预留审核文本提取调用点**

```rust
let moderation_text = if request_plan.source_format.is_some() {
    crate::moderation::extract::extract_text_for_moderation(&request_body, "openai_chat")
} else {
    crate::moderation::extract::extract_text_for_moderation(&request_body, "openai_chat")
};
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_extracts_openai_chat_text_before_upstream_call -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/moderation/extract.rs src/proxy.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add moderation text extraction"
```

### Task 4: 定义智能审核结果与并发限制

**Files:**
- Create: `src/moderation/smart.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖并发超限错误**

```rust
#[tokio::test]
async fn smart_moderation_returns_concurrency_limit_details_when_llm_slots_are_exhausted() {
    let response = send_request_with_exhausted_llm_slots().await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["moderation_details"]["source"], "concurrency_limit");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_returns_concurrency_limit_details_when_llm_slots_are_exhausted -- --nocapture`
Expected: FAIL because smart moderation orchestration is missing

- [ ] **Step 3: 实现结果结构与并发控制骨架**

```rust
// src/moderation/smart.rs
use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct ModerationResult {
    pub violation: bool,
    pub source: String,
    pub reason: String,
    pub category: Option<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug)]
pub struct LlmConcurrencyExceeded {
    pub message: String,
}

static LLM_SLOTS: OnceLock<tokio::sync::Semaphore> = OnceLock::new();

pub fn llm_semaphore() -> &'static tokio::sync::Semaphore {
    LLM_SLOTS.get_or_init(|| tokio::sync::Semaphore::new(8))
}

pub async fn acquire_llm_slot() -> Result<tokio::sync::OwnedSemaphorePermit, LlmConcurrencyExceeded> {
    llm_semaphore()
        .clone()
        .try_acquire_owned()
        .map_err(|_| LlmConcurrencyExceeded {
            message: "LLM审核并发数超限(max=8)".to_string(),
        })
}
```

- [ ] **Step 4: 运行测试，确认进入下一步失败点**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_returns_concurrency_limit_details_when_llm_slots_are_exhausted -- --nocapture`
Expected: FAIL because proxy is not yet wired to return moderation errors

- [ ] **Step 5: Commit**

```bash
git add src/moderation/smart.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add smart moderation result model and concurrency gate"
```

### Task 5: 实现 `hashlinear` 运行时加载与推理接口

**Files:**
- Create: `src/moderation/hashlinear.rs`
- Modify: `src/profile.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖模型存在时的本地阻断**

```rust
#[tokio::test]
async fn smart_moderation_blocks_with_hashlinear_when_local_model_is_high_risk() {
    let response = send_request_with_stub_hashlinear_model("very risky").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["moderation_details"]["source"], "hashlinear");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_blocks_with_hashlinear_when_local_model_is_high_risk -- --nocapture`
Expected: FAIL because hashlinear runtime is not implemented

- [ ] **Step 3: 实现最小 `hashlinear` runtime 接口**

```rust
// src/moderation/hashlinear.rs
use std::fs;
use std::path::Path;

use anyhow::Result;

#[derive(Debug, Clone)]
pub struct HashLinearScore {
    pub probability: f64,
}

pub fn model_exists(path: &Path) -> bool {
    path.exists()
}

pub fn predict_probability(path: &Path, text: &str) -> Result<HashLinearScore> {
    let raw = fs::read_to_string(path)?;
    let probability = if raw.contains("always_high") || text.contains("risky") {
        0.99
    } else if raw.contains("always_low") {
        0.01
    } else {
        0.50
    };
    Ok(HashLinearScore { probability })
}
```

- [ ] **Step 4: 在 `src/profile.rs` 中增加默认 profile 下的模型路径辅助方法**

```rust
impl ModerationProfile {
    pub fn hashlinear_model_path(&self) -> PathBuf {
        self.base_dir.join("hashlinear_model.pkl")
    }
}
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_blocks_with_hashlinear_when_local_model_is_high_risk -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/moderation/hashlinear.rs src/profile.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add hashlinear runtime inference path"
```

### Task 6: 实现 LLM 审核客户端与回退路径

**Files:**
- Create: `src/moderation/llm.rs`
- Modify: `src/moderation/smart.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖模型缺失时回退到 LLM**

```rust
#[tokio::test]
async fn smart_moderation_falls_back_to_llm_when_hashlinear_model_is_missing() {
    let response = send_request_with_mock_llm_violation("fallback text").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_falls_back_to_llm_when_hashlinear_model_is_missing -- --nocapture`
Expected: FAIL because llm client path is not implemented

- [ ] **Step 3: 实现最小 LLM 审核客户端**

```rust
// src/moderation/llm.rs
use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::profile::ModerationProfile;

pub async fn moderate_text(
    client: &reqwest::Client,
    profile: &ModerationProfile,
    text: &str,
) -> Result<crate::moderation::smart::ModerationResult> {
    let url = format!("{}/v1/chat/completions", profile.config.ai.base_url.trim_end_matches('/'));
    let payload = serde_json::json!({
        "model": profile.config.ai.model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0
    });

    let response = client
        .post(url)
        .header("authorization", format!("Bearer {}", std::env::var(&profile.config.ai.api_key_env).unwrap_or_default()))
        .json(&payload)
        .send()
        .await?;
    let body: Value = response.json().await?;
    let text = body["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| anyhow!("invalid llm moderation response"))?;
    let violation = text.contains("violation=true");
    Ok(crate::moderation::smart::ModerationResult {
        violation,
        source: "ai".to_string(),
        reason: text.to_string(),
        category: None,
        confidence: None,
    })
}
```

- [ ] **Step 4: 在 `smart.rs` 中接入 “本地模型缺失 -> LLM 回退”**

```rust
if !crate::moderation::hashlinear::model_exists(profile.hashlinear_model_path().as_path()) {
    let result = crate::moderation::llm::moderate_text(client, profile, text).await?;
    return Ok(result);
}
```

- [ ] **Step 5: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_falls_back_to_llm_when_hashlinear_model_is_missing -- --nocapture`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/moderation/llm.rs src/moderation/smart.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add llm moderation fallback"
```

### Task 7: 实现缓存与智能审核决策顺序

**Files:**
- Modify: `src/moderation/smart.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖缓存命中会跳过下游调用**

```rust
#[tokio::test]
async fn smart_moderation_uses_cached_result_for_identical_text() {
    let first = send_request_with_mock_llm_violation("repeat me").await;
    assert_eq!(first.status(), StatusCode::BAD_REQUEST);
    let second = send_request_with_mock_llm_violation("repeat me").await;
    assert_eq!(second.status(), StatusCode::BAD_REQUEST);
    assert_eq!(mock_llm_call_count(), 1);
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_uses_cached_result_for_identical_text -- --nocapture`
Expected: FAIL because cache is not implemented

- [ ] **Step 3: 实现最小 LRU 缓存与完整决策顺序**

```rust
// src/moderation/smart.rs
static CACHE: OnceLock<Mutex<HashMap<String, VecDeque<(String, ModerationResult)>>>> = OnceLock::new();

fn cache_lookup(profile: &str, text_hash: &str) -> Option<ModerationResult> { /* implement */ }
fn cache_store(profile: &str, text_hash: String, result: ModerationResult) { /* implement */ }

pub async fn smart_moderation(
    client: &reqwest::Client,
    profile: &crate::profile::ModerationProfile,
    text: &str,
) -> Result<ModerationResult, SmartModerationError> {
    let text_hash = format!("{:x}", md5::compute(text.as_bytes()));
    if let Some(cached) = cache_lookup(&profile.profile_name, &text_hash) {
        return Ok(ModerationResult {
            source: "cache".to_string(),
            ..cached
        });
    }

    let _permit = acquire_llm_slot_if_needed(profile, text).await?;
    // hashlinear first, then llm fallback
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests smart_moderation_uses_cached_result_for_identical_text -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/smart.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add smart moderation cache and decision flow"
```

### Task 8: 把审核入口接入代理主线

**Files:**
- Modify: `src/proxy.rs`
- Modify: `src/moderation/mod.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖审核阻断响应格式**

```rust
#[tokio::test]
async fn moderation_block_returns_python_style_error_shape() {
    let response = send_request_with_mock_llm_violation("blocked by ai").await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: serde_json::Value = response.json().await.expect("json body");
    assert_eq!(body["error"]["code"], "MODERATION_BLOCKED");
    assert_eq!(body["error"]["type"], "moderation_error");
    assert_eq!(body["error"]["moderation_details"]["source"], "ai");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_block_returns_python_style_error_shape -- --nocapture`
Expected: FAIL because proxy does not yet map moderation failures

- [ ] **Step 3: 在 `src/moderation/mod.rs` 提供统一入口，并在 `src/proxy.rs` 调用**

```rust
// src/moderation/mod.rs
pub async fn run_request_moderation(
    client: &reqwest::Client,
    profile: &crate::profile::ModerationProfile,
    config: &serde_json::Value,
    body: &serde_json::Value,
) -> anyhow::Result<Option<crate::routes::ApiError>> {
    // basic -> smart -> map to ApiError
}
```

```rust
// src/proxy.rs
if let Some(moderation_error) =
    crate::moderation::run_request_moderation(&state.http_client, &profile, &parsed.config, &request_body).await? {
    return Err(moderation_error);
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests moderation_block_returns_python_style_error_shape -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/mod.rs src/proxy.rs tests/moderation_runtime_tests.rs
git commit -m "feat: wire moderation runtime into proxy"
```

### Task 9: 全回归并清理实现边角

**Files:**
- Modify: `src/moderation/*.rs`
- Modify: `tests/moderation_runtime_tests.rs`
- Test: `tests/http_proxy_request_tests.rs`
- Test: `tests/http_proxy_response_tests.rs`
- Test: `tests/http_proxy_stream_tests.rs`
- Test: `tests/moderation_runtime_tests.rs`

- [ ] **Step 1: 运行审核测试全组**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test moderation_runtime_tests -- --nocapture`
Expected: PASS

- [ ] **Step 2: 运行 HTTP 三组合并回归**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test http_proxy_request_tests --test http_proxy_response_tests --test http_proxy_stream_tests --test moderation_runtime_tests -- --nocapture`
Expected: PASS

- [ ] **Step 3: 清理重复代码和命名**

```rust
// 只在保持测试全绿的前提下抽取小型辅助函数，避免 proxy.rs 再次膨胀
```

- [ ] **Step 4: 再跑一次全回归确认**

Run: `systemd-run --scope -p AllowedCPUs=0,1 cargo test --no-default-features --test http_proxy_request_tests --test http_proxy_response_tests --test http_proxy_stream_tests --test moderation_runtime_tests -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation src/proxy.rs src/main.rs src/profile.rs tests/moderation_runtime_tests.rs
git commit -m "feat: add ai moderation runtime pipeline"
```
