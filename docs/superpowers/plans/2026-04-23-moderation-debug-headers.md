# Moderation Debug Headers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default `x-prismguard-*` response headers that expose local-model confidence/latency and LLM review result/latency for every proxy request, then deploy by replacing the local binary from GitHub artifacts and restarting under supervisor.

**Architecture:** Extend smart moderation to return a debug record alongside the moderation result, thread that record through proxy success and error responses, and keep header formatting centralized so JSON, SSE, moderation-blocked, and generic proxy errors all emit the same debug surface. Deployment is handled out-of-band from local compilation by fetching a built binary and restarting the managed process.

**Tech Stack:** Rust (`axum`, `reqwest`, existing moderation modules), existing integration tests in `tests/http_proxy_request_tests.rs` and `tests/http_proxy_response_tests.rs`, `gh`, `supervisorctl`.

---

### Task 1: Add a moderation debug info model and local-model timing coverage

**Files:**
- Modify: `src/moderation/smart.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn smart_moderation_low_risk_hashlinear_response_includes_debug_headers() {
    // Reuse the existing low-risk hashlinear fixture shape, but assert headers:
    // x-prismguard-local-model-source=hashlinear_model
    // x-prismguard-local-model-decision=allow
    // x-prismguard-local-model-confidence is present
    // x-prismguard-local-model-latency-ms is present
    // x-prismguard-llm-reviewed=false
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test smart_moderation_low_risk_hashlinear_response_includes_debug_headers -- --nocapture`
Expected: FAIL because no debug headers are emitted yet.

- [ ] **Step 3: Write minimal implementation**

```rust
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModerationDebugInfo {
    pub local_model_source: Option<String>,
    pub local_model_confidence: Option<f64>,
    pub local_model_latency_ms: Option<f64>,
    pub local_model_decision: Option<String>,
    pub llm_reviewed: bool,
    pub llm_result: Option<String>,
    pub llm_latency_ms: Option<f64>,
    pub llm_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SmartModerationOutcome {
    pub result: SmartModerationResult,
    pub debug: ModerationDebugInfo,
}
```

Then update `run_local_model(...)` usage to time the local model call and populate `local_model_*` fields before returning `allow`, `block`, or `uncertain`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test smart_moderation_low_risk_hashlinear_response_includes_debug_headers -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/smart.rs tests/http_proxy_request_tests.rs
git commit -m "feat: record local moderation debug info"
```

### Task 2: Add LLM timing/result/error capture to the smart moderation cold path

**Files:**
- Modify: `src/moderation/smart.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn smart_moderation_llm_fallback_response_includes_llm_debug_headers() {
    // Reuse an existing uncertain-hashlinear -> LLM fixture.
    // Assert:
    // x-prismguard-local-model-decision=uncertain
    // x-prismguard-llm-reviewed=true
    // x-prismguard-llm-result=block or allow depending on fixture
    // x-prismguard-llm-latency-ms is present
}

#[tokio::test]
async fn smart_moderation_llm_error_response_includes_llm_error_header() {
    // Reuse or adapt an LLM parse failure fixture.
    // Assert x-prismguard-llm-result=error and x-prismguard-llm-error is present/truncated.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test smart_moderation_llm_fallback_response_includes_llm_debug_headers smart_moderation_llm_error_response_includes_llm_error_header -- --nocapture`
Expected: FAIL because LLM debug headers are not emitted yet.

- [ ] **Step 3: Write minimal implementation**

```rust
async fn llm_moderate(...) -> Result<SmartModerationResult, SmartModerationError> { ... }

async fn run_ai_moderation_with_history(...) -> Result<SmartModerationOutcome, SmartModerationError> {
    let started = Instant::now();
    let result = llm_moderate(...).await;
    debug.llm_reviewed = true;
    debug.llm_latency_ms = Some(started.elapsed().as_secs_f64() * 1000.0);
    match &result {
        Ok(parsed) => debug.llm_result = Some(if parsed.violation { "block" } else { "allow" }.to_string()),
        Err(err) => {
            debug.llm_result = Some("error".to_string());
            debug.llm_error = Some(truncate_header_value(&format!("{err:#}")));
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test smart_moderation_llm_fallback_response_includes_llm_debug_headers smart_moderation_llm_error_response_includes_llm_error_header -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/smart.rs tests/http_proxy_request_tests.rs
git commit -m "feat: capture llm moderation debug timing"
```

### Task 3: Thread moderation debug info into success and error responses

**Files:**
- Modify: `src/proxy.rs`
- Modify: `src/routes.rs`
- Test: `tests/http_proxy_response_tests.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn successful_json_response_includes_prismguard_debug_headers() {
    // Make a request that passes moderation and reaches upstream.
    // Assert headers on success response.
}

#[tokio::test]
async fn moderation_blocked_error_includes_prismguard_debug_headers() {
    // Make a request blocked by local model or LLM.
    // Assert headers are present on the BAD_REQUEST response too.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test successful_json_response_includes_prismguard_debug_headers moderation_blocked_error_includes_prismguard_debug_headers -- --nocapture`
Expected: FAIL because proxy success/error responses do not attach debug headers.

- [ ] **Step 3: Write minimal implementation**

```rust
pub struct ApiError {
    ...
    extra_headers: HeaderMap,
}

fn moderation_debug_headers(debug: &ModerationDebugInfo) -> HeaderMap { ... }

fn json_success_headers(extra: &HeaderMap) -> HeaderMap { ... }
fn stream_success_headers(headers: &HeaderMap, extra: &HeaderMap) -> HeaderMap { ... }
fn build_response(..., extra_headers: &HeaderMap, ...) -> Result<Response, ApiError> { ... }
```

Update `proxy_entry_with_cfg(...)` to build `HeaderMap` from moderation debug info and attach it to both success responses and `ApiError`s created after moderation.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test successful_json_response_includes_prismguard_debug_headers moderation_blocked_error_includes_prismguard_debug_headers -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/proxy.rs src/routes.rs tests/http_proxy_request_tests.rs tests/http_proxy_response_tests.rs
git commit -m "feat: expose moderation debug headers"
```

### Task 4: Add formatting safeguards and cache-path coverage

**Files:**
- Modify: `src/moderation/smart.rs`
- Modify: `src/proxy.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[tokio::test]
async fn smart_moderation_cached_response_does_not_fake_llm_timing_headers() {
    // Same text twice; second request should not report new LLM timing if served from cache/history.
}

#[tokio::test]
async fn llm_error_header_is_truncated_to_safe_length() {
    // Force a long error string and assert header length is bounded.
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test smart_moderation_cached_response_does_not_fake_llm_timing_headers llm_error_header_is_truncated_to_safe_length -- --nocapture`
Expected: FAIL because cache-path semantics and truncation are not encoded yet.

- [ ] **Step 3: Write minimal implementation**

```rust
fn truncate_header_value(value: &str) -> String {
    value.chars().take(160).collect()
}
```

Ensure cache-hit outcomes only emit:
- `x-prismguard-local-model-source: none`
- `x-prismguard-local-model-decision: skipped`
- `x-prismguard-llm-reviewed: false`

unless richer information is genuinely available.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test smart_moderation_cached_response_does_not_fake_llm_timing_headers llm_error_header_is_truncated_to_safe_length -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/moderation/smart.rs src/proxy.rs tests/http_proxy_request_tests.rs
git commit -m "fix: stabilize moderation debug header semantics"
```

### Task 5: Deployment without local Rust build

**Files:**
- Modify: none
- Test: runtime smoke verification against deployed binary

- [ ] **Step 1: Identify the CI artifact or release asset that contains the new `Prismguand-Rust` binary**

Run: `gh run list --limit 10`
Expected: A recent successful run or release artifact that corresponds to the new commit containing the debug headers.

- [ ] **Step 2: Download the built binary**

Run: `gh run download <run-id> --name <artifact-name> --dir /tmp/prismguard-artifact`
Expected: The built `Prismguand-Rust` binary is available under `/tmp/prismguard-artifact`.

- [ ] **Step 3: Replace the local binary atomically**

Run: `install -m 755 /tmp/prismguard-artifact/Prismguand-Rust /services/apps/Prismguand-Rust/target/release/Prismguand-Rust`
Expected: Local service binary is replaced with the downloaded build.

- [ ] **Step 4: Restart the managed service**

Run: `supervisorctl restart prismguard-rust`
Expected: Supervisor reports the service restarted successfully.

- [ ] **Step 5: Verify headers on a live request**

Run: `curl -i -sS http://127.0.0.1:<port>/<config>$<upstream> ...`
Expected: Response includes the new `x-prismguard-*` headers, proving the deployed binary is active.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "chore: deploy moderation debug header build"
```

### Task 6: Final verification

**Files:**
- Modify: none
- Test: targeted cargo tests + live curl validation

- [ ] **Step 1: Run targeted tests for the modified behavior**

Run: `cargo test smart_moderation -- --nocapture`
Expected: Relevant moderation tests pass.

- [ ] **Step 2: Run response-header coverage tests**

Run: `cargo test debug_headers -- --nocapture`
Expected: The new response-header assertions pass.

- [ ] **Step 3: Verify the deployed service with both a likely-local-model request and a likely-LLM-fallback request**

Run: two `curl` commands against the live service
Expected: Headers clearly distinguish local-model confidence from LLM fallback timing.

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: finalize moderation debug headers rollout"
```
