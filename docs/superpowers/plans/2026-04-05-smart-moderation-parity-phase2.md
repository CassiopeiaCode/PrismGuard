# Smart Moderation Parity Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the remaining Rust smart-moderation control-flow semantics with the Python reference for AI sampling, retry-model selection, and bounded cache behavior.

**Architecture:** Keep the current Rust smart-moderation pipeline intact and patch only the policy seams that still differ from Python. Add small, isolated helpers with direct tests first, then wire them into the request path and re-run the existing single-core aggregate suite.

**Tech Stack:** Rust, Tokio, Reqwest, Serde JSON, existing integration-test harness

---

### Task 1: Add failing policy tests for smart-moderation parity helpers

**Files:**
- Modify: `src/moderation/smart.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write failing tests for AI review-rate edge behavior**

Add focused tests that exercise helper-level behavior for:

```rust
#[test]
fn ai_review_rate_zero_never_forces_ai() {}

#[test]
fn ai_review_rate_one_always_forces_ai() {}
```

and one deterministic intermediate-rate test using an injected RNG or helper input.

- [ ] **Step 2: Run the targeted test selection to verify failure**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-httpreq cargo test --no-default-features --test http_proxy_request_tests smart_moderation -- --nocapture
```

Expected: at least one new test fails because the helper or hook does not exist yet.

- [ ] **Step 3: Add failing tests for retry model selection policy**

Add small unit tests in `src/moderation/smart.rs` under `#[cfg(test)]` for a helper that:

- prefers untried models first
- does not repeat before exhaustion
- still returns a model after all candidates were attempted

- [ ] **Step 4: Run the same targeted command and verify failure moves to helper implementation**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-httpreq cargo test --no-default-features --test http_proxy_request_tests smart_moderation -- --nocapture
```

Expected: new helper-policy tests fail with missing or incorrect behavior.

### Task 2: Implement smart-moderation policy helpers

**Files:**
- Modify: `src/moderation/smart.rs`

- [ ] **Step 1: Implement request-time AI sampling helper**

Add a helper with explicit inputs so it can be tested directly, for example:

```rust
fn should_force_ai_review_with_draw(rate: f64, draw: f64) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    draw < rate
}
```

Then make `should_force_ai_review(...)` call this helper using a runtime random draw.

- [ ] **Step 2: Implement retry-model selection helper**

Add a helper shaped like:

```rust
fn pick_model_for_attempt(models: &[String], attempted: &[String], draw_index: usize) -> Option<String> {
    let remaining = models
        .iter()
        .filter(|model| !attempted.iter().any(|used| used == *model))
        .cloned()
        .collect::<Vec<_>>();
    let pool = if remaining.is_empty() { models.to_vec() } else { remaining };
    if pool.is_empty() {
        return None;
    }
    Some(pool[draw_index % pool.len()].clone())
}
```

Wire the retry loop to use this policy instead of attempt-index rotation.

- [ ] **Step 3: Re-run targeted parity tests**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-httpreq cargo test --no-default-features --test http_proxy_request_tests smart_moderation -- --nocapture
```

Expected: targeted smart-moderation tests pass.

### Task 3: Add bounded profile/cache parity

**Files:**
- Modify: `src/profile.rs`
- Modify: `src/moderation/smart.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Write failing test for bounded profile-cache or bounded moderation-cache buckets**

Add one direct unit test around a helper that evicts the oldest profile bucket when the configured cap is exceeded.

- [ ] **Step 2: Run targeted tests to verify failure**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-httpreq cargo test --no-default-features --test http_proxy_request_tests smart_moderation -- --nocapture
```

Expected: cache-bucket test fails before implementation.

- [ ] **Step 3: Implement bounded cache-bucket logic**

Add a constant such as:

```rust
const MAX_CACHE_PROFILES: usize = 50;
```

and evict the oldest profile bucket before inserting a new one once the cap is reached. If profile caching is added, use the same bounded approach there as well.

- [ ] **Step 4: Re-run targeted tests**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-httpreq cargo test --no-default-features --test http_proxy_request_tests smart_moderation -- --nocapture
```

Expected: cache-related tests pass.

### Task 4: Re-run full single-core verification

**Files:**
- Modify: `src/moderation/smart.rs`
- Modify: `src/profile.rs`
- Test: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: Re-run the aggregate integration suite**

Run:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_HOME=/tmp/prismguard-cargo-home-phase2-alltests CARGO_TARGET_DIR=/tmp/prismguard-main-phase2-alltests cargo test --no-default-features --tests -- --nocapture
```

Expected: all integration tests pass.

- [ ] **Step 2: Re-run the parity-focused targets if aggregate exposes drift**

Run as needed:

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-bow cargo test --no-default-features --test bow_runtime_parity_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-fasttext cargo test --no-default-features --test fasttext_runtime_parity_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-hashlinear cargo test --no-default-features --test hashlinear_runtime_jieba_parity_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-phase2-pebble cargo test --no-default-features --test pebble_compat_parity_tests -- --nocapture
```

Expected: parity targets remain green.
