# Streaming Parity Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete SSE response transformation parity across the four supported request formats for Rust proxy streaming paths.

**Architecture:** Replace the current one-off Responses-to-Chat stream conversion with a small internal event transcoder modeled after the Python reference. Decode upstream SSE into internal events, then emit target-format SSE via per-format sinks.

**Tech Stack:** Rust, Axum proxy tests, serde_json SSE payload handling

---

### Task 1: Lock missing stream conversions with failing tests

**Files:**
- Modify: `tests/http_proxy_stream_tests.rs`
- Test: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Write failing tests**

Add focused tests covering:
- Claude Messages SSE -> OpenAI Chat SSE
- Gemini alt=sse -> OpenAI Chat SSE
- OpenAI Chat SSE -> OpenAI Responses SSE

- [ ] **Step 2: Run targeted tests to verify they fail**

Run: `taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_stream_tests -- --nocapture`
Expected: new stream parity tests fail because `maybe_transform_sse` only supports `openai_responses -> openai_chat`

### Task 2: Implement generic SSE transcoder

**Files:**
- Modify: `src/streaming.rs`
- Test: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Add internal event decoder + target sinks**

Implement a generic transcoder for:
- `openai_chat`
- `openai_responses`
- `claude_chat`
- `gemini_chat`

- [ ] **Step 2: Keep current Responses-to-Chat semantics green**

Preserve existing behavior for:
- finish reasons
- usage propagation
- buffered tool-call argument handling

- [ ] **Step 3: Run targeted tests to verify green**

Run: `taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_stream_tests -- --nocapture`
Expected: new and existing stream tests pass

### Task 3: Verify no regression in response/request parity

**Files:**
- Modify: `src/streaming.rs`
- Test: `tests/http_proxy_request_tests.rs`
- Test: `tests/http_proxy_response_tests.rs`

- [ ] **Step 1: Run related parity suites**

Run:
- `taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_request_tests -- --nocapture`
- `taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_response_tests -- --nocapture`

- [ ] **Step 2: Run final stream suite again**

Run: `taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_stream_tests -- --nocapture`
Expected: all three suites stay green
