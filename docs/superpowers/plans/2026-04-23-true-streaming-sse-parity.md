# True Streaming SSE Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Prismguand-Rust preserve true streaming for SSE passthrough and SSE format conversion, while keeping `delay_stream_header` semantics aligned with GuardianBridge-UV.

**Architecture:** Split proxy response handling into two execution models: whole-body handling for non-stream responses and incremental byte-stream handling for SSE responses. Introduce an incremental SSE pre-reader and an incremental transcoder so the proxy can pre-read only a bounded prefix for validation, then replay buffered bytes and continue streaming without waiting for the full upstream body.

**Tech Stack:** Rust, Axum, Reqwest byte streams, futures/stream adapters, existing SSE transformer and proxy tests.

---

## File Map

- Modify: `src/proxy.rs`
  Responsibility: split non-stream and stream response handling, add bounded SSE pre-read, build streaming response bodies from `reqwest::Response::bytes_stream()`.
- Modify: `src/streaming.rs`
  Responsibility: convert SSE transformation from whole-body parsing to incremental frame parsing with carry-over buffer and flush semantics.
- Modify: `tests/http_proxy_stream_tests.rs`
  Responsibility: cover passthrough streaming latency behavior, delay-header bounded pre-read behavior, and transformed-stream chunked behavior.
- Optional modify: `tests/http_proxy_response_tests.rs`
  Responsibility: keep non-stream `delay_stream_header` behavior unchanged if any shared helper changes require test adjustments.

## Acceptance Criteria

- [ ] Direct SSE passthrough is true streaming: first downstream event arrives well before upstream stream completion.
- [ ] SSE with response format conversion is also true streaming: transformed downstream events arrive incrementally, not only at stream end.
- [ ] `delay_stream_header=true` behaves like Python: bounded pre-read before committing headers, reject empty/too-short streams before headers, then replay pre-read bytes and continue true streaming.
- [ ] Protection limit matches current Python semantics: after bounded pre-read exceeds threshold, force pass-through instead of buffering the full stream.
- [ ] Non-stream responses keep existing behavior.

### Task 1: Split Stream and Non-Stream Proxy Paths

**Files:**
- Modify: `src/proxy.rs`
- Test: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Introduce a dedicated streaming response path**

Refactor `build_proxy_response(...)` so SSE responses stop using `upstream_response.bytes().await`. Keep the current whole-body logic only for non-stream responses.

Implementation target:

```rust
async fn build_proxy_response(
    upstream_response: reqwest::Response,
    client_format: Option<RequestFormat>,
    upstream_format: Option<RequestFormat>,
    delay_stream_header: bool,
    request_expects_stream: bool,
    moderation_debug: &HeaderMap,
) -> Result<Response, ApiError> {
    let status = upstream_response.status();
    let header_says_stream = is_stream_response(upstream_response.headers());

    if header_says_stream || request_expects_stream {
        return build_streaming_proxy_response(
            upstream_response,
            client_format,
            upstream_format,
            delay_stream_header,
            moderation_debug,
        )
        .await;
    }

    build_non_stream_proxy_response(
        upstream_response,
        client_format,
        upstream_format,
        delay_stream_header,
        moderation_debug,
    )
    .await
}
```

- [ ] **Step 2: Preserve existing non-stream behavior unchanged**

Move the current `bytes().await`-based JSON/text handling into `build_non_stream_proxy_response(...)`. Do not change validation semantics there.

- [ ] **Step 3: Add a failing timing regression test for passthrough SSE**

Test idea:

```rust
#[tokio::test]
async fn passthrough_sse_starts_before_upstream_finishes() {
    // upstream emits many chunks with delays
    // assert first downstream line arrives much earlier than total duration
}
```

- [ ] **Step 4: Verify the new test would fail on the old structure and pass after the split**

Run:

```bash
cargo test http_proxy_stream_tests::passthrough_sse_starts_before_upstream_finishes -- --nocapture
```

Expected after implementation: PASS, with first-event latency significantly less than total stream duration.

### Task 2: Implement Bounded Pre-Read for `delay_stream_header`

**Files:**
- Modify: `src/proxy.rs`
- Test: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Replace whole-buffer validation with an incremental checker**

Add a checker state object in `src/proxy.rs` analogous to Python’s `StreamChecker`. It should support:

```rust
struct StreamPrecheck {
    format: RequestFormat,
    total_bytes: usize,
    accumulated_text_chars: usize,
    has_tool_call: bool,
}

impl StreamPrecheck {
    fn push_chunk(&mut self, chunk: &[u8]) -> PrecheckStatus;
}
```

Where `PrecheckStatus` is:

```rust
enum PrecheckStatus {
    Pending,
    Passed,
    Failed(String),
    ForcePassProtectionLimit,
}
```

- [ ] **Step 2: Pre-read only a bounded prefix before sending headers**

In the streaming path:
- create `bytes_stream()`
- read chunks into `Vec<Bytes>` buffer
- call `StreamPrecheck::push_chunk(...)` on each chunk
- stop pre-reading on:
  - content passes validation
  - tool call detected
  - protection limit exceeded
  - upstream ends before valid content

- [ ] **Step 3: On success, replay buffered chunks and continue streaming**

Implement a combined stream model equivalent to Python:

```rust
buffer.into_iter().chain(remaining_stream)
```

In practice this will be a custom async stream that:
- yields buffered chunks first
- then yields remaining upstream chunks
- closes the `reqwest::Response` cleanly on completion/error

- [ ] **Step 4: Keep Python-compatible error semantics**

If the upstream ends during pre-read without enough content:
- do not commit SSE headers
- return JSON error response
- preserve current `UPSTREAM_STREAM_ERROR` / `429` behavior for the delay-header failure case

- [ ] **Step 5: Add regression tests**

Add tests for:
- empty SSE rejected before headers
- too-short SSE rejected before headers
- large unrecognized SSE passes after protection limit
- long valid SSE with `delay_stream_header=true` still starts before full completion

### Task 3: Make SSE Transformation Incremental

**Files:**
- Modify: `src/streaming.rs`
- Modify: `src/proxy.rs`
- Test: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Replace whole-body `maybe_transform_sse(...)` with an incremental transcoder API**

Current API:

```rust
pub fn maybe_transform_sse(raw: &[u8], upstream_format: Option<RequestFormat>, client_format: Option<RequestFormat>) -> Option<Vec<u8>>
```

Target API shape:

```rust
pub struct StreamTranscoder {
    pending: Vec<u8>,
    // existing state fields...
}

impl StreamTranscoder {
    pub fn new(from_format: RequestFormat, to_format: RequestFormat) -> Self;
    pub fn feed_chunk(&mut self, raw: &[u8]) -> Vec<u8>;
    pub fn flush(&mut self) -> Vec<u8>;
}
```

- [ ] **Step 2: Add carry-over buffering for partial SSE frames**

Do not split each incoming chunk independently. Instead:
- append bytes to `pending`
- extract only complete SSE frames terminated by `\n\n`
- leave incomplete tail bytes in `pending`
- on stream end, process whatever can be finalized in `flush()`

- [ ] **Step 3: Preserve existing internal event mapping**

Keep the current logical mapping in `decode_openai_chat`, `decode_openai_responses`, `decode_claude`, and `decode_gemini`. The change is in framing and feeding, not in event semantics.

- [ ] **Step 4: Integrate incremental transformation into the streaming proxy path**

When `src_format != target_format`, the streaming response body should:
- replay buffered chunks through the transcoder
- continue feeding live upstream chunks through the transcoder
- yield transformed SSE output incrementally
- emit transcoder flush output at end of stream

- [ ] **Step 5: Add boundary-splitting tests**

Add tests where a single SSE frame is intentionally split across multiple chunks. Verify:
- transformed output is correct
- first transformed event arrives before total stream completion
- no frame loss occurs at chunk boundaries

### Task 4: Verification and Deployment

**Files:**
- Modify: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: Add end-to-end timing assertions for both passthrough and transformed streams**

Required cases:
- passthrough SSE, no delay
- passthrough SSE, `delay_stream_header=true`
- transformed SSE, no delay
- transformed SSE, `delay_stream_header=true`

Each should verify:
- first downstream event time << full stream completion time
- stream still finishes correctly with `[DONE]`

- [ ] **Step 2: Re-run the manual reproduction used in debugging**

Manual validation target:
- direct upstream first event near immediate
- via PrismGuard first event still incremental, not end-of-stream

- [ ] **Step 3: Commit only streaming-related Rust changes**

Suggested commit split:

```bash
git add src/proxy.rs tests/http_proxy_stream_tests.rs
git commit -m "fix: stream SSE responses without full buffering"

git add src/streaming.rs tests/http_proxy_stream_tests.rs
git commit -m "fix: transform SSE incrementally across chunk boundaries"
```

- [ ] **Step 4: Deploy with the existing no-local-build workflow**

Use the existing workflow the user required:
- push commit
- wait for GitHub Actions artifact
- download fresh binary with `gh`
- replace local binary
- restart with `supervisorctl`
- verify against live endpoint

## Risks To Watch

- Cross-chunk frame parsing bugs in transformed SSE.
- Accidentally committing SSE headers before delay-header validation completes.
- Resource leaks if the upstream response body is not closed on aborted downstream streams.
- Mixing transformed and untransformed stream code paths in one generator without clear ownership.

## Success Definition

The work is complete only when all four combinations below are true-streaming under live verification:

- passthrough SSE + `delay_stream_header=false`
- passthrough SSE + `delay_stream_header=true`
- transformed SSE + `delay_stream_header=false`
- transformed SSE + `delay_stream_header=true`
