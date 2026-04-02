# Rust HTTP Parity Design

**Date:** 2026-04-03
**Target:** `/services/apps/Prismguand-Rust`
**Reference implementation:** `/services/apps/GuardianBridge-UV`

## Goal

Continue the GuardianBridge Python-to-Rust rewrite with one acceptance criterion above all others: external HTTP behavior must match the Python implementation as closely as possible. Internal implementation details may differ, but request handling, response payloads, status codes, path rewriting, and stream behavior should be aligned with the Python service.

## Non-Goals

- Reproducing Python internals line-for-line
- Large structural refactors before compatibility gaps are covered
- Premature cleanup of Rust modules that are already functional
- Introducing new user-visible behavior beyond parity with Python

## Acceptance Criteria

The Rust service is considered aligned for a covered scenario when:

1. The same incoming request shape produces the same effective upstream request semantics as Python.
2. The same failure mode produces the same HTTP status code and near-identical JSON error structure.
3. The same upstream response shape produces the same downstream non-stream or stream protocol shape.
4. Path rewriting preserves business prefixes in the same way Python does.
5. Compatibility is verified through black-box HTTP tests, not only internal unit tests.

## Required Execution Constraint

This repository has a local Rust environment, but any heavy operation must run inside a constrained `systemd-run` context capped at at most `2x` CPU cores.

This is a hard execution rule for:

- `cargo build`
- `cargo test`
- any heavy integration test runs
- any compile-like or CPU-heavy verification command

Lightweight read-only inspection commands may run normally. The implementation plan must embed this restriction directly into every heavy verification step.

## Current State Summary

The Rust project already covers a subset of the Python proxy behavior:

- URL config parsing exists
- request format detection and request transformation exist
- upstream forwarding exists
- basic streamed passthrough exists

The main gaps against Python are:

- request body decode and JSON fallback behavior are less complete
- request decision logic is narrower than Python
- upstream error passthrough is less precise
- non-stream response conversion is missing or incomplete
- stream protocol conversion and delayed-header validation are not yet parity-complete

## Compatibility Model

The project should optimize for behavioral parity, not structural symmetry. The Rust implementation may keep its current modules, but the end-to-end proxy path must behave like the Python version.

The proxy path is defined as four compatibility stages:

1. Request normalization
2. Request decision and transformation
3. Upstream forwarding
4. Response adaptation

Each stage should expose behavior that is observable from HTTP clients and tests.

## Stage 1: Request Normalization

Responsibilities:

- parse `{config}${upstream}` URLs
- support `!ENV_KEY${upstream}` config loading
- preserve Python-compatible config parse error messages where practical
- read request bodies for supported methods
- handle compressed request bodies for `gzip`, `deflate`, and `br`
- recover from JSON parse failures using Python-compatible fallback behavior
- preserve the original upstream path and prefix information for later rewriting

Required parity details:

- malformed config input should map to `CONFIG_PARSE_ERROR`-style responses
- compressed JSON requests should be accepted when Python accepts them
- unsupported or undecodable content-encoding failures should align with Python-visible error behavior
- requests that Python treats as empty-object or passthrough should not become Rust-only hard failures

## Stage 2: Request Decision And Transformation

Responsibilities:

- execute request format detection with Python-compatible precedence
- enforce `strict_parse` behavior and error strings closely aligned with Python
- honor `from`, `to`, and `auto`
- support `disable_tools` compatibility behavior
- build an internal request plan that captures:
  - source format
  - target format
  - transformed body
  - transformed path
  - stream intent
- surface source format and moderation-style metadata in Python-compatible error envelopes

Required parity details:

- undetected requests in non-strict mode must pass through unchanged
- strict parse failures must produce Python-compatible error semantics
- transformed paths must target the same endpoint families as Python
- source format metadata should be present in error responses when Python includes it

## Stage 3: Upstream Forwarding

Responsibilities:

- filter request headers using Python-compatible deny/allow rules
- force `Accept-Encoding: identity` where Python does
- add Gemini `alt=sse` for streaming Gemini requests
- forward request body in raw or transformed form as required
- preserve upstream path prefixes when a transformed endpoint replaces only the API suffix
- distinguish stream and non-stream upstream handling

Required parity details:

- non-200 upstream responses must be passed through with matching status code and raw body semantics
- content type should be preserved where Python preserves it
- response header filtering should follow Python behavior for security- and transport-related headers
- upstream transport failures should map to Python-style `UPSTREAM_ERROR` or `PROXY_ERROR` envelopes

## Stage 4: Response Adaptation

Responsibilities:

- support non-stream response conversion when source and target formats differ
- support stream protocol conversion between:
  - `openai_chat`
  - `openai_responses`
  - `claude_chat`
  - `gemini_chat`
- preserve pass-through behavior when no conversion is required
- support delayed stream header behavior before committing an SSE response
- surface pre-stream validation failures as normal JSON errors if headers were not yet sent

Required parity details:

- non-stream JSON responses should be convertible from upstream format back to client format
- non-JSON upstream responses should still pass through as Python does
- transformed streams must remain incrementally consumable
- delayed-header validation must fail early in the same class of cases as Python
- once stream headers are committed, behavior must remain streaming and cannot degrade into a late JSON error

## Error Model

Rust should normalize externally visible failures into Python-compatible envelopes:

```json
{
  "error": {
    "code": "SOME_CODE",
    "message": "Human-readable message",
    "type": "some_error_type"
  }
}
```

Additional fields should be included when Python includes them, especially:

- `source_format`
- `moderation_details`

Priority compatibility targets:

1. `CONFIG_PARSE_ERROR`
2. `MODERATION_BLOCKED`
3. `PROXY_ERROR`
4. `UPSTREAM_ERROR`
5. stream pre-read / upstream stream validation failures

Absolute string identity is not required for every message, but status code, top-level error shape, and field presence must be kept as close as practical.

## Path Rewriting Rules

Path rewriting must match Python semantics:

- if the upstream URL is only an API endpoint, replace it directly with the transformed endpoint
- if the upstream URL contains a business prefix plus an API endpoint, retain the prefix and replace only the API endpoint suffix
- if no recognizable API suffix exists, use the transformed path directly

Examples:

- `/v1/messages` -> `/v1/chat/completions`
- `/secret_endpoint/v1/messages` -> `/secret_endpoint/v1/chat/completions`
- `/proxy/google/v1beta/models/gemini-2.5-flash:streamGenerateContent` -> `/proxy/google/v1/chat/completions` when converting Gemini request format to OpenAI Chat

## Recommended Delivery Order

Implementation should proceed in three batches:

### Batch 1: High-Value HTTP Parity

- request normalization parity
- path rewriting parity
- upstream non-200 passthrough
- Python-style error envelope alignment

This batch should maximize compatibility gains for ordinary request/response traffic with minimal structural churn.

### Batch 2: Non-Stream Response Parity

- response body parsing parity
- non-stream format conversion
- non-JSON passthrough behavior

This batch should complete most ordinary API compatibility scenarios before tackling stream complexity.

### Batch 3: Stream Parity

- SSE protocol conversion
- delayed stream header validation
- Gemini streaming quirks
- pre-read failure behavior

This batch is intentionally isolated because it has the highest complexity and regression risk.

## Testing Strategy

Behavioral parity must be validated with black-box tests first, then supported with lower-level unit coverage.

### Test Types

1. Request-processing unit tests
   Cover detection order, strict parse behavior, path rewriting, and transformed request bodies.

2. Proxy-level integration tests with a fake upstream
   Cover:
   - non-200 passthrough
   - non-JSON passthrough
   - request transformation across supported formats
   - non-stream response conversion
   - stream protocol conversion
   - delayed-header validation failures

3. Regression cases copied from Python-observed behavior
   When a Python/Rust mismatch is found, add a test that captures the external HTTP delta before fixing Rust.

### Test Standard

For each compatibility gap:

1. write a failing test against Rust behavior
2. verify the test fails for the expected reason
3. implement the minimal fix
4. re-run the constrained verification command

## Design Decisions

### Decision 1: Favor compatibility wrappers over big refactors

Rust should keep its current module layout unless a boundary actively blocks parity work. The project needs parity first, elegance second.

### Decision 2: Use Python as the behavioral oracle

When Rust behavior and Python behavior differ, Python wins unless the difference is clearly accidental or harmful. The primary objective is replacement compatibility.

### Decision 3: Separate non-stream and stream work

Stream behavior has materially different failure boundaries and output guarantees. It should not be coupled to the simpler non-stream conversion work.

### Decision 4: Preserve black-box observability

Tests should validate what clients actually receive, not only internal request plans. This prevents false confidence from internal equivalence that does not produce the same HTTP behavior.

## Risks

1. Overfitting to current Python bugs
   Some Python behavior may be incidental rather than desirable. For this project, compatibility still takes precedence unless the user explicitly chooses to diverge.

2. Stream complexity
   SSE conversion can fail in subtle ways around framing, buffering, and tool-call deltas. This is why stream parity is deferred to its own batch.

3. Dirty worktree interaction
   The current Rust repository already contains unrelated local changes. Implementation work must avoid reverting or trampling those edits.

4. Verification cost
   Rust compile-and-test cycles can be expensive, so the `systemd-run` CPU cap is mandatory and must be baked into the plan rather than applied ad hoc.

## Open Implementation Constraint

The next step is not implementation. The next step is a concrete implementation plan that maps this design into:

- exact files to modify or create
- exact tests to write first
- exact constrained verification commands
- a batch order that preserves a working tree throughout

That plan should use TDD and assume zero prior context for the executor.
