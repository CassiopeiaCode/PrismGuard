# Rust HTTP 一致性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `/services/apps/Prismguand-Rust` 在客户端可观察的 HTTP 行为上补齐与 `/services/apps/GuardianBridge-UV` 的一致性，重点覆盖请求入口、错误响应、普通响应转换和流式响应转换。

**Architecture:** 保留现有 Rust 模块边界，在 `src/routes.rs`、`src/proxy.rs`、`src/format.rs` 周围补一层兼容语义。实现顺序按“请求入口和透传 -> 非流响应 -> 流响应”推进，所有改动都由黑盒测试或针对性的单元测试先驱动，再用最小实现通过。所有重型验证命令统一通过 `systemd-run --scope -p AllowedCPUs=0,1` 执行，确保最多只占用 2 个核心。

**Tech Stack:** Rust 2021、axum、reqwest、tokio、serde_json、hyper、brotli、flate2

## 执行进展

截至当前计划更新时，HTTP 一致性主链路已经基本完成，计划中的高价值部分已经落地并通过黑盒验证。

当前稳定基线：

- `http_proxy_request_tests`: `21/21`
- `http_proxy_response_tests`: `26/26`
- `http_proxy_stream_tests`: `18/18`

已经稳定实现并锁定的行为包括：

- 请求侧：
  - 压缩 JSON 请求体解码
  - 业务前缀路径保留
  - `!ENV_KEY${upstream}` 配置 URL
  - Gemini 流式 `?alt=sse`
  - 上游非 200 JSON 原样透传
  - `openai_chat -> openai_responses` 的 `response_format`、`max_tokens/max_completion_tokens`、`stream_options.include_usage`、`tool_choice` 等参数归一
- 非流响应：
  - `openai_responses -> openai_chat` 的文本、工具调用、图片、reasoning、usage、finish_reason 和额外字段透传
  - `function_call.arguments` 对象转 JSON 字符串
  - `created` / `usage` 的缺失与 `null` 行为按 Python 基线收紧
- 流响应：
  - 文本 delta
  - 工具调用增量
  - 参数先到后补名字/`output_item.added` 的缓冲回放
  - `response.in_progress` 起始
  - `response.completed/failed/incomplete/error` 的终态映射
  - 最终 chunk 的 usage 规整
  - 缺失 created 时的非零时间戳补齐

当前剩余工作更偏向低收益边角审计，而不是主功能缺失。后续继续沿本计划执行时，应优先补文档和极少数未被黑盒锁住的字段存在性细节，而不是再做大块重构。

---

## 文件结构

本次实现按现有代码边界拆分，避免大规模重构：

- 修改 `src/routes.rs`
  负责 URL 配置解析、统一错误结构、协议感知错误映射、路径改写辅助逻辑暴露。

- 修改 `src/proxy.rs`
  负责请求体读取与解压回退、上游 URL 计算、非 200 透传、非流和流响应分流处理。

- 修改 `src/format.rs`
  负责请求格式识别、严格解析、请求体转换、源格式与目标格式推断细节。

- 新建 `src/response.rs`
  负责非流响应格式转换、协议感知错误响应构造、供 `proxy.rs` 调用。

- 新建 `src/streaming.rs`
  负责 SSE 帧解析、协议间流转换、延迟发送响应头前的预读检查。

- 修改 `src/main.rs`
  注册新增模块。

- 修改 `tests/format_process_tests.rs`
  补请求处理和路径改写单测。

- 新建 `tests/http_proxy_request_tests.rs`
  覆盖请求入口、配置解析、非 200 透传、错误响应协议形状。

- 新建 `tests/http_proxy_response_tests.rs`
  覆盖非流响应转换与非 JSON 透传。

- 新建 `tests/http_proxy_stream_tests.rs`
  覆盖流式协议转换、Gemini `alt=sse`、延迟发送响应头失败场景。

## 统一验证命令约定

本计划中的所有重操作都使用这一前缀：

```bash
systemd-run --scope -p AllowedCPUs=0,1
```

示例：

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test format_process_tests strict_parse_returns_error_when_detection_fails -- --exact --nocapture
```

预期：

- `systemd-run` 正常创建 scope
- `cargo test` 输出单测结果
- 失败时看到明确的断言失败原因
- 通过时看到 `test result: ok`

## Task 1: 请求入口与路径改写一致性

**Files:**
- Modify: `src/routes.rs`
- Modify: `src/proxy.rs`
- Modify: `src/format.rs`
- Modify: `tests/format_process_tests.rs`
- Create: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 先写请求入口与路径改写失败测试**

在 `tests/format_process_tests.rs` 追加以下用例：

```rust
#[test]
fn strict_parse_reports_detected_but_disallowed_format() {
    let config = json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_chat"
        }
    });

    let error = process_request(
        &config,
        "/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .expect_err("strict mode should reject mismatched formats");

    match error {
        RequestProcessError::StrictParse(message) => {
            assert!(message.contains("Format mismatch"));
            assert!(message.contains("claude_chat"));
            assert!(message.contains("openai_chat"));
        }
        other => panic!("expected strict parse error, got {other:?}"),
    }
}

#[test]
fn preserves_business_prefix_when_rewriting_path() {
    let plan = process_request(
        &transform_config(true, "openai_chat"),
        "/secret_endpoint/v1/messages",
        &[("anthropic-version".to_string(), "2023-06-01".to_string())],
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role": "user", "content": "hello"}]
        }),
    )
    .expect("request should transform");

    assert_eq!(plan.path, "/secret_endpoint/v1/chat/completions");
}
```

在 `tests/http_proxy_request_tests.rs` 新建最小黑盒用例：

```rust
#[tokio::test]
async fn compressed_json_request_is_decoded_and_forwarded() {
    let (upstream_url, seen) = spawn_upstream_echo_server().await;
    let app = test_app();
    let payload = gzip_json(json!({
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "hello"}]
    }));

    let request = Request::builder()
        .method("POST")
        .uri(format!("/{cfg}${upstream}/v1/chat/completions",
            cfg = encoded_transform_config("openai_chat"),
            upstream = upstream_url))
        .header("content-encoding", "gzip")
        .header("content-type", "application/json")
        .body(Body::from(payload))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(seen.json_body()["messages"][0]["content"], "hello");
}
```

- [ ] **Step 2: 运行测试，确认它们先失败**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test format_process_tests strict_parse_reports_detected_but_disallowed_format -- --exact --nocapture
```

Expected: FAIL，报错里看不到 `Format mismatch` 或路径仍未保留业务前缀。

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests compressed_json_request_is_decoded_and_forwarded -- --exact --nocapture
```

Expected: FAIL，原因是当前还没有完整的测试 harness 或请求解压后的转发行为不符合预期。

- [ ] **Step 3: 实现最小请求入口和路径改写修复**

在 `src/format.rs` 中补齐严格解析和已识别但不允许格式的报错路径，核心修改应接近以下结构：

```rust
if parsed.is_none() && strict_parse {
    let excluded = all_request_formats()
        .into_iter()
        .filter(|format| !candidates.contains(format))
        .collect::<Vec<_>>();
    let detectable_excluded = detect_formats_from_candidates(&excluded, path, headers, &plan.body);

    let mut message = if !detectable_excluded.is_empty() {
        let expected = expected_formats_label(from_cfg, &candidates);
        let detected = detectable_excluded
            .iter()
            .map(|format| format!("'{}'", format.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Format mismatch: Request appears to be in format [{detected}], but only [{expected}] is allowed."
        )
    } else {
        let expected = expected_formats_label(from_cfg, &candidates);
        format!(
            "Unable to parse request format. Expected format: {expected}. Please verify your request body structure matches the expected format."
        )
    };

    if !parse_errors.is_empty() {
        message.push_str(&format!(" Parse errors: {}", parse_errors.join("; ")));
    }

    return Err(RequestProcessError::StrictParse(message));
}
```

在 `src/proxy.rs` 中补一个可复用的路径改写函数，而不是直接拼接：

```rust
fn merge_transformed_path(original_path: &str, transformed_path: Option<&str>) -> String {
    let Some(transformed_path) = transformed_path else {
        return original_path.to_string();
    };

    let api_markers = [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1beta/models/",
    ];

    if let Some(marker_pos) = api_markers.iter().filter_map(|marker| original_path.find(marker)).min() {
        let prefix = &original_path[..marker_pos];
        return format!("{prefix}{transformed_path}");
    }

    transformed_path.to_string()
}
```

并在代理入口中用它替换 `final_url` 的路径计算。

- [ ] **Step 4: 跑针对性测试，确认通过**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test format_process_tests strict_parse_reports_detected_but_disallowed_format -- --exact --nocapture
```

Expected: PASS

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test format_process_tests preserves_business_prefix_when_rewriting_path -- --exact --nocapture
```

Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/format.rs src/proxy.rs tests/format_process_tests.rs tests/http_proxy_request_tests.rs
git commit -m "test: cover request normalization parity"
```

## Task 2: 错误响应映射与非 200 透传

**Files:**
- Modify: `src/routes.rs`
- Modify: `src/proxy.rs`
- Create: `src/response.rs`
- Modify: `src/main.rs`
- Modify: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 先写失败测试，锁定错误协议和非 200 透传**

在 `tests/http_proxy_request_tests.rs` 中加入：

```rust
#[tokio::test]
async fn upstream_non_200_is_passed_through_without_json_rewrap() {
    let (upstream_url, _) = spawn_upstream_fixed_response(
        StatusCode::TOO_MANY_REQUESTS,
        vec![("content-type", "application/json")],
        br#"{"error":{"message":"busy"}}"#,
    ).await;

    let app = test_app();
    let request = json_proxy_request(
        &app,
        upstream_url,
        json!({"model":"gpt-4.1-mini","messages":[{"role":"user","content":"hello"}]}),
    );

    let response = request.await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["content-type"], "application/json");
    let body = body_json(response).await;
    assert_eq!(body["error"]["message"], "busy");
}

#[tokio::test]
async fn strict_parse_error_uses_openai_style_error_envelope() {
    let app = test_app_with_config(json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_chat"
        }
    }));

    let response = send_proxy_request(
        &app,
        "/v1/messages",
        json!({
            "model": "claude-sonnet-4-5",
            "anthropic_version": "2023-06-01",
            "messages": [{"role":"user","content":"hello"}]
        }),
    ).await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert!(body.get("error").is_some());
    assert!(body["error"]["message"].as_str().unwrap().contains("Format mismatch"));
}
```

- [ ] **Step 2: 运行测试，确认当前会失败**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests upstream_non_200_is_passed_through_without_json_rewrap -- --exact --nocapture
```

Expected: FAIL，当前实现可能会重新包装错误或遗漏 header 语义。

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests strict_parse_error_uses_openai_style_error_envelope -- --exact --nocapture
```

Expected: FAIL，当前严格解析和错误码映射尚未按协议感知拆开。

- [ ] **Step 3: 实现最小错误映射和透传**

在 `src/response.rs` 新建协议感知错误辅助函数：

```rust
use axum::http::StatusCode;
use serde_json::{json, Value};

pub fn openai_error(status: StatusCode, code: &str, message: impl Into<String>, error_type: &str) -> (StatusCode, Value) {
    (
        status,
        json!({
            "error": {
                "code": code,
                "message": message.into(),
                "type": error_type
            }
        }),
    )
}
```

在 `src/routes.rs` 中把 `ApiError::into_response` 改成通过协议感知构造，而不是把所有错误都固定为同一 Rust 内部结构。

在 `src/proxy.rs` 中保持非 200 上游响应“状态码 + 原始 body + 过滤后的 header”透传：

```rust
if !status.is_success() {
    let body = upstream_response.bytes().await.map_err(ApiError::from)?;
    return build_response(status, &headers, boxed(Body::from(body)));
}
```

- [ ] **Step 4: 运行请求级测试，确认通过**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests upstream_non_200_is_passed_through_without_json_rewrap -- --exact --nocapture
```

Expected: PASS

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests strict_parse_error_uses_openai_style_error_envelope -- --exact --nocapture
```

Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/main.rs src/routes.rs src/proxy.rs src/response.rs tests/http_proxy_request_tests.rs
git commit -m "feat: align proxy error envelopes and passthrough"
```

## Task 3: 非流式响应转换与非 JSON 透传

**Files:**
- Modify: `src/proxy.rs`
- Modify: `src/format.rs`
- Modify: `src/response.rs`
- Create: `tests/http_proxy_response_tests.rs`

- [ ] **Step 1: 先写失败测试**

在 `tests/http_proxy_response_tests.rs` 中新建：

```rust
#[tokio::test]
async fn upstream_openai_responses_json_is_transformed_back_to_openai_chat() {
    let upstream_body = json!({
        "id": "resp_123",
        "object": "response",
        "model": "gpt-4.1-mini",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "hello"
            }]
        }]
    });

    let (upstream_url, _) = spawn_upstream_json(StatusCode::OK, upstream_body).await;
    let app = test_app_with_config(json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }));

    let response = send_proxy_request(
        &app,
        &format!("{upstream_url}/v1/chat/completions"),
        json!({
            "model":"gpt-4.1-mini",
            "messages":[{"role":"user","content":"hi"}]
        }),
    ).await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hello");
}

#[tokio::test]
async fn non_json_success_response_is_passed_through_as_text_body() {
    let (upstream_url, _) = spawn_upstream_fixed_response(
        StatusCode::OK,
        vec![("content-type", "text/plain; charset=utf-8")],
        b"plain text body",
    ).await;

    let app = test_app();
    let response = send_proxy_request_raw(&app, &format!("{upstream_url}/v1/chat/completions")).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(body_text(response).await, "plain text body");
}
```

- [ ] **Step 2: 运行测试，确认先失败**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_response_tests upstream_openai_responses_json_is_transformed_back_to_openai_chat -- --exact --nocapture
```

Expected: FAIL，当前 Rust 还没有完整的非流响应反向转换链路。

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_response_tests non_json_success_response_is_passed_through_as_text_body -- --exact --nocapture
```

Expected: FAIL，当前响应路径可能没有稳定区分 JSON 与纯文本返回。

- [ ] **Step 3: 实现最小非流转换能力**

在 `src/response.rs` 中新增：

```rust
pub fn maybe_transform_json_response(
    body: Value,
    upstream_format: Option<RequestFormat>,
    client_format: Option<RequestFormat>,
) -> Result<Value, ApiError> {
    match (upstream_format, client_format) {
        (Some(from), Some(to)) if from != to => transform_response_body(from, to, body),
        _ => Ok(body),
    }
}
```

在 `src/proxy.rs` 的非流分支中改成：

```rust
let bytes = upstream_response.bytes().await.map_err(ApiError::from)?;
if let Ok(json_body) = serde_json::from_slice::<Value>(&bytes) {
    let adapted = maybe_transform_json_response(
        json_body,
        request_plan.target_format,
        request_plan.source_format,
    )?;
    let encoded = serde_json::to_vec(&adapted).map_err(ApiError::from)?;
    return build_response(status, &headers, boxed(Body::from(encoded)));
}

return build_response(status, &headers, boxed(Body::from(bytes)));
```

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_response_tests upstream_openai_responses_json_is_transformed_back_to_openai_chat -- --exact --nocapture
```

Expected: PASS

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_response_tests non_json_success_response_is_passed_through_as_text_body -- --exact --nocapture
```

Expected: PASS

- [ ] **Step 5: 提交这一小步**

```bash
git add src/proxy.rs src/response.rs tests/http_proxy_response_tests.rs
git commit -m "feat: add non-stream response parity"
```

## Task 4: 流式协议转换与延迟响应头

**Files:**
- Create: `src/streaming.rs`
- Modify: `src/proxy.rs`
- Modify: `src/response.rs`
- Create: `tests/http_proxy_stream_tests.rs`

- [ ] **Step 1: 先写失败测试**

在 `tests/http_proxy_stream_tests.rs` 中加入两个核心场景：

```rust
#[tokio::test]
async fn openai_responses_sse_is_transformed_back_to_openai_chat_sse() {
    let upstream_sse = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
        "event: response.output_text.delta\n",
        "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"item_id\":\"msg_1\",\"content_index\":0,\"delta\":\"hello\"}\n\n",
        "data: [DONE]\n\n"
    );

    let (upstream_url, _) = spawn_upstream_sse(upstream_sse).await;
    let app = test_app_with_config(json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses"
        }
    }));

    let response = send_stream_proxy_request(&app, upstream_url).await;
    let text = body_text(response).await;
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("\"content\":\"hello\""));
    assert!(text.contains("[DONE]"));
}

#[tokio::test]
async fn delayed_stream_header_returns_json_error_before_committing_stream() {
    let (upstream_url, _) = spawn_upstream_sse("").await;
    let app = test_app_with_config(json!({
        "format_transform": {
            "enabled": true,
            "strict_parse": true,
            "from": "openai_chat",
            "to": "openai_responses",
            "delay_stream_header": true
        }
    }));

    let response = send_stream_proxy_request(&app, upstream_url).await;
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = body_json(response).await;
    assert!(body["error"]["message"].as_str().unwrap().contains("Stream"));
}
```

- [ ] **Step 2: 运行测试，确认先失败**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests openai_responses_sse_is_transformed_back_to_openai_chat_sse -- --exact --nocapture
```

Expected: FAIL，当前只支持流透传，不支持协议转换。

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests delayed_stream_header_returns_json_error_before_committing_stream -- --exact --nocapture
```

Expected: FAIL，当前还没有预读检查和“发头前失败”的稳定处理。

- [ ] **Step 3: 实现最小流式转换器和预读**

在 `src/streaming.rs` 中实现最小 SSE 帧解析器与 OpenAI Responses -> OpenAI Chat 适配器：

```rust
pub struct SseFrame {
    pub event: Option<String>,
    pub data: String,
}

pub struct SseBuffer {
    buffer: String,
}

impl SseBuffer {
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<SseFrame> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut out = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let raw = self.buffer[..idx].to_string();
            self.buffer = self.buffer[idx + 2..].to_string();
            if let Some(frame) = parse_sse_frame(&raw) {
                out.push(frame);
            }
        }
        out
    }
}
```

在 `src/proxy.rs` 中把流式分支改为：

```rust
if is_stream_response(upstream_response.headers()) {
    return build_streaming_proxy_response(
        upstream_response,
        request_plan.source_format,
        request_plan.target_format,
        parsed.config.get("format_transform")
            .and_then(|v| v.get("delay_stream_header"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    ).await;
}
```

并在 `build_streaming_proxy_response` 中支持：

- 无转换时直接透传
- 需要转换时走 `streaming.rs`
- `delay_stream_header=true` 时先预读若干 chunk
- 如果预读阶段确认失败，则在 header 未发出前返回普通 JSON 错误

- [ ] **Step 4: 运行测试，确认通过**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests openai_responses_sse_is_transformed_back_to_openai_chat_sse -- --exact --nocapture
```

Expected: PASS

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests delayed_stream_header_returns_json_error_before_committing_stream -- --exact --nocapture
```

Expected: PASS

- [ ] **Step 5: 跑本批完整测试**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests -- --nocapture
```

Expected: `test result: ok`

- [ ] **Step 6: 提交这一小步**

```bash
git add src/proxy.rs src/response.rs src/streaming.rs tests/http_proxy_stream_tests.rs
git commit -m "feat: add streaming protocol parity"
```

## Task 5: 收口验证

**Files:**
- Modify: `docs/superpowers/specs/2026-04-03-rust-http-parity-design.md`
- Modify: `docs/superpowers/plans/2026-04-03-rust-http-parity.md`

- [ ] **Step 1: 跑当前批次下的全部测试**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test format_process_tests -- --nocapture
```

Expected: `test result: ok`

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_request_tests -- --nocapture
```

Expected: `test result: ok`

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_response_tests -- --nocapture
```

Expected: `test result: ok`

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test --test http_proxy_stream_tests -- --nocapture
```

Expected: `test result: ok`

- [ ] **Step 2: 跑受限全量验证**

Run:

```bash
systemd-run --scope -p AllowedCPUs=0,1 cargo test -- --nocapture
```

Expected: `test result: ok`

- [ ] **Step 3: 记录实现状态并提交**

```bash
git add docs/superpowers/specs/2026-04-03-rust-http-parity-design.md docs/superpowers/plans/2026-04-03-rust-http-parity.md
git commit -m "docs: record rust parity implementation plan"
```

## 自检

### Spec 覆盖检查

- 请求规范化：Task 1 覆盖
- 请求决策与严格解析：Task 1、Task 2 覆盖
- 错误协议映射：Task 2 覆盖
- 非流响应转换：Task 3 覆盖
- 流式协议转换与延迟 header：Task 4 覆盖
- 受限 `systemd-run` 验证约束：Task 1-5 全部显式覆盖

### 占位符检查

- 无 `TODO` / `TBD`
- 每个任务都包含明确文件、测试和命令
- 每个重操作都给出受限执行命令

### 一致性检查

- 任务顺序与设计文档的三批交付顺序一致
- 错误响应要求已放宽为“协议正确”，未再要求逐字复刻 Python
- 后续执行仅在测试先失败后实现，符合 TDD
