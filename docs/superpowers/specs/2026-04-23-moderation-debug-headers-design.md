# Moderation Debug Headers Design

**Date:** 2026-04-23

## Goal

让 PrismGuard 在所有代理响应上默认返回一组 `x-prismguard-*` 调试 header，用于直接观察智能审核链路里本地模型和 LLM 的决策、置信度与耗时，快速判断“为什么正常请求会回退到 LLM”。

## Problem

当前系统的外部可观测性只覆盖两类情况：

1. 审核明确阻断时，错误体里会带 `moderation_details`。
2. 通过 `py-scripts` 和 debug 路由可以间接推断 profile 配置和本地模型状态。

但对于最关键的冷路径问题，外部仍然看不到：

- 本地模型是否执行了
- 本地模型输出概率是多少
- 本地模型是明确放行/明确阻断，还是因为不确定而回退到 LLM
- LLM 是否实际执行了
- LLM 执行了多久
- LLM 最终是 allow、block，还是 parse error / timeout / other error

这导致性能与行为排查只能靠间接推理，无法从单个请求直接判断根因。

## Scope

本次只做三件事：

1. 在 `smart_moderation` 内部收集本地模型与 LLM 审核调试信息。
2. 在代理响应中默认附加 `x-prismguard-*` header。
3. 覆盖成功响应、审核阻断响应和审核异常响应的 header 输出。

本次不做：

- 新的请求开关或 debug opt-in 机制
- 改动审核判定逻辑
- 修改 profile 配置结构
- 修改发布流程以外的 CI/CD 设计

## Header Contract

采用平铺 header 方案，字段短小、可 grep、适合 curl 与日志观察。

### Local model

- `x-prismguard-local-model-source`
  - 值：`hashlinear_model` / `bow_model` / `fasttext_model` / `none`
- `x-prismguard-local-model-confidence`
  - 本地模型概率，字符串形式输出，保留有限小数位
- `x-prismguard-local-model-latency-ms`
  - 本地模型耗时，毫秒字符串
- `x-prismguard-local-model-decision`
  - 值：`allow` / `block` / `uncertain` / `skipped`

### LLM review

- `x-prismguard-llm-reviewed`
  - 值：`true` / `false`
- `x-prismguard-llm-result`
  - 值：`allow` / `block` / `error`
- `x-prismguard-llm-latency-ms`
  - LLM 审核总耗时，毫秒字符串
- `x-prismguard-llm-error`
  - 仅在 LLM 失败时返回，截断到安全长度

### Presence rules

- Header 只在对应信息存在时返回。
- 如果本地模型未执行，则：
  - 返回 `x-prismguard-local-model-source: none`
  - 返回 `x-prismguard-local-model-decision: skipped`
  - 不返回 confidence / latency
- 如果没有进入 LLM，则：
  - 返回 `x-prismguard-llm-reviewed: false`
  - 不返回 result / latency / error

## Internal Data Model

在 `src/moderation/smart.rs` 内新增一个调试结构，例如 `ModerationDebugInfo`，用于贯穿整个智能审核链路。

推荐字段：

- `local_model_source: Option<String>`
- `local_model_confidence: Option<f64>`
- `local_model_latency_ms: Option<f64>`
- `local_model_decision: Option<LocalModelDecision>`
- `llm_reviewed: bool`
- `llm_result: Option<LlmResult>`
- `llm_latency_ms: Option<f64>`
- `llm_error: Option<String>`

智能审核返回值扩展为“审核结果 + 调试信息”的组合，而不是只返回 `SmartModerationResult`。

## Control Flow

### Entry

在 `proxy.rs` 调用 `smart::smart_moderation(...)` 时，拿到审核结果和调试信息。

### Cache hit

若命中内存 cache：

- 审核结果直接返回
- 调试信息至少标明：
  - local model skipped
  - llm reviewed false
- 本次设计不单独新增 cache-specific header，避免范围膨胀

### Local model path

本地模型执行时：

1. 记录开始时间
2. 调用 `run_local_model`
3. 如果有概率输出：
   - 记录 `source`
   - 记录 `confidence`
   - 根据阈值记录 `decision = allow | block | uncertain`
4. 记录本地模型耗时

注意：即使结果是 `uncertain`，也必须把本地模型概率和耗时传到最终响应头里。

### LLM path

当进入 LLM 审核时：

1. 标记 `llm_reviewed = true`
2. 记录开始时间
3. 在 `llm_moderate` 成功时：
   - 根据结果记录 `llm_result = allow | block`
4. 在 `llm_moderate` 失败时：
   - 记录 `llm_result = error`
   - 记录截断后的 `llm_error`
5. 无论成功失败，都记录 LLM 审核耗时

## Response Integration

在 `proxy.rs` 统一构造一组调试 header，并附加到：

1. 正常成功响应
2. `ApiError::moderation_blocked(...)` 产生的审核阻断响应
3. 审核异常转成的普通错误响应

关键点：

- 当前成功 JSON 路径和流式路径分别经过 `json_success_headers()` / `stream_success_headers()`。
- 需要把这两个点改成可接收“额外审核调试 header”。
- 错误响应也需要统一注入同一组 header，避免只有成功请求能看到调试信息。

## Formatting rules

- 数值 header 输出为短字符串，避免过长。
- `llm_error` 截断，例如最多 160 或 200 字符。
- 不输出 prompt、不输出审核原文、不输出完整上游 JSON。
- Header 名固定小写风格，延续现有 `x-*` 用法。

## Expected debugging outcomes

改动完成后，针对单个请求，用户可以直接判断：

- 本地模型有没有跑
- 本地模型给出的概率是否落在不确定区间
- 是否因此回退到 LLM
- 真正耗时是在本地模型阶段，还是在 LLM 阶段
- LLM 是成功返回，还是 parse error / timeout / other error

对于当前怀疑的问题，预期可以直接观察到类似：

- `x-prismguard-local-model-confidence: 0.43`
- `x-prismguard-local-model-decision: uncertain`
- `x-prismguard-llm-reviewed: true`
- `x-prismguard-llm-result: error`
- `x-prismguard-llm-latency-ms: 3900`

这会把“按理说不该触发 LLM”的怀疑变成可以验证的事实。

## Testing strategy

至少覆盖以下情况：

1. 本地模型明确低风险放行
- 返回 local model source / confidence / latency / decision=allow
- 返回 `llm-reviewed=false`

2. 本地模型明确高风险阻断
- 返回 local model source / confidence / latency / decision=block
- 返回 `llm-reviewed=false`
- 阻断响应也带 header

3. 本地模型不确定，回退到 LLM 且 LLM allow
- 返回 local model confidence + decision=uncertain
- 返回 `llm-reviewed=true` + `llm-result=allow`

4. 本地模型不确定，回退到 LLM 且 LLM block
- 返回 local model confidence + decision=uncertain
- 返回 `llm-reviewed=true` + `llm-result=block`
- 阻断响应也带 header

5. 本地模型不确定，回退到 LLM 且 LLM parse error / timeout
- 返回 local model confidence + decision=uncertain
- 返回 `llm-reviewed=true` + `llm-result=error`
- 返回截断后的 `x-prismguard-llm-error`

6. 命中历史/内存缓存
- 确认不会错误伪造本地模型或 LLM 耗时

## Deployment note

用户要求不在本地编译和测试 Rust，而是修改代码后：

1. 提交代码
2. 通过 `gh` 获取新的二进制
3. 替换本地运行中的二进制
4. 用 `supervisor` 重启服务

这部分属于实现后的发布步骤，不属于本规格的功能设计本体，但实现时必须遵守。
