# Rust fastText Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Rust 版 `smart_moderation` 增加 `fasttext` 本地模型运行时支持，并保持缺失或损坏时自动回退 LLM。

**Architecture:** 参考刚完成的 `bow` 方案，新增 `src/moderation/fasttext.rs` 读取 `fasttext_runtime.json` sidecar，`smart.rs` 把 `fasttext` 作为本地模型优先级最高的分支处理。黑盒测试继续放在 `tests/http_proxy_request_tests.rs`。

**Tech Stack:** Rust, Serde JSON, 进程内 runtime 缓存, Axum black-box tests

---

### Task 1: 写失败测试锁定 fastText 路由优先级

**Files:**
- Modify: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 新增高风险本地阻断用例**

测试名建议：

- `smart_moderation_uses_local_fasttext_runtime_before_llm`

要求：

- `local_model_type = "fasttext"`
- `fasttext_model.bin` 作为 Python 基线占位文件存在
- `fasttext_runtime.json` 存在
- 触发本地高风险后返回 `400 MODERATION_BLOCKED`
- `moderation_details.source == "fasttext_model"`

- [ ] **Step 2: 单核运行目标测试，确认 red**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests smart_moderation_uses_local_fasttext_runtime_before_llm -- --nocapture`
Expected: FAIL because Rust 目前没有 fastText runtime 路由

### Task 2: 实现 fastText runtime

**Files:**
- Create: `src/moderation/fasttext.rs`
- Modify: `src/moderation/mod.rs`
- Modify: `src/profile.rs`
- Modify: `src/moderation/smart.rs`

- [ ] **Step 1: 增加 profile 路径辅助函数**

实现：

- `fasttext_runtime_json_path()`

- [ ] **Step 2: 实现 runtime 加载与预测**

提供：

- `fasttext::predict_proba()`
- 文件签名缓存
- 最小 token 权重线性推理

- [ ] **Step 3: 接入 smart_moderation 本地路由最前位**

顺序变为：

1. `fasttext`
2. `hashlinear`
3. `bow`

- [ ] **Step 4: 单核运行目标测试，确认 green**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests smart_moderation_uses_local_fasttext_runtime_before_llm -- --nocapture`
Expected: PASS

### Task 3: 补最小回退回归

**Files:**
- Modify: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 新增 runtime 缺失回退 LLM 用例**

测试名建议：

- `smart_moderation_falls_back_to_llm_when_fasttext_runtime_is_missing`

- [ ] **Step 2: 单核运行 fasttext 子集**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests fasttext_runtime -- --nocapture`
Expected: PASS
