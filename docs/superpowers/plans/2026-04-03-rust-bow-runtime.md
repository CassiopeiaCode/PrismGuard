# Rust BoW Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Rust 版 `smart_moderation` 增加 `bow` 本地模型运行时路径，保持 Python 既有 `bow_model.pkl` / `bow_vectorizer.pkl` 基础不变，并允许 Rust 追加 `bow_runtime.*` sidecar 完成本地决策。

**Architecture:** 新增 `src/moderation/bow.rs` 负责 runtime 文件加载、缓存与预测；`src/profile.rs` 增加 BoW runtime 路径；`src/moderation/smart.rs` 调整本地模型路由顺序。黑盒测试继续放在 `tests/http_proxy_request_tests.rs`，先验证高风险本地阻断，再扩低风险和回退。

**Tech Stack:** Rust, Serde JSON, Axum black-box tests, reqwest, 进程内 runtime 缓存

---

### Task 1: 定义 BoW runtime 行为并钉住高风险黑盒用例

**Files:**
- Create: `docs/superpowers/specs/2026-04-03-rust-bow-runtime-design.md`
- Modify: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 写失败测试**

新增一个 `smart_moderation_uses_local_bow_runtime_before_llm()` 用例，构造：

- `local_model_type = "bow"`
- 存在 `bow_model.pkl` 与 `bow_vectorizer.pkl` 占位文件
- 存在 `bow_runtime.json` 与 `bow_runtime.coef.f32`
- 发送高风险文本后返回 `400 MODERATION_BLOCKED`
- `moderation_details.source == "bow_model"`

- [ ] **Step 2: 单核运行目标测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests smart_moderation_uses_local_bow_runtime_before_llm -- --nocapture`
Expected: FAIL because Rust 目前没有 `bow` runtime 路径

### Task 2: 实现 BoW runtime 读取与预测

**Files:**
- Create: `src/moderation/bow.rs`
- Modify: `src/moderation/mod.rs`
- Modify: `src/profile.rs`
- Modify: `src/moderation/smart.rs`

- [ ] **Step 1: 新增 BoW runtime 路径与加载缓存**

实现：

- `ModerationProfile::bow_runtime_json_path()`
- `ModerationProfile::bow_runtime_coef_path()`
- `bow::runtime_exists()`
- `bow::predict_proba()`

- [ ] **Step 2: 在 smart_moderation 接入 bow 路由**

本地模型分发顺序：

1. `fasttext` 返回 `None`
2. `hashlinear` 走现有实现
3. `bow` 尝试读取 runtime 并做高低阈值决策

- [ ] **Step 3: 单核运行目标测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests smart_moderation_uses_local_bow_runtime_before_llm -- --nocapture`
Expected: PASS

### Task 3: 补齐最小回归

**Files:**
- Modify: `tests/http_proxy_request_tests.rs`

- [ ] **Step 1: 增加低风险放行或缺失 runtime 回退测试之一**

优先补 `smart_moderation_falls_back_to_llm_when_bow_runtime_is_missing()`，保证 runtime 缺失时不会比 Python 更严格。

- [ ] **Step 2: 单核运行相关 smart_moderation 测试**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test http_proxy_request_tests smart_moderation -- --nocapture`
Expected: PASS
