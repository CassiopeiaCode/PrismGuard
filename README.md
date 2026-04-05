# PrismGuard Rust

这是 PrismGuard 的 Rust 版本代理与智能审核运行时实现，对齐基准是 Python 参考项目 `/services/apps/GuardianBridge-UV`。

本项目的目标不是逐行翻译 Python，而是做到下面这几件事：
- 按 Python 定义实现请求/响应格式转换语义
- 按 Python 行为实现智能审核结果与决策路径
- 保持 Pebble / Rocks 历史库行为兼容
- 在允许内部实现不同的前提下，保证外部结果一致

## 当前状态

当前已经覆盖的核心能力：
- 4 种请求格式：
  - `openai_chat`
  - `openai_responses`
  - `claude_chat`
  - `gemini_chat`
- 智能审核三模型候选行为
- RocksDB/SQLite 历史样本持久化
- 本地模型训练与运行时：
  - `bow`
  - `fasttext`
  - `hashlinear`
- 4 种格式之间的流式 SSE 转换

本次对齐收口的基础提交是：
- `025446e` `Close parity gaps and enable default storage-backed moderation`

## 已与 Python 对齐的部分

### 1. 格式转换

请求转换严格参考 Python 中对格式的定义，尤其是最容易偏掉的部分：
- system / instructions 的映射
- tools 与 tool_choice 的归一化
- OpenAI Responses 请求结构
- 图片字段的多种变体
- function call / function result 的归一化

响应转换目前已经对齐的行为包括：
- OpenAI Responses JSON -> OpenAI Chat JSON
- status / finish_reason 映射
- usage 归一化
- tool call 重建
- image/content 多形态结构处理

### 2. 流式转换

Rust 现在和 Python 一样，走统一的 SSE 转码思路：
- 先把上游 SSE 解码成内部事件
- 再由目标格式专属 sink 重新发射 SSE

当前流式层支持的格式：
- `openai_chat`
- `openai_responses`
- `claude_chat`
- `gemini_chat`

已经重点验证过的方向：
- OpenAI Responses SSE -> OpenAI Chat SSE
- Claude SSE -> OpenAI Chat SSE
- Gemini SSE -> OpenAI Chat SSE
- OpenAI Chat SSE -> OpenAI Responses SSE

流式层还对齐了下面这些 Python 风格细节：
- tool call delta 重组
- 先到参数、后到元信息时的缓冲与回放
- finish_reason 映射
- 只输出一次 `[DONE]`
- 非 200 SSE 透传
- `delay_stream_header` 行为

### 3. 智能审核

Rust 版智能审核已经按 Python 的三段式行为对齐：
- 本地模型先做明显低风险 / 高风险判断
- 不确定区间再走 AI 审核
- `ai_review_rate` 可以强制走 AI 审核

已经对齐的关键点：
- LLM 并发上限：`5`
- 多候选模型重试
- 逗号分隔的模型候选列表解析
- 历史命中优先复用
- AI 审核结果写回历史库

### 4. Pebble / 历史库兼容

Rust 存储层按 Python 项目当前真实使用的历史语义兼容：
- 旧 `history.db` 自动迁移
- RocksDB 历史样本持久化
- `text_latest:*` 指针语义
- balanced / latest / random 训练样本加载
- cleanup 行为与 Python 的重复样本处理保持一致

## 运行说明

### 默认特性

当前默认特性已经开启基于存储的审核路径：

```toml
[features]
default = ["storage-debug"]
```

这意味着默认构建就会包含和 Python 一样的历史复用与落盘逻辑，不再需要额外手动开 feature。

### RocksDB 构建说明

项目当前使用的是 Rust 生态里的 `rocksdb` crate，不是单独手写的 C++ 存储实现。

但这里要明确一点：
- 这**不等于**“已经改成优先链接系统预装 RocksDB 动态库”
- 冷启动构建时，仍可能触发 `librocksdb-sys` 编译，所以第一次编译会比较慢

如果日常开发想避免重复慢编译，建议固定复用同一个 `CARGO_TARGET_DIR`。

## 单核构建 / 测试约定

本项目开发过程严格按单核规则执行。

统一写法：

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --tests -- --nocapture
```

常用定向验证命令：

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_request_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_response_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_stream_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test moderation_runtime_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test pebble_compat_parity_tests -- --nocapture
```

## 当前验证结果

已在单核条件下确认通过：
- `http_proxy_request_tests`: 103 通过
- `http_proxy_response_tests`: 53 通过
- `http_proxy_stream_tests`: 40 通过

本轮对齐收口中，之前已验证通过：
- `moderation_runtime_tests`: 23 通过
- `pebble_compat_parity_tests`: 4 通过
- `scheduler_tests`: 10 通过
- `training_tests`: 42 通过
- `format_process_tests`: 31 通过
- bow / fasttext / hashlinear 运行时 parity 套件通过

## 仓库关键文件

Rust 侧核心文件：
- [src/format.rs](/services/apps/Prismguand-Rust/src/format.rs)：请求识别与请求体转换
- [src/response.rs](/services/apps/Prismguand-Rust/src/response.rs)：非流式 JSON 响应转换
- [src/streaming.rs](/services/apps/Prismguand-Rust/src/streaming.rs)：SSE 流式转码
- [src/moderation/smart.rs](/services/apps/Prismguand-Rust/src/moderation/smart.rs)：智能审核主流程
- [src/profile.rs](/services/apps/Prismguand-Rust/src/profile.rs)：审核与训练配置
- [src/storage.rs](/services/apps/Prismguand-Rust/src/storage.rs)：历史样本存储与迁移
- [src/sample_rpc.rs](/services/apps/Prismguand-Rust/src/sample_rpc.rs)：训练样本 RPC
- [src/training.rs](/services/apps/Prismguand-Rust/src/training.rs)：本地模型训练与运行时流程

Rust 侧关键测试：
- [tests/http_proxy_request_tests.rs](/services/apps/Prismguand-Rust/tests/http_proxy_request_tests.rs)
- [tests/http_proxy_response_tests.rs](/services/apps/Prismguand-Rust/tests/http_proxy_response_tests.rs)
- [tests/http_proxy_stream_tests.rs](/services/apps/Prismguand-Rust/tests/http_proxy_stream_tests.rs)
- [tests/moderation_runtime_tests.rs](/services/apps/Prismguand-Rust/tests/moderation_runtime_tests.rs)
- [tests/pebble_compat_parity_tests.rs](/services/apps/Prismguand-Rust/tests/pebble_compat_parity_tests.rs)

Python 参考文件：
- `/services/apps/GuardianBridge-UV/ai_proxy/transform/formats/openai_responses.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/transform/formats/openai_chat.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/transform/formats/claude_chat.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/transform/formats/gemini_chat.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/proxy/stream_transformer.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/moderation/smart/ai.py`
- `/services/apps/GuardianBridge-UV/ai_proxy/moderation/smart/storage.py`

## 当前仍未承诺的内容

下面这些目前不承诺与 Python 完全一致：
- 逐行实现一致
- 日志输出完全一致
- 异常文案完全一致
- 所有测试 crate 零 warning
- 已切换到系统 RocksDB 动态库链接

当前实际标准是：
- 格式转换定义以 Python 为准
- Pebble / 历史库兼容必须行为正确
- 智能审核必须结果兼容
- Rust 内部实现可以不同，但外部行为必须对齐
