# PrismGuard

**企业级 AI API 智能中间件** - 内容审核 · 格式转换 · 透明代理

PrismGuard 是一个专为 AI API 设计的智能中间件，提供三段式内容审核、多格式自动转换和透明代理能力。它的目标是：**在不修改现有客户端代码的前提下**，为你的 AI 调用链路增加审核、转换与可靠性保障。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 目录

- [你能用它做什么（Use Cases）](#你能用它做什么use-cases)
- [快速开始（5 分钟跑起来）](#快速开始5-分钟跑起来)
- [核心概念（先理解再配置）](#核心概念先理解再配置)
- [URL 配置（最重要）](#url-配置最重要)
- [审核系统（Moderation）](#审核系统moderation)
- [格式转换系统（Transform）](#格式转换系统transform)
- [运维与安全（Production）](#运维与安全production)
- [工具与脚本（Tools）](#工具与脚本tools)
- [FAQ / Troubleshooting](#faq--troubleshooting)
- [License](#license)
- [致谢](#致谢)

---

## 你能用它做什么（Use Cases）

### 1) 零代码接入：只改 base_url

通过在 URL 中携带配置和上游地址，你可以让现有 SDK（例如 OpenAI SDK）**零改动**接入 PrismGuard。

```python
from openai import OpenAI

# 使用预定义配置（从环境变量读取）
base_url = "http://localhost:8000/!PROXY_CONFIG_DEFAULT$https://api.openai.com/v1"

client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
print(response)
```

### 2) 仅做内容审核：关键词 + 智能审核（不转换格式）

最常见：你想要审核输入内容，但不改变请求格式与上游端点。

- 基础审核：关键词黑名单（支持热重载）
- 智能审核：三段式（缓存/抽样 AI/本地模型/AI复核）+ 增量学习

### 3) 跨厂商格式互转：OpenAI ↔ Claude ↔ Gemini ↔ OpenAI Responses

PrismGuard 会自动检测格式并转为内部格式，完成审核后再转成目标格式发往上游。

支持格式：

| 格式 | 说明 |
|------|------|
| `openai_chat` | OpenAI Chat Completions API |
| `claude_chat` | Anthropic Messages API |
| `gemini_chat` | Google Gemini API |
| `openai_responses` | OpenAI Responses API |

### 4) 工具调用治理：可禁用/拒绝工具请求

如果你不希望用户请求中包含 tools / tool_choice / tool calls / tool results，可以使用 `disable_tools` 直接拒绝此类请求（适用于高风险场景、或只允许纯文本聊天的业务）。

### 5) 流式可靠性：防空回复、兼容 Gemini SSE

开启 `delay_stream_header` 后，PrismGuard 会在发送响应头前预读流式内容，直到满足：

- 已累计文本内容长度 > 2；或
- 检测到工具调用

以降低“空回复/断流导致客户端误判成功”的概率。

---

## 快速开始（5 分钟跑起来）

### 1) 安装

```bash
# 克隆项目
git clone <repository-url>
cd PrismGuard

# 安装依赖（推荐使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 2) 配置

```bash
# 复制配置文件
cp .env.example .env
cp -r configs.example configs

# 编辑 .env，设置 AI 审核 API Key（示例为 OpenAI）
# MOD_AI_API_KEY=sk-your-openai-api-key
```

### 3) 启动

```bash
# 开发模式
python -m ai_proxy.app

# 生产模式
bash start.sh
```

服务默认启动在：`http://localhost:8000`

### 4) 最小可用示例（复制即可跑）

#### A. 仅审核（不过格式）

```python
from openai import OpenAI

base_url = "http://localhost:8000/!PROXY_CONFIG_DEFAULT$https://api.openai.com/v1"
client = OpenAI(api_key="sk-xxx", base_url=base_url)

resp = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp)
```

#### B. OpenAI → Claude（用 OpenAI SDK 调 Claude）

```python
from openai import OpenAI

base_url = "http://localhost:8000/!PROXY_CONFIG_CLAUDE$https://api.anthropic.com/v1"

client = OpenAI(
    api_key="sk-ant-xxx",
    base_url=base_url
)

resp = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp)
```

#### C. OpenAI → Gemini（用 OpenAI SDK 调 Gemini）

```python
from openai import OpenAI

base_url = "http://localhost:8000/!PROXY_CONFIG_GEMINI$https://generativelanguage.googleapis.com"

client = OpenAI(
    api_key="your-gemini-api-key",
    base_url=base_url
)

resp = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp)
```

更多示例请参见 `examples/`。

**示例索引（建议从这里开始）**：

- `examples/client_example.py`：3 种接入方式（URL 编码 / 环境变量 / OpenAI→Claude）
- `examples/client_usage.py`：演示工具调用、基础审核与格式转换（需要自行填 API Key）
- `examples/config_examples.py`：常用 URL 配置生成示例（适合复制到 `.env` 或做临时测试）
- `examples/stream_delay_header.py`：流式请求 + `delay_stream_header=true` 防空回复（将新增）
- `examples/disable_tools_demo.py`：`disable_tools=true` 的拒绝效果与报错示例（将新增）
- `examples/responses_stream_transform.py`：`openai_chat` ↔ `openai_responses` 流式互转（将新增）

---

## 核心概念（先理解再配置）

### 1) 请求处理链路（高层）

```
客户端请求
    ↓
URL 解析（配置 + 上游地址）
    ↓
格式检测与转换（可选）
    ↓
文本抽取
    ↓
基础审核（关键词）
    ↓
智能审核（三段式，可选）
    ↓
转发到上游 API（可选：响应转换 / 流式检查）
    ↓
返回响应
```

### 2) Profile：一套审核策略 = 一个目录

每个审核配置称为一个 Profile，位于：

```
configs/mod_profiles/{profile}/
├── profile.json         # 审核参数配置
├── ai_prompt.txt        # AI 审核提示词
├── history.db           # 审核历史数据库（自动生成）
├── hashlinear_model.pkl # HashLinear 模型（主推，本地审核默认推荐）
├── bow_model.pkl        # BoW 模型（可选）
├── bow_vectorizer.pkl   # BoW 向量化器（可选）
└── fasttext_model.bin   # fastText 模型（可选）
```

### 3) “内部格式”（InternalChatRequest）

PrismGuard 会将不同厂商的请求转为统一的内部格式，以便：

- 用同一套抽取逻辑拿到待审核文本（避免把工具参数/工具结果误当成用户输入）
- 在不同厂商之间做格式互转
- 在需要时对响应做转换（部分格式支持）

---

## URL 配置（最重要）

### 1) URL 结构

代理入口格式：

- **内联 JSON 配置**：`/{url_encoded_json_config}${upstream_full_url}`
- **环境变量配置**：`/!{ENV_KEY}${upstream_full_url}`

其中 `upstream_full_url` 是完整上游 URL（含 path），例如：

- `https://api.openai.com/v1/chat/completions`
- `https://api.anthropic.com/v1/messages`
- `https://generativelanguage.googleapis.com/v1beta/models/...:generateContent`

### 2) 预定义配置（环境变量）

在 `.env` 中定义配置，缩短 URL：

```bash
# 仅审核，不转换格式
PROXY_CONFIG_DEFAULT={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true,"profile":"default"},"format_transform":{"enabled":false}}

# OpenAI → Claude 转换
PROXY_CONFIG_CLAUDE={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true},"format_transform":{"enabled":true,"from":"openai_chat","to":"claude_chat"}}

# OpenAI → Gemini 转换
PROXY_CONFIG_GEMINI={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true},"format_transform":{"enabled":true,"from":"openai_chat","to":"gemini_chat"}}
```

### 3) 配置字段速查

#### `basic_moderation`

```json
{
  "enabled": true,
  "keywords_file": "configs/keywords.txt",
  "error_code": "BASIC_MODERATION_BLOCKED"
}
```

- `keywords_file`：关键词文件路径，支持热重载
- `error_code`：命中关键词时返回的错误码（会拼到 message 中）

#### `smart_moderation`

```json
{
  "enabled": true,
  "profile": "default"
}
```

- `profile`：对应 `configs/mod_profiles/{profile}`
- Profile 内的 `ai.model` 支持填写逗号分隔的多个模型，例如 `gpt-4o-mini, gpt-4.1-mini`
- Profile 内的 `ai.timeout` 表示 LLM 单次调用超时时间（秒），可选，默认 `10`
- Profile 内的 `ai.max_retries` 表示最大重试次数，可选，默认 `2`

#### `format_transform`

```json
{
  "enabled": true,
  "from": "auto",
  "to": "openai_chat",
  "strict_parse": false,
  "disable_tools": false,
  "delay_stream_header": false
}
```

- `enabled`：是否启用格式转换/检测
- `from`：来源格式（`auto` / 字符串 / 字符串数组）
- `to`：目标格式（不变则相当于只做检测+审核）
- `strict_parse`：
  - `true`：无法识别格式时直接报错（提示期望格式）
  - `false`：无法识别则透传（不做转换）
- `disable_tools`：禁用工具调用；检测到 tools/tool_choice/tool calls/tool results 会拒绝请求
- `delay_stream_header`：流式响应头延迟放行，防空回复（对非流式也会做内容检查）

### 4) 常用组合（Recipes）

- **只审核不过格式**：`format_transform.enabled=false`
- **审核 + 格式互转**：`format_transform.enabled=true` 且设置 `from/to`
- **只转换不审核**：关闭 `basic_moderation.enabled` 和 `smart_moderation.enabled`
- **严格接入（防透传）**：`strict_parse=true`
- **禁用工具调用**：`disable_tools=true`
- **流式空回复防护**：`delay_stream_header=true`

---

## 审核系统（Moderation）

### 1) 基础审核：关键词过滤

- 关键词文件：默认 `configs/keywords.txt`
- 支持热重载（修改文件后无需重启服务）
- 内部有缓存上限，避免无限增长

### 2) 智能审核：三段式决策 + 缓存

流程概述：

1. LRU 缓存命中：直接返回
2. 随机抽样：按 `ai_review_rate` 直接走 AI 审核并记录（用于持续产生标注）
3. 本地模型：HashLinear / BoW / fastText 预测违规概率
   - `p < low_risk_threshold`：低风险放行
   - `p > high_risk_threshold`：高风险拒绝
   - 中间区域：交给 AI 复核
4. 无本地模型或异常：回退 AI 审核
5. 结果写入缓存

### 3) 增量学习闭环：样本入库 + 训练数据管理

- AI 审核结果会写入 `history.db`（SQLite）
- 会先查库去重（同文本不重复请求 AI）
- 训练数据加载支持 3 种策略（由 profile 配置决定）：
  - `balanced_undersample`：欠采样随机 1:1 平衡；每类取 `min(少数类总数, max_samples/2)`，合并后打乱（实现见 [`load_balanced_samples()`](ai_proxy/moderation/smart/storage.py:180)）
  - `latest_full`：full-最新（不强制 1:1）；每类最多取 `max_samples/2` 个“最新样本”，合并后打乱（实现见 [`load_balanced_latest_samples()`](ai_proxy/moderation/smart/storage.py:221)）
  - `random_full`：full-随机（不强制 1:1）；每类最多取 `max_samples/2` 个“随机样本”，合并后打乱（实现见 [`load_balanced_random_samples()`](ai_proxy/moderation/smart/storage.py:250)）
- full 模式的核心约束：**每类都有独立上限 `max_samples/2`，不会被另一类挤占**；因此当分布为 正=25/负=75 且 `max_samples=100` 时，训练集会是 `25 正 + 50 负 = 75`（不会去补满到 100）。
- 数据库可按上限清理并 VACUUM 释放空间

### 4) 本地模型：优先推荐 HashLinear

- `local_model_type = "hashlinear"`：主推方案，模型小、常量内存、推理稳定，适合作为线上默认选择
- HashLinear 可配合 Rust 扩展加速推理；扩展不可用时会自动回退到 Python 实现
- `local_model_type = "bow"`：兼容型备选，便于已有 BoW 数据继续沿用
- `local_model_type = "fasttext"`：轻量备选，但需要关注 NumPy / fastText 兼容性
- fastText 分词支持（在 `configs/mod_profiles/{profile}/profile.json` 中配置）：
  - `fasttext_training.use_jieba=true`：使用 jieba 分词（中文推荐）
  - `fasttext_training.use_tiktoken=true`：使用 tiktoken 分词（实验性）
  - 两者都为 `false`：使用字符级 n-gram（原版 fastText 路径）
  - 两者都为 `true`：先 tiktoken 再 jieba（实验性组合）

推荐配置片段（HashLinear）：

```json
{
  "local_model_type": "hashlinear",
  "hashlinear_training": {
    "sample_loading": "balanced_undersample",
    "analyzer": "char",
    "ngram_range": [2, 4],
    "n_features": 1048576
  }
}
```

fastText 配置片段（仅在需要时启用）：

```json
{
  "local_model_type": "fasttext",
  "fasttext_training": {
    "sample_loading": "balanced_undersample",
    "use_jieba": true,
    "use_tiktoken": false,
    "tiktoken_model": "cl100k_base"
  }
}
```

> 说明：训练与预测会根据上述开关自动选择对应实现（参见 [`train_local_model()`](ai_proxy/moderation/smart/scheduler.py:42) 与 [`local_model_predict_proba()`](ai_proxy/moderation/smart/ai.py:20)）。

> 注意：若使用 fastText，建议遵循项目的依赖检查提示（例如 `numpy<2.0`）。

AI 审核配置片段示例：

```json
{
  "ai": {
    "model": "gpt-4o-mini, gpt-4.1-mini",
    "timeout": 10,
    "max_retries": 2
  }
}
```

- 当 `model` 配置多个模型时，审核会在每次尝试时随机选择一个模型调用
- 当一次调用超时或报错时，会在 `max_retries` 范围内继续重试
- 上述字段缺省时会自动使用默认值，不会因为旧配置缺字段而报错

---

## 格式转换系统（Transform）

### 1) 自动检测与 strict/透传策略

- `from="auto"`：按已注册解析器顺序检测
- `strict_parse=true`：无法识别则报错（便于强约束接入）
- `strict_parse=false`：无法识别则透传（兼容更多请求，但风险更高）

### 2) 路径与流式：目标端点如何决定

- OpenAI Chat：`/v1/chat/completions`
- Claude Messages：`/v1/messages`
- OpenAI Responses：`/v1/responses`
- Gemini：`/v1beta/models/{model}:generateContent` 或 `:streamGenerateContent`（流式）

### 3) 流式响应互转：OpenAI Chat ↔ OpenAI Responses

当前内置的流式转换器主要覆盖：

- `openai_responses` ↔ `openai_chat`

其它格式的流式互转如需支持，可在转换器层继续扩展。

---

## 运维与安全（Production）

### 1) 环境依赖检查（启动即检查）

应用启动时会进行关键依赖检查（例如 NumPy / fastText 兼容性）。若检查失败，启动会被阻止并给出可操作的修复建议。

### 2) 性能指标（参考）

| 操作 | 耗时 |
|------|------|
| 关键词过滤 | <1ms |
| HashLinear 推理 | 1-3ms |
| HashLinear 推理（Rust） | 更低延迟，取决于机器与构建方式 |
| AI 审核 | 500-2000ms |
| 格式转换 | <2ms |
| 缓存命中 | <0.1ms |

**资源占用**：
- 内存：100-200 MB
- CPU：单核可运行
- 磁盘：模型 5-10 MB，每 1000 条样本约 1 MB

### 3) 安全建议

1. **API Key 管理**：使用环境变量，不要提交 `.env` 到版本控制
2. **访问控制**：生产环境使用反向代理，配置 IP 白名单
3. **数据隐私**：审核历史包含用户输入，定期清理过期数据
4. **监控告警**：定期检查误判率，调整阈值

### 4) RocksDB 多 worker（避免 LOCK 竞争）

PrismGuard 的审核样本库使用 RocksDB（`history.rocks/`）。RocksDB 在**同一 DB 路径**上只允许一个进程持有写锁，否则会出现：

`IO error: While lock file: <profile>/history.rocks/LOCK: Resource temporarily unavailable`

为彻底避免多进程争抢写锁，本项目在多 worker 部署时采用 **“单 writer + 其它 worker 走 IPC 代理”**：

- 仅一个 worker 会以 read-write 打开 RocksDB，并启动本地 IPC 服务
- 其它 worker **不会本地 open RocksDB**，所有读/写都转发给 writer（因此不会触发 LOCK 竞争）
- 模型训练调度器也只在 writer worker 启动，避免重复调度

可选环境变量：

- `AI_PROXY_ROCKS_IPC_HOST`（默认 `127.0.0.1`）
- `AI_PROXY_ROCKS_IPC_PORT`（默认 `51234`）
- `AI_PROXY_ROCKS_WRITER_LOCK`（默认 `configs/mod_profiles/.rocks_writer.lock`）

---

## 工具与脚本（Tools）

> 这些脚本主要用于：训练模型、测试模型、排查数据/兼容性问题。

### 1) 手动训练模型

```bash
# 训练指定 profile 的模型（HashLinear，推荐）
python tools/train_hashlinear_model.py default

# 或使用 BoW / fastText（兼容方案）
python tools/train_bow_model.py default
python tools/train_fasttext_model.py default
```

### 2) 查询审核日志

```bash
# 查询最近记录
python tools/query_moderation_log.py default --limit 10

# 查询违规记录
python tools/query_moderation_log.py default --label 1
```

### 3) 测试模型

```bash
# 测试 HashLinear 模型
python tools/verify_hashlinear_runtime.py default

# 测试 BoW / fastText 模型
python tools/test_bow_model.py default "测试文本"
python tools/test_fasttext_model.py default "测试文本"
```

### 4) 构建 HashLinear Rust 可选加速扩展

`tools/hashlinear_light_runtime.py` 默认会优先尝试导入 `hashlinear_rust_ext`，导入失败、运行异常或未构建扩展时会自动回退到现有 Python 实现，不会影响服务启动。

```bash
# 在当前 Python 环境中构建并安装本地扩展
python tools/build_hashlinear_rust.py
```

说明：

- 该脚本依赖本机已有 `cargo`、`rustc`、`maturin`
- 若已启用 `local_model_type = "hashlinear"`，构建 Rust 扩展后线上审核会优先使用 Rust 加速推理
- 若缺少 Rust 工具链，脚本会明确报错退出，服务仍可继续走 Python fallback
- 可通过环境变量 `HASHLINEAR_USE_RUST=0` 显式禁用 Rust 加速
- 可通过 `HASHLINEAR_USE_RUST=1` 强制尝试加载；若扩展不存在或导入失败，仍会告警后回退

---

## FAQ / Troubleshooting

### 如何禁用工具调用？

在配置中添加：

```json
{
  "format_transform": {
    "enabled": true,
    "disable_tools": true
  }
}
```

### 如何使用 fastText 替代 BoW？

参见 [fastText 迁移指南](docs/FASTTEXT_MIGRATION.md)。

### NumPy 2.0 兼容性问题？

项目在启动时会自动检查依赖兼容性。如遇到问题，请参考 [NumPy 2.0 兼容性说明](docs/NUMPY2_COMPATIBILITY.md)。

### 如何调整审核策略？

编辑 `configs/mod_profiles/{profile}/profile.json`：

- 提高 `ai_review_rate` → 更多 AI 审核（更准确，更贵）
- 降低 `low_risk_threshold` → 更严格（误拦截更多）
- 提高 `high_risk_threshold` → 更宽松（漏掉更多）

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [fastText](https://fasttext.cc/) - 文本分类

---

**PrismGuard** - 守护你的 AI API
