# GuardianBridge (守桥)

**企业级 AI API 智能中间件** - 三段式智能审核 · 多格式透明转换 · 高性能代理

一个专为 AI API 设计的智能中间件系统，提供内容审核、格式转换和透明代理功能。通过 URL 配置即可实现零代码集成，支持 OpenAI、Claude 等主流 AI 服务。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ 核心特性

### 🛡️ 三段式智能审核

创新的混合审核策略，平衡准确性、性能和成本：

- **基础审核层**：关键词黑名单过滤，支持热重载
- **智能审核层**：三段式决策机制 + LRU 缓存
  - **30% 随机抽样** → AI 审核并记录标注数据
  - **本地模型低风险**（p < 0.2）→ 直接放行
  - **本地模型高风险**（p > 0.8）→ 直接拒绝
  - **本地模型不确定**（0.2 ≤ p ≤ 0.8）→ AI 复核
- **词袋线性模型**：轻量高效，适合 1C1G 环境
  - jieba 分词 + 字符级 n-gram（2-3）
  - TF-IDF 特征提取（最多 8000 维）
  - SGDClassifier 增量学习
  - 定时自动训练（可配置间隔）

> Note: BoW training now uses batch mode + layered vocabulary + async scheduling. See the next section for details.

#### BoW 模型与训练

- 当前版本采用 **一次性训练 + 分层词表**：调度器触发时会批量加载 `max_samples` 条样本，基于文档频率构建多层词表，再用 `TfidfVectorizer(lowercase=False)` + `SGDClassifier` 训练。
- `use_layered_vocab` 与 `vocab_buckets` 可自定义不同频率区间与数量，既保留高价值违规特征，又显著降低 6w+ 字符长文本带来的稀疏矩阵体积。
- 默认 `max_features=8000`（可通过 profile 调整，长文本环境建议降低到 5000 或更小以控制内存）。
- 训练任务通过 `asyncio.to_thread()` 在后台执行，不会阻塞 FastAPI 主线程；若需要立即重新训练，可运行 `python tools/train_bow_model.py <profile>`。
- 配套工具：`tools/diagnose_training_data.py`（检查样本质量）与 `tools/fix_bow_model.py`（辅助修复配置/样本不足问题）。

### 🔄 多格式透明转换

支持主流 AI API 格式的自动检测和相互转换：

- **自动格式检测**：智能识别 OpenAI Chat / Claude Messages API
- **灵活转换策略**：支持任意格式互转（OpenAI ↔ Claude）
- **完整工具调用支持**：
  - OpenAI: `tools` / `tool_calls` / `tool` role
  - Claude: `tools` / `tool_use` / `tool_result`
- **流式和非流式兼容**：自动适配请求类型
- **审核策略优化**：仅审核用户和助手文本，跳过工具参数和结果

### 🚀 高性能代理

- **URL 配置驱动**：无需修改代码，通过 URL 传递配置
- **环境变量优化**：预定义配置缩短 URL（节省 ~220 字符）
- **智能透传**：无法识别的格式自动透传原始请求
- **多上游支持**：兼容任意 OpenAI 兼容的 API 服务
- **错误处理**：详细的错误码和调试信息

## 📦 快速开始

### 前置要求

- Python 3.8+
- pip 或 uv

### 1. 克隆项目

```bash
git clone <repository-url>
cd GuardianBridge
```

### 2. 安装依赖

**方式 A：使用 pip**

```bash
pip install -r requirements.txt
```

**方式 B：使用 uv（推荐，更快）**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 [`.env`](.env:1) 文件，配置必要的参数：

```bash
# AI 审核 API Key（必需，用于智能审核）
MOD_AI_API_KEY=sk-your-openai-api-key

# 服务配置
HOST=0.0.0.0
PORT=8000
DEBUG=True

# 预定义配置（可选，用于缩短 URL）
PROXY_CONFIG_DEFAULT={"basic_moderation":{"enabled":true,"keywords_file":"configs/keywords.txt"},"smart_moderation":{"enabled":true,"profile":"default"},"format_transform":{"enabled":false}}
PROXY_CONFIG_CLAUDE={"basic_moderation":{"enabled":true,"keywords_file":"configs/keywords.txt"},"smart_moderation":{"enabled":true,"profile":"4claude"},"format_transform":{"enabled":true,"from":"openai_chat","to":"claude_chat"}}
```

### 4. 初始化配置文件

```bash
cp -r configs.example configs
```

编辑 [`configs/keywords.txt`](configs/keywords.txt:1) 添加关键词黑名单（每行一个）。

### 5. 启动服务

**方式 A：开发模式（自动重载）**

```bash
python -m ai_proxy.app
```

**方式 B：生产模式（使用 uv）**

```bash
bash start.sh
```

服务将在 `http://localhost:8000` 启动。

## 📖 使用指南

### URL 配置格式

GuardianBridge 支持两种配置方式：

#### 方式 1：URL 编码配置（临时测试）

```
http://proxy-host/{urlencoded_json_config}${upstream_url}
```

适合临时测试，配置直接嵌入 URL。

#### 方式 2：环境变量配置（推荐）

```
http://proxy-host/!{env_key}${upstream_url}
```

**优势**：URL 更短（~80 字符 vs ~300 字符），避免数据库字段溢出。

### 客户端示例

#### 示例 1：基础审核（URL 编码方式）

```python
from openai import OpenAI
import json
import urllib.parse

# 配置审核规则
config = {
    "basic_moderation": {
        "enabled": True,
        "keywords_file": "configs/keywords.txt"
    },
    "smart_moderation": {
        "enabled": True,
        "profile": "default"
    },
    "format_transform": {
        "enabled": False
    }
}

# 生成代理 URL
cfg_str = json.dumps(config, separators=(',', ':'))
cfg_enc = urllib.parse.quote(cfg_str, safe='')
upstream = "https://api.openai.com/v1"
base_url = f"http://localhost:8000/{cfg_enc}${upstream}"

# 使用 OpenAI SDK
client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好，世界！"}]
)
print(response.choices[0].message.content)
```

#### 示例 2：环境变量配置（推荐）

```python
from openai import OpenAI

# 使用预定义的环境变量配置
upstream = "https://api.openai.com/v1"
base_url = f"http://localhost:8000/!PROXY_CONFIG_DEFAULT${upstream}"

client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好，世界！"}]
)
```

#### 示例 3：OpenAI → Claude 格式转换

```python
from openai import OpenAI

# 使用 OpenAI SDK 调用 Claude API
upstream = "https://api.anthropic.com/v1"
base_url = f"http://localhost:8000/!PROXY_CONFIG_CLAUDE${upstream}"

client = OpenAI(
    api_key="sk-ant-xxx",  # Claude API Key
    base_url=base_url
)

# OpenAI SDK 格式会自动转换为 Claude 格式
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "你好，世界！"}]
)
```

#### 示例 4：工具调用（Function Calling）

```python
# 定义工具
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "城市名称"}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京今天天气如何？"}],
    tools=tools
)
```

更多示例请参见 [`examples/client_example.py`](examples/client_example.py:1)。

## ⚙️ 配置详解

### 基础审核配置

```json
{
  "basic_moderation": {
    "enabled": true,
    "keywords_file": "configs/keywords.txt",
    "error_code": "BASIC_MODERATION_BLOCKED"
  }
}
```

**参数说明**：
- [`enabled`](ai_proxy/moderation/basic.py:79): 是否启用基础审核
- [`keywords_file`](ai_proxy/moderation/basic.py:86): 关键词文件路径（支持热重载）
- [`error_code`](ai_proxy/moderation/basic.py:94): 拦截时返回的错误码

### 智能审核配置

```json
{
  "smart_moderation": {
    "enabled": true,
    "profile": "default"
  }
}
```

**Profile 配置文件** (`configs/mod_profiles/{profile}/profile.json`)：

```json
{
  "ai": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key_env": "MOD_AI_API_KEY",
    "timeout": 10
  },
  "probability": {
    "ai_review_rate": 0.3,
    "low_risk_threshold": 0.2,
    "high_risk_threshold": 0.8
  },
  "bow_training": {
    "min_samples": 200,
    "retrain_interval_minutes": 60,
    "max_samples": 50000,
    "max_features": 8000,
    "use_char_ngram": true,
    "char_ngram_range": [2, 3],
    "use_word_ngram": true,
    "word_ngram_range": [1, 2],
    "model_type": "sgd_logistic",
    "batch_size": 2000,
    "max_seconds": 300,
    "max_db_items": 100000,
    "use_layered_vocab": true,
    "vocab_buckets": [
      {"name": "high_freq", "min_doc_ratio": 0.05, "max_doc_ratio": 0.6, "limit": 1200},
      {"name": "mid_freq", "min_doc_ratio": 0.01, "max_doc_ratio": 0.05, "limit": 2600},
      {"name": "low_freq", "min_doc_ratio": 0.002, "max_doc_ratio": 0.01, "limit": 1200}
    ]
  }
}
```

**关键参数**：
- [`ai_review_rate`](ai_proxy/moderation/smart/ai.py:202): AI 审核随机抽样比例（默认 30%）
- [`low_risk_threshold`](ai_proxy/moderation/smart/ai.py:218): 低风险阈值，低于此值直接放行
- [`high_risk_threshold`](ai_proxy/moderation/smart/ai.py:238): 高风险阈值，高于此值直接拒绝
- [`min_samples`](ai_proxy/moderation/smart/bow.py:56): 最少样本数，达到后才开始训练
- [`retrain_interval_minutes`](ai_proxy/moderation/smart/scheduler.py:48): 模型重训练间隔
- [`max_samples`](ai_proxy/moderation/smart/bow.py:66): 每次训练最多加载的样本数，影响训练内存峰值
- [`max_db_items`](ai_proxy/moderation/smart/storage.py:248): 样本库容量上限；超出后按标签平衡随机清理
- [`use_layered_vocab` / `vocab_buckets`](ai_proxy/moderation/smart/profile.py:34): 是否启用分层词表及其频率区间/数量配置

### 格式转换配置

```json
{
  "format_transform": {
    "enabled": true,
    "from": "auto",
    "to": "claude_chat",
    "stream": "auto",
    "strict_parse": false,
    "disable_tools": false
  }
}
```

**参数说明**：
- [`from`](ai_proxy/proxy/router.py:134): 源格式
  - `"auto"`: 自动检测所有支持的格式（`openai_chat`, `claude_chat`, `claude_code`, `openai_codex`）
  - `"openai_chat"`: 仅识别 OpenAI Chat 格式
  - `["openai_chat", "claude_chat"]`: 识别列表中的任意格式
- [`to`](ai_proxy/proxy/router.py:184): 目标格式（`openai_chat` / `claude_chat` / `claude_code` / `openai_codex`）
- [`stream`](ai_proxy/transform/formats/parser.py:1): 流式策略
  - `"auto"`: 保持原请求的流式设置
  - `"force_stream"`: 强制使用流式
  - `"force_non_stream"`: 强制使用非流式
- [`strict_parse`](ai_proxy/proxy/router.py:135): 严格解析模式
  - `false`: 无法解析时透传原始请求
  - `true`: 无法解析时返回错误
- `disable_tools`: 禁用工具调用（新增）
  - `false`: 允许工具调用（默认）
  - `true`: 禁用工具调用，拒绝包含工具的请求

#### 禁用工具调用配置

当 `disable_tools: true` 时：

1. **自动排除格式**：
   - `claude_code` 和 `openai_codex` 格式会被自动排除
   - 这两个格式主要用于工具调用场景

2. **检测并拒绝**：
   - 包含 `tools` 字段（工具定义）
   - 包含 `tool_choice` 字段（工具选择）
   - 包含 `tool_call` 类型的消息块（工具调用）
   - 包含 `tool_result` 类型的消息块（工具结果）

3. **配置优先级**：
   - `disable_tools` 会覆盖 `from` 配置
   - 即使 `from` 设置为 `"claude_code"`，启用 `disable_tools` 后也会被拒绝

**使用场景**：
```json
{
  "format_transform": {
    "enabled": true,
    "from": "auto",
    "to": "openai_chat",
    "disable_tools": true
  }
}
```

- **限制功能**：只允许简单对话，不允许工具调用
- **兼容性**：避免上游 API 不支持工具调用导致的错误
- **安全考虑**：防止工具调用绕过审核机制

**错误信息示例**：
```json
{
  "error": {
    "code": "FORMAT_PARSE_ERROR",
    "message": "Tool calling is disabled by configuration. The request contains tool definitions or tool-related content, which is not allowed. Please remove 'tools', 'tool_choice', tool calls, or tool results from your request.",
    "type": "format_error"
  }
}
```

## 🏗️ 架构设计

### 目录结构

```
ai_proxy/
├── app.py                          # FastAPI 应用入口
├── config.py                       # 全局配置管理
├── proxy/
│   ├── router.py                   # 主路由处理（支持多格式检测）
│   └── upstream.py                 # 上游 
HTTP 客户端
├── moderation/
│   ├── basic.py                    # 基础审核（关键词过滤）
│   └── smart/
│       ├── ai.py                   # AI 审核（三段式决策 + LRU 缓存）
│       ├── bow.py                  # 词袋线性模型（训练和预测）
│       ├── profile.py              # 配置文件管理
│       ├── scheduler.py            # 定时训练调度器
│       └── storage.py              # SQLite 数据存储
└── transform/
    ├── extractor.py                # 文本抽取（避免审核工具参数）
    └── formats/
        ├── internal_models.py      # 内部统一模型（支持工具调用）
        ├── parser.py               # 格式解析器注册表（支持 disable_tools）
        ├── openai_chat.py          # OpenAI Chat 格式解析
        ├── claude_chat.py          # Claude Messages 格式解析
        ├── claude_code.py          # Claude Code (Agent SDK) 格式解析
        └── openai_codex.py         # OpenAI Codex/Completions 格式解析

configs/
├── keywords.txt                    # 关键词黑名单
└── mod_profiles/
    └── {profile}/
        ├── profile.json            # 审核配置
        ├── ai_prompt.txt           # AI 提示词模板
        ├── history.db              # 审核历史（SQLite）
        ├── bow_model.pkl           # 词袋模型
        └── bow_vectorizer.pkl      # TF-IDF 向量化器

tools/
├── train_bow_model.py              # 手动训练模型
├── test_bow_model.py               # 模型测试工具
└── query_moderation_log.py         # 查询审核日志

examples/
├── client_example.py               # 客户端使用示例
└── config_examples.py              # 配置示例
```

### 核心流程

#### 请求处理流程

```
客户端请求
    ↓
URL 解析（配置 + 上游地址）
    ↓
格式检测（OpenAI/Claude/透传）
    ↓
转换为内部格式
    ↓
文本抽取（仅用户和助手内容）
    ↓
基础审核（关键词过滤）
    ↓
智能审核（三段式决策）
    ↓
格式转换（如需要）
    ↓
转发到上游 API
    ↓
返回响应
```

#### 三段式智能审核流程

```
文本输入
    ↓
检查 LRU 缓存
    ↓ (未命中)
生成随机数
    ↓
┌──────────────────┬──────────────────┐
│   30% 抽样       │   70% 本地模型   │
│      ↓           │        ↓         │
│  AI 审核         │   词袋预测概率    │
│      ↓           │        ↓         │
│  保存到数据库     │  ┌────┴────┐    │
│      ↓           │  │         │    │
│  返回结果        │ p<0.2   p>0.8   │
│                  │  ↓         ↓    │
│                  │ 放行      拒绝   │
│                  │           │    │
│                  │  0.2≤p≤0.8│    │
│                  │      ↓     │    │
│                  │  AI 复核 ←┘    │
└──────────────────┴─────┬───────────┘
                          ↓
                    保存到缓存
                          ↓
                      返回结果
```

### 工具调用支持

#### 内部统一模型

使用 [`InternalChatRequest`](ai_proxy/transform/formats/internal_models.py:43) 统一不同 API 格式：

```python
class InternalContentBlock:
    type: "text" | "tool_call" | "tool_result"
    text: str | None
    tool_call: InternalToolCall | None
    tool_result: InternalToolResult | None
```

#### 格式映射

| OpenAI | Claude | Internal |
|--------|--------|----------|
| `tools` | `tools` | [`InternalTool`](ai_proxy/transform/formats/internal_models.py:8) |
| `tool_calls` | `tool_use` | [`InternalToolCall`](ai_proxy/transform/formats/internal_models.py:15) |
| `role="tool"` | `tool_result` | [`InternalToolResult`](ai_proxy/transform/formats/internal_models.py:22) |

#### 审核策略

- ✅ **审核内容**：user 和 assistant 的 `text` 类型内容块
- ❌ **不审核**：
  - 工具定义（`tools`）
  - 工具调用参数（`tool_call.arguments`）
  - 工具返回结果（`tool_result.output`）

参见 [`extract_text_from_internal()`](ai_proxy/transform/extractor.py:30)

## 🔧 高级功能

### 手动训练模型

```bash
# 训练指定 profile 的词袋模型
python tools/train_bow_model.py default

# 训练 claude 配置的模型
python tools/train_bow_model.py 4claude
```

模型训练需要满足最小样本数（默认 200 条），可在 [`profile.json`](ai_proxy/moderation/smart/profile.py:67) 中配置。

- 训练前可执行 `python tools/diagnose_training_data.py <profile>` 检查标签分布；若违规样本过少，可暂时提高 `ai_review_rate` 积累标注或使用 `tools/fix_bow_model.py` 获取修复建议。
- `bow_training.use_layered_vocab` 默认开启，若有特殊语料（多语言/大小写敏感），可在 profile 中调整 `vocab_buckets` 与 `max_features`。
- 训练日志会打印词表覆盖度和训练准确率；如发现异常，可删除最新模型文件后重新训练，或调高 `min_samples` 暂停自动训练。

### 查询审核日志

```bash
# 查询最近的审核记录
python tools/query_moderation_log.py default --limit 10

# 查询违规记录
python tools/query_moderation_log.py default --label 1

# 按类别查询
python tools/query_moderation_log.py default --category "spam"
```

### 自定义审核 Profile

1. 创建新的 profile 目录：

```bash
mkdir -p configs/mod_profiles/my_profile
```

2. 创建配置文件 `configs/mod_profiles/my_profile/profile.json`：

```json
{
  "ai": {
    "base_url": "https://your-api.com/v1",
    "model": "your-model",
    "api_key_env": "YOUR_API_KEY"
  },
  "probability": {
    "ai_review_rate": 0.3,
    "low_risk_threshold": 0.2,
    "high_risk_threshold": 0.8
  },
  "bow_training": {
    "min_samples": 200,
    "retrain_interval_minutes": 60,
    "max_samples": 20000,
    "max_features": 5000,
    "max_db_items": 50000,
    "use_layered_vocab": true,
    "vocab_buckets": [
      {"name": "high_freq", "min_doc_ratio": 0.05, "max_doc_ratio": 0.6, "limit": 1000},
      {"name": "mid_freq", "min_doc_ratio": 0.01, "max_doc_ratio": 0.05, "limit": 2500},
      {"name": "low_freq", "min_doc_ratio": 0.002, "max_doc_ratio": 0.01, "limit": 1500}
    ]
  }
}
```

3. 创建 AI 提示词模板 `configs/mod_profiles/my_profile/ai_prompt.txt`：

```
你是一个内容审核专家。请判断以下文本是否违规。

文本内容：
{{text}}

请以 JSON 格式返回结果：
{
  "violation": true/false,
  "category": "类别（如: spam, abuse, hate等）",
  "reason": "简短原因"
}
```

4. 在配置中使用：

```python
config = {
    "smart_moderation": {
        "enabled": True,
        "profile": "my_profile"
    }
}
```

## 🚦 错误码参考

| 错误码 | 说明 | HTTP 状态码 |
|--------|------|------------|
| `CONFIG_PARSE_ERROR` | 配置解析错误 | 400 |
| `BASIC_MODERATION_BLOCKED` | 基础审核拦截（关键词匹配） | 400 |
| `SMART_MODERATION_BLOCKED` | 智能审核拦截（AI 或模型判定） | 400 |
| `FORMAT_PARSE_ERROR` | 格式解析错误（strict_parse=true 或 disable_tools=true） | 400 |
| `FORMAT_TRANSFORM_ERROR` | 格式转换错误 | 500 |
| `UPSTREAM_ERROR` | 上游请求失败 | 500 |
| `PROXY_ERROR` | 代理内部错误 | 500 |

### 错误响应格式

```json
{
  "error": {
    "code": "SMART_MODERATION_BLOCKED",
    "message": "Smart moderation blocked by bow_model (confidence: 0.856)",
    "type": "moderation_error",
    "source_format": "openai_chat",
    "moderation_details": {
      "source": "bow_model",
      "reason": "BoW: high risk (p=0.856)",
      "category": null,
      "confidence": 0.856
    }
  }
}
```

## 📊 性能优化

### 缓存策略

1. **LRU 缓存**（每个 profile 20 条）
   - 缓存 AI 和模型的审核结果
   - 基于文本 MD5 哈希
   - 自动淘汰最旧记录

2. **模型缓存**
   - 词袋模型和向量化器常驻内存
   - 检测文件修改时间，自动重载
   - 避免重复加载和内存泄漏

3. **数据库查询优化**
   - 先查数据库再调用 AI（避免重复审核）
   - 使用索引加速文本查找

### 资源占用

- **内存**：约 100-200 MB（含模型）
- **CPU**：单核可运行，词袋预测 <5ms
- **磁盘**：
  - 模型文件：约 5-10 MB
  - 数据库：每 1000 条样本约 1 MB

### 性能基准

| 操作 | 耗时 |
|------|------|
| 关键词过滤 | <1ms |
| 词袋模型预测 | 3-5ms |
| AI 审核 | 500-2000ms |
| 格式转换 | <2ms |
| 缓存命中 | <0.1ms |

## 🛠️ 开发指南

### 添加新的格式支持

1. 在 [`ai_proxy/transform/formats/`](ai_proxy/transform/formats/) 创建新的解析器：

```python
# my_format.py
from ai_proxy/transform/formats/internal_models import InternalChatRequest

def can_parse_my_format(path, headers, body) -> bool:
    """检测是否为目标格式"""
    return path.startswith("/my/api") or headers.get("x-api-type") == "my_format"

def from_my_format(body: dict) -> InternalChatRequest:
    """转换为内部格式"""
    # 实现转换逻辑
    pass

def to_my_format(req: InternalChatRequest) -> dict:
    """从内部格式转换"""
    # 实现转换逻辑
    pass
```

2. 在 [`parser.py`](ai_proxy/transform/formats/parser.py:75) 注册解析器：

```python
from ai_proxy.transform.formats import my_format

class MyFormatParser:
    name = "my_format"
    
    def can_parse(self, path, headers, body):
        return my_format.can_parse_my_format(path, headers, body)
    
    # ... 其他方法

PARSERS["my_format"] = MyFormatParser()
```

### 自定义审核逻辑

继承或修改 [`smart_moderation()`](ai_proxy/moderation/smart/ai.py:172) 函数：

```python
async def custom_moderation(text: str, cfg: dict) -> Tuple[bool, Optional[ModerationResult]]:
    """自定义审核逻辑"""
    # 1. 调用外部审核 API
    # 2. 使用规则引擎
    # 3. 多模型融合决策
    pass
```

### 调试技巧

1. **启用详细日志**：

编辑 [`.env`](.env:1)：
```bash
DEBUG=True
LOG_LEVEL=DEBUG
```

2. **查看请求详情**：

所有审核请求都会打印详细信息：
```
[DEBUG] ========== 
请求处理开始 ==========
  路径: /v1/chat/completions
  格式转换: 启用
  检测到格式: openai_chat
  抽取文本长度: 42 字符
[DEBUG] 基础审核开始
  待审核文本: 你好，世界！
  关键词文件: configs/keywords.txt
  已加载关键词数量: 10
[DEBUG] 基础审核结果: ✅ 通过
[DEBUG] 智能审核开始
  待审核文本: 你好，世界！
  使用配置: default
  AI审核概率: 30.0%
[DEBUG] 决策路径: 词袋模型预测
  违规概率: 0.123
  阈值: 低风险 < 0.200, 高风险 > 0.800
[DEBUG] 词袋模型结果: ✅ 低风险放行
[DEBUG] ========== 请求通过审核 ==========
```

3. **测试模型预测**：

```bash
python tools/test_bow_model.py default "测试文本"
```

## 📈 监控和运维

### 日志文件

- **访问日志**：`logs/access.log`
- **审核日志**：`logs/moderation.log`
- **训练日志**：`logs/training.log`

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/

# 查看模型状态
ls -lh configs/mod_profiles/*/bow_model.pkl
```

### 定时任务

服务启动时会自动启动模型训练调度器（默认每 10 分钟检查一次）。可在 [`app.py`](ai_proxy/app.py:39) 中调整：

```python
start_scheduler(check_interval_minutes=10)  # 修改检查间隔
```

调度器会逐个 profile 获取锁并在后台线程中调用 `train_bow_model()`，因此不会阻塞 FastAPI 主事件循环；若某个 profile 正在训练，会在下一轮自动跳过。

### 数据备份

定期备份审核数据库：

```bash
# 备份所有 profile 的数据库
cp -r configs/mod_profiles/*/history.db backups/
```

## 🔒 安全建议

1. **API Key 管理**
   - 使用环境变量存储敏感信息
   - 不要将 `.env` 文件提交到版本控制

2. **访问控制**
   - 在生产环境使用反向代理（Nginx/Caddy）
   - 配置 IP 白名单或 API Key 验证

3. **审核策略**
   - 定期审查关键词列表
   - 监控 AI 审核的误判率
   - 调整阈值以平衡准确性和性能

4. **数据隐私**
   - 审核历史包含用户输入，注意数据保护
   - 定期清理过期数据
   - 考虑加密存储敏感内容

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

### 提交 Issue

- 描述问题或功能需求
- 提供复现步骤（如适用）
- 附上相关日志或错误信息

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送分支：`git push origin feature/your-feature`
5. 创建 Pull Request

### 代码规范

- 遵循 PEP 8 风格指南
- 添加必要的注释和文档字符串
- 确保所有测试通过

## 📝 更新日志

### v1.1.0 (2024-11)

- ✨ 新增 Claude Code (Agent SDK) 格式支持
- ✨ 新增 OpenAI Codex/Completions 格式支持
- ✨ 新增 `disable_tools` 配置选项，禁用工具调用
- ✨ 格式识别互斥机制，避免误识别
- 🐛 修复 `cache_control` 字段检测逻辑
- 🐛 修复 `role="tool"` 消息格式冲突

### v1.0.0 (2024-11)

- ✨ 三段式智能审核系统
- ✨ OpenAI ↔ Claude 格式转换
- ✨ 完整工具调用支持
- ✨ LRU 缓存优化
- ✨ 词袋线性模型自动训练
- ✨ 环境变量配置支持

## 📄 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Web 框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习库
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [OpenAI](https://openai.com/) - AI API 标准
- [Anthropic](https://www.anthropic.com/) - Claude API

## 📞 联系方式

- 项目主页：[GitHub Repository]
- 问题反馈：[GitHub Issues]
- 邮件：your-email@example.com

---

**GuardianBridge** - 守护你的 AI API，让内容更安全 🛡️
