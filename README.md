# GuardianBridge (守桥)

**高级 AI API 中间件** - 智能审核 · 格式转换 · 透明代理

支持文本审核（基础关键词 + AI + 随机森林）、格式转换（OpenAI/Claude 互转）和工具调用的代理服务。

## 🚀 功能特性

### 1. 文本审核
- **基础审核**：关键词过滤，支持热加载
- **智能审核**：AI 审核 + 词袋线性模型（jieba + TF-IDF + SGDClassifier），自动学习优化
- **三段式决策**：
  - 30% 随机抽样 → AI 审核并记录标注
  - 本地模型低风险（p<0.2）→ 直接放行
  - 本地模型高风险（p>0.8）→ 直接拒绝
  - 本地模型不确定 → AI 复核

### 2. 格式转换
- **多来源支持**：自动检测 OpenAI Chat / Claude Chat 格式
- **灵活转换**：支持任意格式互转
- **工具调用**：完整支持 tools / tool_calls / tool_use / tool_result
- **流式兼容**：支持流式和非流式请求

### 3. 透明代理
- **URL 配置**：通过 URL 传递配置，无需修改代码
- **完全透传**：未识别的格式自动透传
- **多上游**：支持任意兼容的上游服务

## 📦 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，设置 MOD_AI_API_KEY
```

### 3. 启动服务

```bash
python -m ai_proxy.app
```

服务将在 `http://localhost:8000` 启动。

## 📖 使用方式

### URL 格式

支持两种配置方式：

#### 1. URL 编码配置（临时测试）

```
http://proxy-host/{urlencoded_json_config}${upstream_url}
```

#### 2. 环境变量配置（推荐，URL更短）

```
http://proxy-host/!{env_key}${upstream_url}
```

### 基础示例（URL编码方式）

```python
from openai import OpenAI
import json
import urllib.parse

# 配置
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

# 使用代理
client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 环境变量配置方式（推荐）

**优点**：URL 更短，避免数据库字段溢出

#### 1. 配置环境变量

在 `.env` 文件中添加：

```bash
# 默认配置（基础+智能审核）
PROXY_CONFIG_DEFAULT={"basic_moderation":{"enabled":true,"keywords_file":"configs/keywords.txt"},"smart_moderation":{"enabled":true,"profile":"default"},"format_transform":{"enabled":false}}

# Claude 转换配置
PROXY_CONFIG_CLAUDE={"basic_moderation":{"enabled":true,"keywords_file":"configs/keywords.txt"},"smart_moderation":{"enabled":true,"profile":"4claude"},"format_transform":{"enabled":true,"from":"openai_chat","to":"claude_chat"}}
```

#### 2. 使用客户端

```python
from openai import OpenAI

# 使用环境变量配置
upstream = "https://api.openai.com/v1"
base_url = f"http://localhost:8000/!PROXY_CONFIG_DEFAULT${upstream}"

client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
```

#### URL 长度对比

- URL 编码方式：~300+ 字符
- 环境变量方式：~80 字符
- **节省**：~220+ 字符

### 工具调用示例

```python
# 支持 OpenAI 工具调用
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=tools
)
```

### 格式转换示例（环境变量方式）

```python
# OpenAI SDK + Claude API（自动转换）
upstream = "https://api.anthropic.com/v1"
base_url = f"http://localhost:8000/!PROXY_CONFIG_CLAUDE${upstream}"

# 使用 OpenAI SDK，实际调用 Claude API
client = OpenAI(
    api_key="sk-ant-xxx",  # Claude API Key
    base_url=base_url
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 完整客户端示例

参见 [`examples/client_example.py`](examples/client_example.py)：

- URL 编码配置
- 环境变量配置（推荐）
- OpenAI → Claude 转换

## ⚙️ 配置说明

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

### 智能审核配置

```json
{
  "smart_moderation": {
    "enabled": true,
    "profile": "default"
  }
}
```

智能审核配置文件位于 `configs/mod_profiles/{profile}/profile.json`：

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
    "random_seed": 42,
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
    "model_type": "sgd_logistic"
  }
}
```

### 格式转换配置

```json
{
  "format_transform": {
    "enabled": true,
    "from": "auto",
    "to": "openai_chat",
    "stream": "auto"
  }
}
```

#### from 参数说明

- **字符串**：如 `"openai_chat"`，只识别该格式
- **数组**：如 `["openai_chat", "claude_chat"]`，识别列表中的任意格式
- **"auto"**：自动检测所有支持的格式

#### 支持的格式

- `openai_chat`：OpenAI Chat Completions API
- `claude_chat`：Claude Messages API

#### stream 参数

- `"auto"`：保持原请求的流式设置
- `"force_stream"`：强制使用流式
- `"force_non_stream"`：强制使用非流式

## 🛠️ 目录结构

```
ai_proxy/
├── app.py                      # 主入口
├── config.py                   # 全局配置
├── proxy/
│   ├── router.py              # 路由处理（支持多来源检测）
│   └── upstream.py            # 上游客户端
├── moderation/
│   ├── basic.py               # 基础审核
│   └── smart/
│       ├── profile.py         # 配置管理
│       ├── ai.py              # AI 审核（三段式决策）
│       ├── bow.py             # 词袋线性模型
│       └── storage.py         # 数据存储
└── transform/
    ├── extractor.py           # 文本抽取（避免审核工具参数）
    └── formats/
        ├── internal_models.py # 内部模型（支持工具调用）
        ├── parser.py          # 格式解析器注册表
        ├── openai_chat.py     # OpenAI 格式（支持 tools）
        └── claude_chat.py     # Claude 格式（支持 tool_use）

configs/
├── keywords.txt               # 关键词列表
└── mod_profiles/
    └── default/
        ├── profile.json       # 审核配置
        ├── ai_prompt.txt      # AI 提示词
        ├── history.db         # 审核历史
        ├── bow_model.pkl      # 词袋线性模型
        └── bow_vectorizer.pkl # TF-IDF 向量化器

examples/
├── config_examples.py         # 配置示例
└── client_usage.py           # 客户端使用示例
```

## 🔧 工具调用支持

### 内部统一模型

代理使用内部统一格式来处理不同 API 的工具调用：

```python
# 内部内容块类型
- text: 文本内容
- tool_call: 工具调用（OpenAI tool_calls / Claude tool_use）
- tool_result: 工具结果（OpenAI tool role / Claude tool_result）
```

### OpenAI ↔ Claude 转换

| OpenAI | Claude | Internal |
|--------|--------|----------|
| `tools` | `tools` | `InternalTool` |
| `tool_calls` | `tool_use` | `InternalToolCall` |
| role=`tool` | `tool_result` | `InternalToolResult` |

### 审核策略

- ✅ **审核**：user 和 assistant 的文本内容
- ❌ **不审核**：工具参数（`arguments`/`input`）和工具结果（`output`）

## 📊 智能审核工作流程

```
请求 → 抽取文本 → 基础审核（关键词）
                ↓ 通过
              智能审核
         /              \
    30% 随机抽样    70% 本地模型
        ↓                ↓
      AI审核         词袋预测
        ↓           /    |    \
    记录到DB    低风险  不确定  高风险
        ↓        ↓      ↓      ↓
    定期训练 ← 放行   AI复核  拒绝
```

### 词袋模型特点

- **轻量高效**：适合 1C1G 环境，内存占用小
- **快速推理**：线性模型预测速度快
- **增量学习**：SGDClassifier 支持在线更新
- **混合特征**：jieba 分词 + 字符级 n-gram

## 🚦 错误码

- `CONFIG_PARSE_ERROR`：配置解析错误
- `BASIC_MODERATION_BLOCKED`：基础审核拦截
- `SMART_MODERATION_BLOCKED`：智能审核拦截
- `FORMAT_TRANSFORM_ERROR`：格式转换错误
- `UPSTREAM_ERROR`：上游请求错误
- `PROXY_ERROR`：代理错误

## 📝 开发计划

- [ ] 实现流式响应格式转换
- [ ] 添加词袋模型定时训练任务
- [ ] 完善日志和监控
- [ ] 支持更多格式（OpenAI Responses 等）
- [ ] 添加性能指标和统计
- [ ] Web 管理界面
- [ ] 模型性能评估和 A/B 测试

## 📄 License

MIT

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！