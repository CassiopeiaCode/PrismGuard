# GuardianBridge

**企业级 AI API 智能中间件** - 内容审核 · 格式转换 · 透明代理

一个专为 AI API 设计的智能中间件，提供三段式内容审核、多格式自动转换和透明代理功能。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 核心特性

### 🛡️ 三段式智能审核

- **基础层**：关键词黑名单过滤，支持热重载
- **本地模型**：词袋模型（BoW）或 fastText 快速预测
  - 低风险（p < 0.2）直接放行
  - 高风险（p > 0.8）直接拒绝
  - 不确定区域触发 AI 复核
- **AI 复核**：30% 随机抽样 + 模型不确定时调用外部 AI 审核
- **增量学习**：AI 审核结果自动保存为训练样本，模型持续优化

### 🔄 多格式透明转换

支持主流 AI API 格式的自动检测和相互转换：

| 格式 | 说明 |
|------|------|
| `openai_chat` | OpenAI Chat Completions API |
| `claude_chat` | Anthropic Messages API |
| `gemini_chat` | Google Gemini API |
| `openai_responses` | OpenAI Responses API |

**特性**：
- 自动格式检测
- 任意格式互转
- 完整工具调用支持
- 多模态输入（图像）
- 流式和非流式兼容

### 🚀 零代码集成

通过 URL 配置即可使用，无需修改客户端代码：

```python
from openai import OpenAI

# 使用预定义配置
base_url = "http://localhost:8000/!PROXY_CONFIG_DEFAULT$https://api.openai.com/v1"

client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
```

## 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd GuardianBridge

# 安装依赖（推荐使用 uv）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 配置

```bash
# 复制配置文件
cp .env.example .env
cp -r configs.example configs

# 编辑 .env，设置 AI 审核 API Key
# MOD_AI_API_KEY=sk-your-openai-api-key
```

### 启动

```bash
# 开发模式
python -m ai_proxy.app

# 生产模式
bash start.sh
```

服务将在 `http://localhost:8000` 启动。

## 使用示例

### 基础审核

```python
from openai import OpenAI

# 配置：仅基础审核和智能审核
base_url = "http://localhost:8000/!PROXY_CONFIG_DEFAULT$https://api.openai.com/v1"

client = OpenAI(api_key="sk-xxx", base_url=base_url)
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 格式转换：OpenAI → Claude

```python
from openai import OpenAI

# 使用 OpenAI SDK 调用 Claude API
base_url = "http://localhost:8000/!PROXY_CONFIG_CLAUDE$https://api.anthropic.com/v1"

client = OpenAI(
    api_key="sk-ant-xxx",  # Claude API Key
    base_url=base_url
)

# OpenAI 格式自动转换为 Claude 格式
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "你好"}]
)
```

### 格式转换：OpenAI → Gemini

```python
from openai import OpenAI

base_url = "http://localhost:8000/!PROXY_CONFIG_GEMINI$https://generativelanguage.googleapis.com"

client = OpenAI(
    api_key="your-gemini-api-key",
    base_url=base_url
)

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[{"role": "user", "content": "你好"}]
)
```

更多示例请参见 `examples/` 目录。

## 配置说明

### 预定义配置（环境变量）

在 `.env` 中定义配置，缩短 URL：

```bash
# 仅审核，不转换格式
PROXY_CONFIG_DEFAULT={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true,"profile":"default"},"format_transform":{"enabled":false}}

# OpenAI → Claude 转换
PROXY_CONFIG_CLAUDE={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true},"format_transform":{"enabled":true,"from":"openai_chat","to":"claude_chat"}}

# OpenAI → Gemini 转换
PROXY_CONFIG_GEMINI={"basic_moderation":{"enabled":true},"smart_moderation":{"enabled":true},"format_transform":{"enabled":true,"from":"openai_chat","to":"gemini_chat"}}
```

### Profile 配置

每个审核配置称为一个 "Profile"，位于 `configs/mod_profiles/{profile}/` 目录：

```
configs/mod_profiles/default/
├── profile.json         # 审核参数配置
├── ai_prompt.txt        # AI 审核提示词
├── history.db           # 审核历史数据库
├── bow_model.pkl        # 词袋模型（自动生成）
└── bow_vectorizer.pkl   # TF-IDF 向量化器（自动生成）
```

**核心参数** (`profile.json`)：

```json
{
  "ai": {
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key_env": "MOD_AI_API_KEY"
  },
  "probability": {
    "ai_review_rate": 0.3,        // 30% 随机抽样
    "low_risk_threshold": 0.2,    // 低于此值直接放行
    "high_risk_threshold": 0.8    // 高于此值直接拒绝
  },
  "bow_training": {
    "min_samples": 200,           // 最少样本数才开始训练
    "retrain_interval_minutes": 60, // 重训练间隔
    "max_samples": 50000,         // 每次训练最多样本数
    "max_features": 8000          // TF-IDF 最大特征数
  }
}
```

## 工具脚本

### 手动训练模型

```bash
# 训练指定 profile 的模型
python tools/train_bow_model.py default

# 或使用 fastText（需要额外配置）
python tools/train_fasttext_model.py default
```

### 查询审核日志

```bash
# 查询最近记录
python tools/query_moderation_log.py default --limit 10

# 查询违规记录
python tools/query_moderation_log.py default --label 1
```

### 测试模型

```bash
# 测试词袋模型
python tools/test_bow_model.py default "测试文本"

# 测试 fastText 模型
python tools/test_fasttext_model.py default "测试文本"
```

## 架构设计

```
客户端请求
    ↓
URL 解析（配置 + 上游地址）
    ↓
格式检测与转换
    ↓
文本抽取
    ↓
基础审核（关键词）
    ↓
智能审核（三段式）
    ├─ 30% → AI 审核
    └─ 70% → 本地模型
        ├─ p < 0.2 → 放行
        ├─ p > 0.8 → 拒绝
        └─ 0.2 ≤ p ≤ 0.8 → AI 复核
    ↓
转发到上游 API
    ↓
返回响应
```

## 性能指标

| 操作 | 耗时 |
|------|------|
| 关键词过滤 | <1ms |
| 词袋模型预测 | 3-5ms |
| AI 审核 | 500-2000ms |
| 格式转换 | <2ms |
| 缓存命中 | <0.1ms |

**资源占用**：
- 内存：100-200 MB
- CPU：单核可运行
- 磁盘：模型 5-10 MB，每 1000 条样本约 1 MB

## 常见问题

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

## 安全建议

1. **API Key 管理**：使用环境变量，不要提交 `.env` 到版本控制
2. **访问控制**：生产环境使用反向代理，配置 IP 白名单
3. **数据隐私**：审核历史包含用户输入，定期清理过期数据
4. **监控告警**：定期检查误判率，调整阈值

## 更新日志

### v1.1.0 (2024-12)

- ✅ 修复 NumPy 2.0 兼容性检查（启动时自动检测）
- ✅ 修复 OpenAI Responses 格式文本提取
- ✅ 改进 Gemini 流式请求检测（使用端点而非 stream 字段）
- ✅ 改进 fastText 概率计算（处理边缘情况）
- ✨ 新增 Gemini 格式支持
- ✨ 新增 OpenAI Responses API 支持
- ✨ 新增 `disable_tools` 配置项

### v1.0.0 (2024-11)

- ✨ 初始版本发布

## License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [scikit-learn](https://scikit-learn.org/) - 机器学习
- [jieba](https://github.com/fxsjy/jieba) - 中文分词
- [fastText](https://fasttext.cc/) - 文本分类

---

**GuardianBridge** - 守护你的 AI API 🛡️