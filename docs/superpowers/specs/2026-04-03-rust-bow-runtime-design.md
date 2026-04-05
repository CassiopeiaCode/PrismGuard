# Rust BoW 审核运行时设计

## 目标

在不破坏现有 Python 基线文件约定的前提下，为 Rust 版 `smart_moderation` 增加 `bow` 本地模型决策能力，使其在 `local_model_type = "bow"` 时优先尝试本地审核，再按现有逻辑回退到 LLM。

## 背景

当前 Rust 版 `smart_moderation` 已支持：

- `basic_moderation`
- `smart_moderation`
- `hashlinear` 本地运行时
- LLM 回退
- 历史样本复用与写回

但与 Python 基线相比，仍缺少 `bow` 与 `fasttext` 本地模型路径。Python 的 `local_model_predict_proba()` 路由顺序为：

1. `fasttext`
2. `hashlinear`
3. 默认 `bow`

本轮优先补最窄缺口 `bow`。

## 兼容边界

必须保留 Python 现有基础文件命名与目录约定：

- `bow_model.pkl`
- `bow_vectorizer.pkl`

Rust 不要求直接反序列化 Python 的 `joblib/pickle` 文件。本轮允许在 Python 基线之上追加 Rust 专用 runtime sidecar 文件，只要不改变 Python 原有基础文件即可。

## 运行时文件

新增 Rust BoW runtime 文件：

- `bow_runtime.json`
- `bow_runtime.coef.f32`

其中：

- `bow_runtime.json` 保存运行时元数据、词表、IDF、拦截项与分词配置
- `bow_runtime.coef.f32` 保存二分类正类系数

`bow_model.pkl` 与 `bow_vectorizer.pkl` 仍保留为 Python 基线文件。Rust 本轮只把它们视为“该 profile 使用 BoW”的兼容占位，不直接解析其内容。

## 决策顺序

Rust `smart_moderation` 本地模型路由调整为：

1. `fasttext`
   当前未实现，返回 `None`
2. `hashlinear`
   维持现状
3. `bow`
   若存在 Rust runtime sidecar，则执行本地推理
4. 其余情况回退到 LLM

这意味着：

- 不改变 Python 原有 `local_model_type` 语义
- Rust 可以在 Python 基线上增加更易落地的 runtime 文件
- 当 `bow` runtime 缺失或损坏时，请求不会直接失败，而是回退到 LLM

## BoW Runtime 结构

`bow_runtime.json` 最小字段约定：

- `runtime_version`
- `tokenizer`
- `intercept`
- `idf`
- `vocabulary`
- `classes`

约束：

- `runtime_version` 首版固定为 `1`
- `classes` 固定为 `[0, 1]`
- `idf.len() == vocabulary.len()`
- `bow_runtime.coef.f32` 的浮点数数量必须等于 `vocabulary.len()`

## 文本特征

本轮实现最小可用推理语义：

- 词级 token：按空白切分
- 可选字符级 n-gram：按 Unicode 字符窗口切分
- 词频使用原始计数
- TF-IDF 使用 `tf * idf`
- 线性打分使用 `intercept + sum(weight_i * tfidf_i)`
- 输出概率使用 sigmoid

这样可以与 Python 的线性 BoW 决策保持同类行为，同时避免本轮被 `jieba`、`scikit-learn joblib` 兼容问题阻塞。

## 缓存与错误策略

BoW runtime 与 `hashlinear` 一样按文件签名做进程内缓存。

错误策略：

- runtime 文件缺失：返回 `None`，允许 LLM 回退
- runtime 文件损坏：返回 `None`，允许 LLM 回退
- 配置非法：返回 `None`，允许 LLM 回退

只有代理主链路已有的审核阻断结果继续走 `MODERATION_BLOCKED / moderation_error`。

## 测试要求

黑盒测试至少覆盖：

1. `local_model_type = "bow"` 且 runtime 高风险时，本地直接阻断，不打 LLM
2. `local_model_type = "bow"` 且 runtime 低风险时，直接放行，不打 LLM
3. `local_model_type = "bow"` 且 runtime 缺失时，回退到 LLM

本轮先至少交付第 1 条黑盒测试与实现，后续再扩第 2、3 条。
