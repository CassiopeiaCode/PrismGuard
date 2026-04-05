# Rust fastText 审核运行时设计

## 目标

在保持 Python `fasttext_model.bin` 基础不变的前提下，为 Rust 版 `smart_moderation` 增加 `fasttext` 本地模型决策路径，并保持“本地不可用时自动回退到 LLM”的语义。

## 当前差距

Python 本地模型路由顺序为：

1. `fasttext`
2. `hashlinear`
3. 默认 `bow`

Rust 当前已经补齐：

- `hashlinear` runtime
- `bow` runtime sidecar

剩余审核侧缺口主要是 `fasttext`。

## 兼容边界

必须保留 Python 现有基础文件：

- `fasttext_model.bin`

Rust 本轮不直接解析 Python `fastText .bin` 内部格式。原因：

- 原生格式兼容复杂
- 需要额外第三方库或 FFI
- 会把当前审核对齐工作拖入更大的绑定成本

因此本轮仍采用“保留 Python 基础文件 + Rust 追加 runtime sidecar”的策略。

## 运行时文件

新增 Rust sidecar：

- `fasttext_runtime.json`

最小字段：

- `runtime_version`
- `intercept`
- `classes`
- `tokenizer`
- `weights`

其中 `weights` 为 token 到线性权重的映射，供 Rust 进行最小二分类推理。

## 分词语义

Python `fasttext_model.py` 分两条路径：

1. 原始 fastText 文本输入
2. `jieba / tiktoken` 预分词版本

Rust 本轮先支持最小统一语义：

- 原文按空白切词
- 可选字符级 token
- 预分词模式通过 runtime 元数据指示

不在第一轮直接实现：

- `jieba`
- `tiktoken`
- 真正的 `.bin` 原生推理

## 回退策略

与 `bow`、`hashlinear` 一致：

- runtime sidecar 缺失：回退 LLM
- runtime 损坏：回退 LLM
- 配置不支持：回退 LLM

不允许因为本地 fastText 不可用而直接阻断请求。

## 实现顺序

1. 黑盒测试先钉住 `local_model_type = "fasttext"` 的高风险本地阻断
2. 新增 `src/moderation/fasttext.rs`
3. 在 `smart.rs` 中把 `fasttext` 路由接到最前面
4. 再补 runtime 缺失回退 LLM 测试

## 后续训练对齐

训练侧后续需要：

- 训练子进程能够产出 `fasttext_runtime.json`
- 保留 `fasttext_model.bin`
- Rust 主进程优先消费 runtime sidecar

这样可以保持 Python 现有训练资产文件不变，同时让 Rust 运行时具备稳定可控的本地推理格式。
