# Rust AI 审核运行时闭环设计

## 背景

当前 Rust 版已经基本完成与 Python 版的 HTTP 代理主链路对齐，请求改写、普通响应、流式响应、错误响应和非 200 透传语义都已有较完整的黑盒覆盖。

下一阶段优先目标不再是继续深挖 HTTP 边角，而是补齐 Python 版在请求处理阶段已经具备的 AI 审核能力，特别是：

- `basic_moderation`
- `smart_moderation`
- LLM 审核调用
- `hashlinear` 本地模型推理
- Python 风格的 `moderation_error / moderation_details`

这一阶段只做“运行时审核闭环”，不提前把训练、调度、样本落库和多 profile 动态切换一起引入。

## 目标

本阶段目标是让 Rust 版具备与 Python 版相同层级的请求时审核能力，并满足以下约束：

- 保持现有 HTTP 代理行为不回退
- 审核只在请求进入上游前发生
- 审核结果通过现有 Rust 错误协议返回
- `hashlinear` 只作为“已有模型时参与决策的本地模型”
- 模型不存在时不报错，而是自动回退到 LLM
- 先支持单个默认 moderation profile

## 非目标

本阶段明确不做以下内容：

- `hashlinear` 模型训练
- 训练调度器
- 样本写入 RocksDB
- 训练日志与后台任务
- 多 moderation profile 动态切换
- 审核样本存储服务
- 模型导出与 runtime 生成

这些内容属于后续“训练与存储”阶段，不纳入当前实现。

## 总体方案

运行时审核链路分为两层：

1. `basic_moderation`
2. `smart_moderation`

请求先抽取审核文本，再先跑 `basic_moderation`。若基础审核未命中，再进入 `smart_moderation`。

`smart_moderation` 的运行时决策顺序保持与 Python 基线一致：

1. 查询缓存
2. 检查并发限制
3. 尝试本地 `hashlinear` 推理
4. 如果本地模型不可用或结果不足以直接决策，则回退到 LLM
5. 写回缓存

审核通过则继续当前代理流程；审核失败则返回 `400 MODERATION_BLOCKED / moderation_error`。

## 模块拆分

为避免把审核细节继续堆进 `src/proxy.rs`，建议新增 `src/moderation/` 子模块，并分成以下组件。

### 1. `src/moderation/basic.rs`

职责：

- 承载基础审核规则
- 输入文本，输出“通过 / 拒绝 + 原因”
- 不依赖模型、不依赖 HTTP、不关心配置加载细节以外的逻辑

### 2. `src/moderation/extract.rs`

职责：

- 从原始请求体抽取审核文本
- 从内部格式请求抽取审核文本
- 与 Python `extract_text_for_moderation / extract_text_from_internal` 语义对齐

约束：

- 同一请求的文本抽取结果必须稳定
- 不能因为是否开启格式转换而改变审核输入的语义

### 3. `src/moderation/smart.rs`

职责：

- `smart_moderation` 的运行时编排中心
- 负责缓存、并发限制、本地模型优先、LLM 回退、结果整合

输出统一审核结果结构，例如：

- `violation`
- `source`
- `reason`
- `category`
- `confidence`

### 4. `src/moderation/hashlinear.rs`

职责：

- 检查默认 profile 下的 `hashlinear` 模型文件是否存在
- 模型加载
- 单条文本推理

约束：

- 只做运行时推理，不做训练
- 模型不存在时返回“不可用”，不应抛出阻断级错误
- 推理失败时由上层决定是否回退到 LLM 或进入公共错误兜底

### 5. `src/moderation/llm.rs`

职责：

- 执行 LLM 审核调用
- 读取默认 profile 中的 AI 配置
- 负责超时、重试、候选模型选择等运行时策略

约束：

- 入口保持可配置，不硬编码单一 provider
- 本阶段只需支持默认 profile，但接口不要写死成不可扩展

## 主链路接入点

审核接入仍放在当前请求处理主链路中，由 `src/proxy.rs` 发起调用。

`src/proxy.rs` 的职责仅保留：

- 收集调用审核所需上下文
- 调用统一审核入口
- 审核失败时映射到现有 `ApiError`

`src/proxy.rs` 不直接实现：

- 基础审核规则
- `hashlinear` 推理
- LLM 调用
- 审核缓存或并发控制

## 运行时数据流

### 场景 A：审核未启用

1. 请求进入代理
2. 不触发审核
3. 直接走现有 HTTP 转发逻辑

### 场景 B：`basic_moderation` 命中

1. 请求进入代理
2. 抽取审核文本
3. 运行 `basic_moderation`
4. 命中违规，立即返回阻断错误
5. 不再执行 `smart_moderation`
6. 不访问模型，不调用 LLM

### 场景 C：`smart_moderation` 走本地模型

1. 请求进入代理
2. 抽取审核文本
3. `basic_moderation` 通过
4. 查询缓存，若命中直接返回
5. 检查并发限制
6. 如果 `hashlinear` 模型存在，则执行本地推理
7. 若本地结果足够明确，则直接给出结论
8. 写入缓存
9. 返回通过或阻断

### 场景 D：`hashlinear` 不可用并回退到 LLM

1. 请求进入代理
2. 抽取审核文本
3. `basic_moderation` 通过
4. 查询缓存
5. 检查并发限制
6. 发现本地模型缺失或不可用
7. 回退到 LLM 审核
8. 写入缓存
9. 返回通过或阻断

## 默认 profile 约束

本阶段只支持单个默认 moderation profile。

实现上需要满足：

- profile 路径和配置解析方式与现有 Rust `profile.rs` 保持兼容
- 后续可以扩展为按配置动态切换 profile
- 当前不为“多 profile”引入额外接口复杂度

## `hashlinear` 运行时语义

`hashlinear` 在这一阶段的定位是“已有模型就参与运行时审核的本地分类器”。

行为要求：

- 若模型文件存在，则尝试加载并执行推理
- 若模型文件不存在，则视为“当前本地模型不可用”，自动回退到 LLM
- 若模型推理结论明确，则可直接给出审核结论
- 若模型推理不足以直接决策，则继续交给 LLM

不允许的行为：

- 因模型缺失而直接阻断请求
- 因训练尚未实现而使审核链路不可用

## LLM 审核运行时语义

LLM 审核必须具备以下能力：

- 读取默认 profile 中的 AI 配置
- 控制超时与重试
- 候选模型选择
- 并发限制

并发限制语义要求与 Python 基线保持一致：

- 达到上限时立即返回，不排队等待
- 以 `concurrency_limit` 的来源信息返回给上层

## 缓存语义

`smart_moderation` 需要为相同文本结果做缓存。

本阶段缓存要求：

- 按文本哈希作为 key
- 保留 profile 维度隔离能力，即使当前只支持默认 profile
- 命中缓存时应跳过本地模型和 LLM
- 审核结果写回缓存时保留来源信息

本阶段不要求完全复刻 Python 的所有缓存淘汰实现细节，但需要保留 LRU 或近似等价语义。

## 错误响应语义

### 审核阻断

基础审核或智能审核阻断时返回：

- HTTP `400`
- `error.code = "MODERATION_BLOCKED"`
- `error.type = "moderation_error"`

### `moderation_details`

当存在结构化审核信息时，应返回 `moderation_details`，至少支持：

- `source`
- `reason`
- `category`
- `confidence`

典型来源包括：

- `basic`
- `hashlinear`
- `ai`
- `concurrency_limit`

### 并发超限

LLM 并发超限时：

- 返回 `400 MODERATION_BLOCKED / moderation_error`
- `moderation_details.source = "concurrency_limit"`

### 内部异常

内部异常处理规则：

- 模型缺失属于预期缺省，应回退到下一层，不返回错误
- 真正不可恢复的运行时异常最终走公共兜底
- 公共兜底保持当前 Rust 已对齐的 `500 PROXY_ERROR / proxy_error`

## 与 HTTP 主链路的兼容要求

引入审核运行时后，必须保证：

- 未启用审核时，现有 HTTP 对齐行为完全不变
- 审核通过时，现有请求改写、普通响应、流式响应行为完全不变
- 审核只影响“是否允许进入上游”这一决策
- 审核逻辑不得破坏当前已完成的错误协议和 header 语义

## 测试策略

本阶段测试只覆盖运行时审核，不覆盖训练。

### 单元测试

- 文本抽取语义
- `basic_moderation` 规则命中与放行
- `hashlinear` 模型存在 / 缺失时的决策
- 缓存命中与跳过下游审核
- 并发限制语义
- `ApiError` 审核错误映射

### 集成测试

- 审核未启用时请求正常透传
- `basic_moderation` 命中时阻断
- `hashlinear` 命中时阻断
- `hashlinear` 缺失时回退到 LLM
- LLM 审核违规时阻断
- LLM 并发超限时返回 `concurrency_limit`
- 审核通过时不影响现有 HTTP 代理黑盒行为

## 风险与控制

### 风险 1：审核文本抽取不稳定

影响：

- 缓存失效
- 本地模型与 LLM 结果不一致

控制：

- 把文本抽取独立成模块
- 先补稳定的单元测试和跨格式输入样例

### 风险 2：把训练逻辑提前耦合进运行时

影响：

- 范围失控
- 审核上线依赖训练链路

控制：

- 本阶段禁止引入训练和调度
- `hashlinear` 只允许做已存在模型的推理

### 风险 3：审核引入后破坏现有 HTTP 行为

影响：

- 已完成的协议对齐回退

控制：

- 保持审核逻辑只在请求进入上游前执行
- 审核通过路径不修改现有代理响应链路
- 每次改动后继续跑现有三组 HTTP 回归

## 实施阶段划分

### 第一阶段

实现运行时审核闭环：

- 基础审核
- 智能审核编排
- `hashlinear` 推理
- LLM 审核
- 并发限制
- 缓存

### 第二阶段

实现训练与调度相关能力：

- 样本写入
- 模型训练
- 后台调度
- 模型导出 / runtime
- 多 profile 扩展

## 结论

本设计将“AI 审核调 LLM + `hashlinear`”明确限定为**运行时审核闭环**，并刻意把训练、调度和存储延后。这样可以在不破坏当前 HTTP 主链路稳定性的前提下，尽快让 Rust 版具备与 Python 版同层级的请求时审核能力。
