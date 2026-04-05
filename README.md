# PrismGuard Rust

## 1. 项目概述

`Prismguand-Rust` 是 PrismGuard 体系里的 Rust 版代理与智能审核运行时。这个仓库不是一个“纯 HTTP 转发器”，也不是一个只做关键词拦截的小工具，而是一个围绕 AI API 代理、协议兼容、请求审核、样本存储、本地模型训练和训练调度建立起来的综合服务。当前代码已经形成一条完整主链路：客户端把请求发给 Rust 服务，Rust 服务完成配置解析、请求格式识别、必要的格式转换、基础审核与智能审核、上游转发、普通 JSON 响应转换、流式 SSE 转码，最后把结果回送给客户端。在开启默认特性的情况下，它还会把审核历史与训练样本保存在 RocksDB 中，并支持通过 Unix Socket 暴露训练样本 RPC，进一步驱动训练子进程与后台调度器。

这个项目的直接对齐对象是 Python 参考实现 `/services/apps/GuardianBridge-UV`。Rust 版并不追求“逐行复刻 Python”，而是追求更重要的目标：保证对外行为与 Python 版等价，特别是在请求格式语义、路径改写方式、错误协议、智能审核决策、历史库兼容与流式协议转换这些最容易让客户端感知差异的地方保持一致。换句话说，用户真正依赖的是协议和行为，不是内部函数名；Rust 版允许采用与 Python 不同的内部结构，但不能让客户端看到不同的外部契约。

从源码结构可以看出，这个仓库已经不再处于“搭架子”的阶段。`src/main.rs` 负责服务模式与训练子进程模式切换、HTTP 服务启动、训练样本 RPC 准备、调度器启动和优雅退出；`src/proxy.rs` 是代理主链路入口，串起请求解析、审核、上游转发与返回阶段；`src/format.rs`、`src/response.rs`、`src/streaming.rs` 则共同承担协议兼容层；`src/moderation/` 下的模块实现基础审核、文本抽取、智能审核与本地模型推理；`src/storage.rs`、`src/sample_rpc.rs`、`src/training.rs`、`src/scheduler.rs` 则构成训练与样本闭环。这说明 README 不能只停留在“如何运行”层面，而必须把项目的系统角色、模块关系和运行边界写清楚。

## 2. 设计目标与非目标

这个仓库最核心的目标可以概括为五条。

第一，维持多种主流 AI API 请求协议之间的兼容与转换能力。当前代码已经覆盖四种主要请求族：`openai_chat`、`openai_responses`、`claude_chat`、`gemini_chat`。源码不是简单地靠字符串替换做协议映射，而是先把外部请求解析为统一内部表示，再按目标协议重组出去。这样做的意义在于，像 system/instructions、tool 声明、tool_choice、图片块、多段文本、function call、function result 这些高歧义字段可以在统一抽象层处理，避免每一对协议之间手写 N x N 互转逻辑。

第二，维持 HTTP 外部行为与 Python 参考实现的一致性。这里的一致性不是指“状态码、日志、错误文案每个字都一样”，而是指客户端真正感知的那些内容必须一致或等价，例如：请求能否被正确识别、严格解析失败时错误协议是否正确、上游路径改写是否符合预期、压缩请求体是否能被接受、非 200 上游响应如何透传、SSE 事件在不同协议间能否被增量消费、结束标记是否正确出现一次等。

第三，在请求进入上游之前完成可配置的审核决策。Rust 版当前支持两层审核：一层是 `basic_moderation`，主要依赖关键词文件做快速拦截；另一层是 `smart_moderation`，它结合历史缓存、并发限制、本地轻量模型与 LLM 审核实现更复杂的决策链。审核不是一个独立旁路服务，而是直接嵌入代理主链路，在 `src/proxy.rs` 中与请求文本抽取、格式识别结果和错误协议映射紧密协作。

第四，把审核历史、训练样本和本地模型运行时接起来，形成一个“能用、能学、能更新”的闭环。项目中不仅有运行时审核，还有 RocksDB/SQLite 兼容存储、样本 RPC、训练子进程入口、按 profile 维度的训练配置、训练状态写回和后台调度器。这意味着本仓库既是网关，也是审核运行时和训练管理器。

第五，尽量保留可调试性。当前路由中除了 `/healthz`、`/docs`、`/redoc` 之外，还提供了 `/debug/settings`、`/debug/proxy-config/:key`、`/debug/profile/:profile`、`/debug/profile/:profile/metrics`、`/debug/url-config` 以及存储调试路由。对一个要做协议兼容与历史存储的代理来说，这些调试面不是锦上添花，而是排查兼容性问题的必要工具。

与这些目标对应，本仓库也有明确的非目标。

第一，它不追求与 Python 实现逐行对照，代码组织方式允许不同。第二，它不承诺日志文本、异常文案、内部中间结构与 Python 完全一样。第三，它当前并未实现 `TRAINING_DATA_RPC_TRANSPORT=tcp`，源码里只把 TCP 作为一个已识别但未落地的传输枚举；真正可用的路径是 Unix Socket。第四，它也没有承诺“所有能想到的上游协议全部支持”，当前重点就是四类主流协议。第五，它不把“内部优雅”置于兼容性之上，如果某个外部语义已经通过测试锁定，那么内部实现需要围绕这个语义收敛，而不是为了重构而重构。

## 3. 当前已实现的核心能力

结合 `Cargo.toml`、源文件和测试套件，当前项目已落地的能力可以分成以下几个层面。

### 3.1 协议识别与请求转换

`src/format.rs` 是整个请求兼容层的核心。它首先定义了 `RequestFormat`，把系统支持的请求族枚举为 `OpenAiChat`、`ClaudeChat`、`OpenAiResponses`、`GeminiChat`。`process_request` 接受配置、路径、请求头和 JSON 请求体，输出 `RequestPlan`。这个计划对象里包含源格式、目标格式、转换后的请求体、转换后的上游路径以及流式标记。只要 `format_transform.enabled` 被打开，代理就会进入这条兼容路径。

格式识别并不是无条件进行。代码会先读取 `from` 配置，根据配置决定候选集合。如果用户写的是单一格式，就只尝试这一种；如果写的是数组，就在候选数组中检测；如果写的是 `auto` 或未指定，就按默认顺序检测。`strict_parse` 决定检测失败后的行为：关闭时走尽量透传，打开时返回明确的格式错误。这个错误会继续在 `src/proxy.rs` 中被包装成适合当前客户端协议的审核式错误响应，避免让客户端拿到不属于它协议族的错误结构。

解析成功后，请求会进入统一内部表示 `InternalRequest`。它把消息、模型、stream、tools、tool_choice 和额外字段统一存放；消息体中的文本、工具调用、工具结果、图片 URL 也被标准化为 `InternalContentBlock`。随后，`emit_request` 会根据目标协议重新生成 JSON。这个架构允许项目在 OpenAI Chat、Claude Messages、OpenAI Responses、Gemini GenerateContent 这些协议之间做较稳的互转，而不是写四套完全独立的请求拼装器。

从测试和实现能看到，项目已经认真处理了多个容易踩坑的细节，包括但不限于：多段 system message 在 OpenAI Chat 转 Claude 时如何折叠成单一 `system` 字符串；图片字段如何在不同协议的结构差异中归一；OpenAI Responses 的 `input`、`instructions`、`response_format` 与工具声明如何转成统一表达；Claude 的内容块和 tool use/tool result 如何回写到其他协议；Gemini 请求路径在流式场景下如何附加 `?alt=sse`。这些都说明它不是一个“只支持 happy path”的实验仓库。

### 3.2 普通 JSON 响应转换

`src/response.rs` 负责非流式 JSON 响应的转换。当前实现里最明确、测试覆盖也最集中的两类转换是：

1. `openai_chat -> claude_chat`
2. `openai_responses -> openai_chat`

实现层面会先验证上游响应是否是对象，随后读取关键字段，例如 `choices[0].message`、`output`、`usage`、`finish_reason` 等，再按目标协议重组。比如在 `openai_chat -> claude_chat` 路径里，会把 assistant 文本、`tool_calls` 和 message content 中的 `tool_use` 块折叠成 Claude 风格的 `content` 数组，同时把 usage 映射成 Claude 的 token 字段；在 `openai_responses -> openai_chat` 路径里，则会把 Responses 的 `output_text`、图片部件、reasoning 项、function call 等内容还原到 Chat Completions 能理解的 message 结构中。

这部分的一个重要特点是“尽量保留额外字段”。代码不会只保留最低限度的核心字段，而是会在不破坏目标协议的前提下，把未被显式消费的额外顶层字段继续带下去。这种策略与 Python 兼容工程常见的做法一致：先保证标准字段正确，再尽量降低信息损失。

### 3.3 SSE 流式协议转换

`src/streaming.rs` 是这个仓库非常关键的一层。很多代理项目做到了普通 JSON 兼容，却在流式响应上明显掉队，因为各家协议的事件粒度、事件名、结束语义、工具调用增量传输方式都不一样。Rust 版的处理方式是先把上游 SSE 解码成统一的内部事件流，再把内部事件交给目标协议专属 sink 重新编码。

内部事件抽象包括：`Start`、`TextDelta`、`ToolCallStart`、`ToolCallArgsDelta`、`Final`、`Done`。这六种事件覆盖了主流 LLM SSE 场景中的核心状态变化。`StreamTranscoder` 在读取原始 SSE 帧后，会根据源协议调用对应的解码逻辑，例如 OpenAI Chat、OpenAI Responses、Claude、Gemini 各有自己的 decode 路径。目标协议也并非直接 `if/else` 拼 JSON，而是封装成 `OpenAiChatSink`、`OpenAiResponsesSink`、`ClaudeSink`、`GeminiSink` 这四种 sink，让“解码”和“重新发射”各自关注自己的职责。

从代码和测试可以确认，当前流式层已经覆盖多项高价值兼容细节：工具调用的起始与参数增量分离传输；当元信息与参数增量乱序到达时的缓冲与回放；`finish_reason` 在不同协议字段中的映射；usage 信息在终态事件中的归一；`[DONE]` 只发一次；无需转换时允许直接透传；在 `delay_stream_header` 场景下优先延迟响应头，便于在真正开始流式输出前把错误转成普通响应。这些能力对于把 Rust 版替换进现有生产路径尤为关键。

### 3.4 基础审核与智能审核

审核层位于 `src/moderation/`。当前模块划分比较清晰：

- `basic.rs` 负责关键词审核与文件热更新感知。
- `extract.rs` 负责从各类协议请求体提取审核文本。
- `smart.rs` 负责编排缓存、并发限制、本地模型、LLM 回退与结果缓存。
- `bow.rs`、`fasttext.rs`、`hashlinear.rs` 则分别承载三类本地运行时模型的推理能力。

`basic_moderation` 的职责很直接：从关键词文件加载规则，做不区分大小写的匹配，并在文件内容变化后自动刷新缓存。对应测试验证了禁用状态、关键词命中、关键词文件修改后缓存刷新等行为。

`extract_text_for_moderation` 的意义被很多系统低估，但这里实现得很扎实。不同协议下“真正应参与审核的文本”并不相同，例如 OpenAI Chat 里要拼接 system 与 user 文本，但要忽略某些工具输出；Claude 里 system 块和 message content 块需要展开；OpenAI Responses 里 `instructions`、`input_text`、`output_text`、`function_call_output` 都可能构成审核语料。该模块通过协议分支把这些文本提取逻辑固定下来，从而避免“开不开格式转换会影响审核结果”这种隐蔽错误。

`smart_moderation` 是更复杂的一层。它首先看缓存，其次检查 LLM 并发配额，再尝试本地模型推理。如果本地模型可以明确给出低风险或高风险结论，就直接返回；如果模型缺失、运行时不可用或结果处在不确定区间，则回退到 LLM 审核。这里有几个特别值得注意的实现点：

1. LLM 并发限制通过信号量控制，达到上限时立即返回，而不是排队等待。
2. profile 的 AI 配置支持逗号分隔的候选模型列表，运行时可以在重试中选择不同候选。
3. 历史命中优先复用，审完后的结果也会回写历史库。
4. `ai_review_rate` 可用于强制提升 AI 审核比例，即使本地模型可用，也能让系统周期性抽样走 LLM。

当前 profile 样例 `configs/mod_profiles/4claudecode/profile.json` 能直观看到这些配置项：AI provider、base URL、模型候选、超时、重试、API Key 环境变量、prompt 模板、AI 审核概率、本地模型类型以及不同训练器的参数都已经有完整结构。

### 3.5 RocksDB 历史样本存储与 SQLite 迁移

`src/storage.rs` 说明本项目并不把历史样本当作“可有可无的日志”。存储层实现了真实的数据布局、采样策略和迁移逻辑。`SampleStorage` 支持读写打开与只读打开，且在打开时会自动尝试做 legacy SQLite 到 RocksDB 的迁移。当前项目使用 `rocksdb` crate，并显式设置比较器，以兼容参考实现使用的 key 排序语义。

从暴露的接口看，存储层至少支持以下能力：

- 读取元数据与样本总数。
- 获取按标签分布统计的样本数量。
- 通过 sample id 或文本查找样本。
- 加载最新样本、随机样本、平衡样本、平衡最新样本、平衡随机样本。
- 对超量样本执行清理。
- 维护 `text_latest:*` 指针，支持文本命中复用。

这部分实现与训练链路紧密耦合，因为训练器并不直接扫整个 RocksDB，而是通过统一采样策略拉取训练样本。README 必须明确这一点：本仓库的存储不是泛用 KV，而是围绕审核历史与模型训练场景设计的专用持久层。

### 3.6 样本 RPC、训练子进程与后台调度器

`src/sample_rpc.rs`、`src/training.rs`、`src/scheduler.rs` 共同组成了训练闭环。

样本 RPC 负责通过 Unix Socket 提供少量但关键的操作：获取样本数、清理超量样本、加载平衡样本、加载平衡最新样本、加载平衡随机样本。协议本身是基于一行一个 JSON 的请求/响应格式。当前代码对 `TRAINING_DATA_RPC_TRANSPORT` 识别了 `unix` 与 `tcp`，但只有 Unix 路径真正可用，因此实际部署时应坚持用 Unix Socket。

训练模块负责三件事。第一，判断是否应该训练，依据包括样本数是否达到最小要求、模型文件是否存在、距离上次训练是否超过间隔。第二，通过样本 RPC 拉取样本并清理过量数据。第三，按 profile 的 `local_model_type` 决定使用 `hashlinear`、`bow` 或 `fasttext` 训练器，并把运行时产物写回 profile 目录。训练期间还会把状态写到 `.train_status.json` 之类的状态文件，供调试和调度器读取。

调度器则负责周期性扫描 `configs/mod_profiles/*/profile.json`，找出所有 profile，调用样本 RPC 获取各自样本数量，根据 cooldown 与训练决策判断哪些 profile 需要训练，然后通过 `systemd-run --scope` 启动受限 CPU 的训练子进程。代码默认值体现了单核保守策略：`TRAINING_SCHEDULER_INTERVAL_MINUTES=10`、`TRAINING_SCHEDULER_FAILURE_COOLDOWN_MINUTES=30`、`TRAINING_SUBPROCESS_ALLOWED_CPUS=0`。这跟文档目录中的设计与计划保持一致，也说明作者非常在意训练任务不要抢占太多资源。

## 4. 运行模式与启动流程

### 4.1 两种启动模式

`src/main.rs` 里定义了两个启动模式：

1. 服务器模式
2. `train-profile` 子命令模式

当命令行没有额外参数时，程序进入服务器模式。它会：

- 获取当前工作目录作为根目录。
- 从根目录加载 `.env` 与环境变量。
- 初始化 tracing 日志。
- 尝试降低进程优先级到 `nice=19`。
- 准备样本 RPC 运行环境并清理旧 Unix Socket。
- 启动样本 RPC 服务。
- 启动后台调度器。
- 构建 `reqwest` HTTP 客户端。
- 组装 Axum 路由并监听 `HOST:PORT`。

当命令行为 `train-profile <profile-name>` 时，程序不启动 HTTP 服务，而是直接进入训练子进程逻辑。调度器正是依赖这个入口拉起单个 profile 的训练。

### 4.2 启动时的重要副作用

很多 README 会遗漏运行时副作用，但这个仓库不能省略。

第一，服务会尝试把当前进程的 nice 值降到 `19`。如果权限不足，会记录 warning，但不会中断启动。第二，若开启样本 RPC 且使用 Unix Socket，服务会在启动前删除旧 socket 文件，以避免上次异常退出留下脏状态。第三，默认构建特性会把存储调试和训练闭环一起编译进来，因此第一次编译并不轻量，尤其 `rocksdb` 相关依赖会显著拖长冷启动构建时间。

## 5. 配置体系

### 5.1 全局服务配置

全局配置在 `src/config.rs` 中定义，由 `Settings::load` 从工作目录下的 `.env` 和当前进程环境变量加载。当前已实现的关键变量如下：

- `HOST`：监听地址，默认 `0.0.0.0`
- `PORT`：监听端口，默认 `8000`
- `DEBUG`：调试开关，默认 `true`
- `LOG_LEVEL`：日志级别，默认 `INFO`
- `ACCESS_LOG_FILE`
- `MODERATION_LOG_FILE`
- `TRAINING_LOG_FILE`
- `TRAINING_DATA_RPC_ENABLED`：默认 `true`
- `TRAINING_DATA_RPC_TRANSPORT`：默认 `unix`
- `TRAINING_DATA_RPC_UNIX_SOCKET`：默认 `<root>/run/sample-store.sock`
- `TRAINING_SCHEDULER_ENABLED`：默认 `true`
- `TRAINING_SCHEDULER_INTERVAL_MINUTES`：默认 `10`
- `TRAINING_SCHEDULER_FAILURE_COOLDOWN_MINUTES`：默认 `30`
- `TRAINING_SUBPROCESS_ALLOWED_CPUS`：默认 `0`

除了这些显式字段，`Settings` 还会保留完整的 `env_map`。这是代理配置的重要基础，因为上游代理 URL 支持 `!ENV_KEY${upstream}` 形式，运行时可以把某个环境变量解析成 JSON 配置，动态生成一条代理规则。

### 5.2 代理配置

本项目的代理入口不是传统的固定配置文件路由，而是把“配置 + 上游 URL”编码在请求路径里。`src/routes.rs` 中的 `parse_url_config` 会解析这类路径。常见形式有两种：

1. `{percent-encoded-json-config}${upstream-url}`
2. `!ENV_KEY${upstream-url}`

这意味着客户端可以按请求维度决定是否做格式转换、是否开启审核、是否启用延迟流式头部等。调试接口 `/debug/url-config?value=...` 可以帮助开发者先验证这段路径能否被正确解析。

一个典型的配置片段可能像这样：

```json
{
  "format_transform": {
    "enabled": true,
    "strict_parse": true,
    "from": "claude_chat",
    "to": "openai_chat",
    "delay_stream_header": true
  },
  "basic_moderation": {
    "enabled": true,
    "keywords_file": "configs/keywords.txt",
    "error_code": "BASIC_MODERATION_BLOCKED"
  },
  "smart_moderation": {
    "enabled": true,
    "profile": "4claudecode"
  }
}
```

### 5.3 Profile 配置

审核与训练的细粒度配置来自 `configs/mod_profiles/<profile>/profile.json`。`src/profile.rs` 定义了它的结构，主要分为这些块：

- `ai`
- `prompt`
- `probability`
- `local_model_type`
- `bow_training`
- `fasttext_training`
- `hashlinear_training`

`ai` 块控制 LLM 审核的 provider、base URL、候选模型、API key 环境变量、超时和重试次数；`prompt` 控制模板文件与文本截断长度；`probability` 控制 `ai_review_rate`、低风险阈值、高风险阈值和随机种子；三个 `*_training` 块分别控制对应训练器的最小样本数、重训间隔、采样方式、特征与超参数。README 里有必要强调一点：虽然当前三个本地模型运行时都已存在，但实际使用哪一种，完全由 profile 的 `local_model_type` 决定。

## 6. 请求生命周期

为了让第一次接手项目的人快速理解系统，最有效的方法不是背 API 列表，而是按请求经过系统的顺序看一遍。

### 第一步：进入路由

请求会先命中 `src/routes.rs` 中定义的 Axum 路由。`/healthz`、`/openapi.json`、`/docs`、`/redoc` 和一系列 `/debug/*` 路由是固定的；其余 `/` 与 `/*cfg_and_upstream` 路由都会进入代理处理函数 `proxy_entry_root` 或 `proxy_entry`。

### 第二步：解析配置与上游 URL

`src/proxy.rs` 通过 `parse_url_config` 解析路径，把请求拆成配置对象和真正的上游 URL。随后它会提取上游 base URL、上游 path 和保留前缀，为后续路径改写做准备。

### 第三步：读取与解压请求体

代理只会对有意义的方法和非空 body 尝试解析 JSON。若请求体使用 `gzip`、`deflate` 或 `br` 压缩，会先解压。解压成功但 JSON 解析失败时，并不总是立即报错，而是按当前语义尝试透传。这一点非常重要，因为兼容代理不能把本来应透传的请求过早转成失败。

### 第四步：格式识别与请求重写

如果配置开启了 `format_transform`，`process_request` 会基于路径、头和 body 判断源协议，并在必要时生成目标协议 JSON 和新的上游路径。如果关闭转换，计划对象保留原始 path 和 body，只把流式标记和一些推断结果带下去。

### 第五步：提取审核文本并做审核决策

若配置启用了 `basic_moderation` 或 `smart_moderation`，代理会根据当前源格式或推断格式，从请求体抽取审核文本。基础审核先跑；若命中，直接返回阻断错误。智能审核随后运行，可能命中缓存、可能由本地模型直接决策，也可能调用远端 LLM。若审核判定违规，代理会返回带有 `moderation_details` 的错误响应，不再继续请求上游。

### 第六步：构建上游请求

代理会过滤请求头，移除部分不应直接转发的头，并按需要把转换后的 JSON 重新编码发往上游。如果请求方法不是应该转发 JSON 的那类方法，或者原始 body 不应被改写，则会回退到原始 body。这里的细节关系到“兼容代理是否偷偷篡改了请求”，所以实现比较保守。

### 第七步：处理上游响应

若上游返回普通 JSON 响应，代理会判断是否需要按客户端目标协议做 JSON 转换；若上游返回 SSE，则进入流式转码逻辑；若上游返回非 200 或非 JSON 内容，则尽量透传原始状态与正文。这样做的核心原则只有一个：该改的时候正确改，不该改的时候少碰它。

## 7. HTTP 路由与调试接口

当前公开路由主要分为四类。

第一类是基础存活与文档路由：

- `/healthz`
- `/openapi.json`
- `/docs`
- `/docs/oauth2-redirect`
- `/redoc`

第二类是运行配置调试：

- `/debug/settings`
- `/debug/proxy-config/:key`
- `/debug/url-config`

第三类是 profile 级调试：

- `/debug/profile/:profile`
- `/debug/profile/:profile/metrics`

第四类是在 `storage-debug` 特性开启且非测试模式下可用的存储调试：

- `/debug/storage/:profile/meta`
- `/debug/storage/:profile/sample/:id`
- `/debug/storage/:profile/find-by-text`

其中 `/debug/profile/:profile` 非常有用，因为它会把 profile 目录、历史库路径、训练状态路径、当前本地模型路径、实时样本数、训练状态中的样本数、训练决策结果和 profile 配置一起返回。对排查“为什么这个 profile 迟迟不训练”“为什么本地模型没生效”这类问题很直接。`/debug/profile/:profile/metrics` 则可以基于随机、最新或平衡采样评估本地模型在历史样本上的 accuracy、precision、recall、F1，这对于观察本地模型是否值得继续保留特别实用。

## 8. 目录结构与模块职责

下面按照当前仓库的实际结构，对关键目录做一个更偏工程视角的说明。

### 根目录

- `Cargo.toml`：crate 定义、特性、依赖与 profile 配置。
- `Cargo.lock`：锁定依赖版本。
- `README.md`：项目说明文档。
- `configs/`：关键词文件、审核 profile、训练状态和运行时模型产物。
- `docs/superpowers/`：本项目内部使用的设计文档与实施计划。
- `run/`：运行期 Unix Socket 等临时文件位置。
- `src/`：业务源码。
- `tests/`：黑盒与行为一致性测试。

### `src/`

- `main.rs`：启动入口、模式切换、调度器与样本 RPC 启动。
- `config.rs`：环境变量加载与全局配置。
- `routes.rs`：路由组装、调试接口和统一错误外壳。
- `proxy.rs`：代理主链路。
- `format.rs`：请求格式识别与请求转换。
- `response.rs`：非流式响应转换。
- `streaming.rs`：SSE 流式解码与再编码。
- `profile.rs`：profile 配置结构与 profile 目录路径约定。
- `storage.rs`：RocksDB 历史存储与 SQLite 迁移。
- `sample_rpc.rs`：训练样本 RPC 协议与 Unix Socket 服务。
- `training.rs`：训练判断、样本拉取、训练器实现、状态落盘。
- `scheduler.rs`：周期扫描 profile、选择训练目标并启动子进程。
- `moderation/`：审核相关子模块。

### `tests/`

测试文件并不是简单的单元测试集合，而是以“行为面”组织的：

- `http_proxy_request_tests.rs`：请求侧兼容。
- `http_proxy_response_tests.rs`：非流式响应兼容。
- `http_proxy_stream_tests.rs`：流式响应兼容。
- `moderation_runtime_tests.rs`：审核抽取、审核拦截、运行时语义。
- `pebble_compat_parity_tests.rs`：历史库兼容。
- `scheduler_tests.rs`：训练调度默认值、冷却逻辑与命令构建。
- `training_tests.rs`：训练相关逻辑。
- `bow_runtime_parity_tests.rs`、`fasttext_runtime_parity_tests.rs`、`hashlinear_runtime_jieba_parity_tests.rs`：本地模型运行时行为。
- `format_process_tests.rs`、`format_runtime.rs`、`format_harness.rs`：请求格式处理内部行为。

从这一点也能看出项目的优先级排序：先锁外部行为，再允许内部实现迭代。

## 9. 快速开始

### 9.1 环境要求

从依赖和代码来看，运行本项目至少需要：

- 稳定可用的 Rust 工具链
- C/C++ 编译环境
- `libclang` 等 RocksDB 相关依赖能成功参与编译
- Linux 环境下可用的 Unix Socket
- 如果要运行调度器训练子进程，最好具备 `systemd-run`

因为默认特性会启用 `rocksdb` 和 `rusqlite`，所以第一次构建时间会比较长。这不是异常，而是存储后端带来的正常成本。

### 9.2 本地构建

最直接的构建方式：

```bash
cargo build
```

如果你只想先确认能否编译主程序，也可以直接：

```bash
cargo run
```

但在这个仓库的历史约定里，重编译与重测试通常会配合固定 `CARGO_TARGET_DIR` 和单核限制，以减少资源竞争和重复编译成本。

### 9.3 本地运行

最简单的启动方式是在仓库根目录准备 `.env`，然后直接：

```bash
cargo run
```

默认监听 `0.0.0.0:8000`。服务启动后可以先检查：

```bash
curl -s http://127.0.0.1:8000/healthz
```

如果返回 `ok=true`、`service=PrismGuard` 等字段，说明服务已经进入可接收请求状态。

### 9.4 训练子命令

对单个 profile 手动触发训练：

```bash
cargo run -- train-profile 4claudecode
```

这个命令会按当前工作目录为根目录读取 profile、连接样本 RPC、拉取样本并执行对应训练器。正常生产路径中，调度器会通过 `systemd-run` 自动拉起这个子命令。

## 10. 代理与调用示例

### 10.1 Claude 请求转 OpenAI Chat

假设客户端想把 Claude 风格请求发给某个 OpenAI Chat 风格上游，可以构造一个 percent-encoded 的 JSON 配置，再把它与上游 URL 拼接在一起。请求大体形态如下：

```bash
curl -X POST \
  "http://127.0.0.1:8000/{ENCODED_CONFIG}\$https://example-upstream/v1/messages" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5",
    "anthropic_version": "2023-06-01",
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

其中 `{ENCODED_CONFIG}` 对应的原始 JSON 可能是：

```json
{
  "format_transform": {
    "enabled": true,
    "strict_parse": true,
    "from": "claude_chat",
    "to": "openai_chat"
  }
}
```

此时代理会把消息体和路径一起改写到 OpenAI Chat 风格上游端点。

### 10.2 使用环境变量托管代理配置

如果不希望每次都在 URL 里塞一长串 percent-encoded JSON，可以把配置放进环境变量，比如 `OPENAI_PROXY_CFG`，然后请求：

```text
http://127.0.0.1:8000/!OPENAI_PROXY_CFG$https://example-upstream/v1/chat/completions
```

代理会把 `OPENAI_PROXY_CFG` 当作 JSON 读取并解析。这样做对多套代理规则轮换很方便。

## 11. 测试与验证策略

这个仓库的测试体系明显偏“黑盒行为验证”，而不是只依赖内部小单元测试。对协议兼容型项目，这是正确方向。

### 11.1 测试重点

当前测试重点主要有四类。

第一类，请求侧兼容。验证压缩请求体解码、路径前缀保留、system message 合并、方法支持等。第二类，响应侧兼容。验证 Responses 与 Chat 之间的普通 JSON 映射，以及 Claude 风格响应重建。第三类，流式侧兼容。验证 SSE 增量文本、工具调用、usage、结束标记与头部延迟行为。第四类，审核、存储、训练与调度。验证关键词热更新、审核文本抽取、历史库兼容、训练决策、调度冷却时间与单核子进程命令构建。

### 11.2 常用测试命令

仓库里现有 README 与计划文档都偏向使用单核约束和固定 target 目录，常见命令可以写成：

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --tests -- --nocapture
```

按测试套件定向执行：

```bash
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_request_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_response_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test http_proxy_stream_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test moderation_runtime_tests -- --nocapture
taskset -c 0 env CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=/tmp/prismguard-target cargo test --test scheduler_tests -- --nocapture
```

如果环境更接近生产调度约束，也可以采用 `systemd-run --scope -p AllowedCPUs=0` 或类似方式执行重测试。

### 11.3 为什么这些测试重要

这个项目的风险不在“函数返回值错了一个布尔值”，而在“某个客户端协议突然被代理改坏了”。这种错误往往只有端到端行为测试才能及时发现。例如：

- 请求路径前缀被错误覆盖，导致业务路由失效。
- Claude 的 system message 被错误放进 message 数组，导致上游 400。
- SSE 工具调用参数分片丢失，客户端拼不回完整函数参数。
- 审核文本抽取遗漏 system/instructions，造成审核漏判。
- 调度器忽略失败冷却时间，导致持续重复训练。

因此，README 必须把“如何验证”写得具体，而不是一句空泛的“运行 cargo test 即可”。

## 12. 默认特性与构建注意事项

`Cargo.toml` 当前定义了一个显式 feature：`storage-debug`，并把它放进默认特性里。意味着默认构建包含：

- RocksDB 历史存储
- SQLite 迁移能力
- 存储调试接口
- 样本 RPC 存储处理

这个设计对日常开发很友好，因为大部分真实能力默认就能使用；但它也意味着构建成本更高。如果未来要做更轻量的编译剥离，可以考虑显式关闭默认特性。不过在当前仓库状态下，README 更应该提醒的是：默认行为已经偏向“真实运行时能力”，而不是一个精简 demo 模式。

此外，`rocksdb` crate 当前启用了 `snappy` 特性。它仍可能触发 `librocksdb-sys` 的本地编译，所以建议在持续开发中固定 `CARGO_TARGET_DIR`，避免每次切目录都重复构建一遍庞大的本地依赖。

## 13. 已知限制与当前边界

基于源码和现有设计文档，可以明确列出以下边界。

第一，`TRAINING_DATA_RPC_TRANSPORT=tcp` 还未实现。配置里可以写，`SampleRpcTransport` 里也有这个枚举，但启动样本 RPC 服务时会明确报未实现，因此实际使用请坚持 Unix Socket。

第二，响应转换当前实现的重点是高价值路径，而不是所有协议之间的全排列。当前最明确的非流式转换是 `openai_chat -> claude_chat` 与 `openai_responses -> openai_chat`；其他情况多数是无需转换时透传或流式层处理。

第三，项目强调外部行为兼容，而不是错误文案完全复制。也就是说，如果你在做回归对比，应该比对协议结构、关键字段和行为路径，而不是拿 Python 的每一段错误文本逐字比。

第四，训练与调度强依赖本地系统能力，尤其 `systemd-run` 与 Unix Socket。在极简容器环境里，需要先确认这些基础设施是否可用。

第五，当前工作流明显假设 Linux 或类 Linux 环境。虽然 Rust 本身可以跨平台，但从 `tokio::net::UnixListener`、`systemd-run`、优先级调整等实现看，本仓库并没有把跨平台作为一等目标。

## 14. 开发与维护建议

如果你准备继续在这个仓库上开发，有几条建议是基于当前代码形态得出的。

第一，新增协议兼容能力时，优先补黑盒测试，再调整 `format.rs`、`response.rs` 或 `streaming.rs`。这种仓库最怕“我觉得这样更对”，但没有行为用例兜底。第二，涉及审核文本抽取或错误协议映射时，不要只改一处。请求格式、审核输入和错误结构是耦合的，必须联动检查 `extract.rs`、`proxy.rs` 和相关测试。第三，存储或训练改动要注意 profile 目录中已有产物和状态文件，别只在理想空目录里验证。第四，调度器改动必须考虑单核、冷却时间和失败路径，否则它会成为最容易悄悄吃掉资源的后台模块。

## 15. 结语

`Prismguand-Rust` 当前已经不是一个“Rust 重写尝试”，而是一个围绕协议兼容、智能审核与训练闭环逐步收口的工程化服务。它的价值不只在于把 Python 逻辑迁到 Rust，更在于用更清晰的模块边界把复杂行为拆开：`format` 处理请求兼容，`response` 和 `streaming` 处理返回兼容，`moderation` 处理审核决策，`storage`/`sample_rpc`/`training`/`scheduler` 处理样本与模型闭环，`routes` 和 `proxy` 则把这些能力稳定地暴露成一个统一代理入口。

如果你把这个仓库当成一个简单网关，很多设计会显得“有点重”；但如果把它放回真实场景，就会发现这些复杂度都有来源：多家模型协议并存、审核要前置、历史结果要复用、本地模型要训练、CPU 要受限、错误要协议正确、SSE 不能乱。这也是为什么这个 README 必须写得足够长，因为项目本身已经具备一套完整系统，而不是几段孤立代码。
