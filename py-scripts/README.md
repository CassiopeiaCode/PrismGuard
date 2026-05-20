# py-scripts

Python 脚本工具集（用 `uv` 管理依赖与虚拟环境），用于在**不重新编译 Rust 主程序**的情况下，复用当前运行中的 PrismGuard 服务做诊断与评测。

## 快速开始

在仓库根目录执行：

```bash
cd py-scripts
uv venv
uv pip install -e .
```

运行离线评测脚本（示例，直接只读打开 `history.rocks`，本地推理，计算多阈值 F1）：

```bash
uv run python eval_profile.py --profile 4claudecode
```

快速试跑（限制样本数）：

```bash
uv run python eval_profile.py --profile 4claudecode --limit 2000
```

说明：
- 默认走 `scikit-learn` 的 `HashingVectorizer` 快速路径（更接近 Rust/训练时的 Hashing 行为，也快很多）
- 如果环境缺少依赖，会自动回退到纯 Python 慢路径（用于小样本调试）

## Inline 审核链路延时诊断

针对“开启本地模型 + LLM 审核后，请求为什么变慢”的专项诊断脚本：

```bash
cd py-scripts
uv run python bench_inline_moderation.py --profile 4claudecode
uv run python bench_inline_moderation.py --profile 4claudecode --mode stream
uv run python bench_inline_moderation.py --profile 4claudecode --base-url http://127.0.0.1:8000
uv run python bench_inline_moderation.py --profile 4claudecode --request-file ./request.json --verbose
```

脚本行为：
- 优先从运行中的 `Prismguand-Rust` 进程环境读取 `HOST`/`PORT`
- 读取失败时回退到仓库根目录 `.env`
- 仍未命中时回退到 `http://127.0.0.1:8000`
- 对候选地址执行 `/healthz` 探活，失败时要求显式传 `--base-url`
- 自动启动本地模拟 upstream，同时覆盖非流式 JSON 和流式 SSE
- 通过 inline config 请求当前运行中的 PrismGuard，并启用指定 `profile` 的 `smart_moderation`
- 默认只随机化 `messages` 中第一条 `user` 纯文本内容，追加 `benchmark_nonce=<随机值>`，避免重复请求命中审核缓存
- 输出 `total_latency_ms`、`first_event_latency_ms`、`proxy_overhead_ms` 以及详细 troubleshooting 建议

常用参数：
- `--profile`：必填，`configs/mod_profiles/<profile>/profile.json` 的目录名
- `--mode {all,non-stream,stream}`：默认同时跑两种模式
- `--request-file`：自定义请求体 JSON；不传时使用内置最小 OpenAI Chat 请求
- `--upstream-delay-ms`：模拟非流式上游延时
- `--stream-first-token-delay-ms`：模拟流式首事件延时
- `--stream-tail-delay-ms`：模拟流式尾部延时
- `--timeout-secs`：代理请求和调试接口超时
- `--verbose`：打印更多 profile 诊断信息
- `--disable-randomize-user-text`：关闭默认的随机 `user` 文本后缀，回到固定文本模式做对照

输出中的排障信息会包含：
- 地址发现过程与最终命中的主程序地址
- `profile` 的 AI 审核配置摘要
- 本地模型产物缺失情况
- `/debug/profile/<profile>` 返回的关键状态
- 代理返回体或错误体摘要
- 流式 SSE 预览和 `[DONE]` 是否出现
- 基于耗时形状的疑难排查建议

## 单条耗时对比（Python vs Rust 在线服务）

基准脚本：

```bash
uv run python bench_single.py --profile 4claudecode
```

说明：
- Python 侧测的是“纯本地推理”耗时（不含 HTTP、不含 DB 扫描）
- Rust 侧用现有的 `/debug/profile/<profile>/metrics?sample_size=1` 作为近似（包含 HTTP + 服务端 DB 读取 + JSON 编解码开销），因此是偏保守的上界

## 修复 RocksDB 元数据（meta:count 等）

当程序异常终止，可能出现 `meta:*`（计数/next_id）与实际 `sample:*` 不一致，进而导致采样、指标、查找性能退化。

修复脚本：`repair_rocks_meta.py`

特性：
- 先检查 RocksDB 的 `LOCK` 文件能否加锁（被占用会直接报错退出）
- 默认 **dry-run**（只打印计划，不写入）
- 可选重建 `text_latest:*` 指针（`--fix-pointers`）

示例：

```bash
cd py-scripts
uv run python repair_rocks_meta.py --profile 4claudecode
uv run python repair_rocks_meta.py --profile 4claudecode --apply
uv run python repair_rocks_meta.py --profile 4claudecode --fix-pointers --apply
```
