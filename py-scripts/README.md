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
