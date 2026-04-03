# Rust Training Scheduler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Rust 版补齐与 Python `scheduler.py` 等价的训练调度闭环，采用“主进程提供 sample RPC + 后台 scheduler 判定 + 独立子进程单 profile 训练”的结构，并严格保持本地重操作单核执行。

**Architecture:** 主进程继续启动 HTTP 服务和 sample RPC Unix socket 服务；新增后台 scheduler 周期扫描 `configs/mod_profiles/*`，读取 `.train_status.json`、通过 RPC 获取样本数、按现有 `evaluate_training_need` 判定是否应训练。真正训练不在主进程内执行，而是由主进程启动一个新的 Rust 子进程，仅负责单个 profile 的训练和状态写回；子进程通过 Unix socket 访问主进程 sample RPC，避免与主进程直接抢占 RocksDB 写锁。

**Tech Stack:** Rust, Tokio, Axum, Serde JSON, Unix socket sample RPC, `std::process::Command`, 现有 `profile.rs` / `training.rs` / `sample_rpc.rs`

---

## 文件结构

### 新增文件

- `src/scheduler.rs`
  训练调度器主模块，负责 profile 扫描、冷却期判定、调度锁、后台循环、子进程拉起。
- `tests/scheduler_tests.rs`
  调度器核心逻辑测试，覆盖 profile 扫描、状态判断、冷却期、命令拼装、状态文件写回。

### 修改文件

- `src/main.rs`
  启动 scheduler 后台任务，并增加子进程训练入口参数分发。
- `src/config.rs`
  增加 scheduler 相关环境变量配置和默认值。
- `src/training.rs`
  增加单 profile 训练入口、训练状态文件写入、RPC 读取样本数/样本清理辅助函数。
- `src/sample_rpc.rs`
  补齐 `get_sample_count` / `cleanup_excess_samples` 的客户端和服务端使用面，保证主进程可为 scheduler/子进程提供统一只读或清理能力。
- `src/profile.rs`
  增加训练器所需的辅助路径/枚举/模型路径选择方法，避免 scheduler 自己分支拼路径。
- `docs/superpowers/specs/2026-04-03-rust-ai-moderation-runtime-design.md`
  在“非目标/后续阶段”后补一段备注，说明训练调度器已拆出独立实现计划，避免文档状态失真。

---

### Task 1: 补齐 scheduler 配置面与 profile 辅助接口

**Files:**
- Modify: `src/config.rs`
- Modify: `src/profile.rs`
- Test: `tests/scheduler_tests.rs`

- [ ] **Step 1: 写失败测试，固定 scheduler 默认配置和模型路径选择**

```rust
#[test]
fn settings_default_scheduler_uses_single_worker_safe_values() {
    let root_dir = std::path::PathBuf::from("/tmp/prismguard-scheduler-defaults");
    std::env::remove_var("TRAINING_SCHEDULER_ENABLED");
    std::env::remove_var("TRAINING_SCHEDULER_INTERVAL_MINUTES");
    std::env::remove_var("TRAINING_SUBPROCESS_ALLOWED_CPUS");

    let settings = config::Settings::load(&root_dir).expect("load settings");

    assert!(settings.training_scheduler_enabled);
    assert_eq!(settings.training_scheduler_interval_minutes, 10);
    assert_eq!(settings.training_subprocess_allowed_cpus, "0");
}

#[test]
fn profile_training_model_path_matches_local_model_type() {
    let profile = write_profile(
        "scheduler-profile-model-path",
        serde_json::json!({"local_model_type": "hashlinear"}),
    );
    assert_eq!(profile.training_model_path(), profile.hashlinear_model_path());
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests settings_default_scheduler_uses_single_worker_safe_values -- --nocapture`
Expected: FAIL with missing `training_scheduler_*` fields or missing `training_model_path`

- [ ] **Step 3: 为 `Settings` 和 `ModerationProfile` 补最小接口**

```rust
// src/config.rs
pub struct Settings {
    // ...
    pub training_scheduler_enabled: bool,
    pub training_scheduler_interval_minutes: u64,
    pub training_scheduler_failure_cooldown_minutes: u64,
    pub training_subprocess_allowed_cpus: String,
}
```

```rust
// src/profile.rs
impl ModerationProfile {
    pub fn training_model_path(&self) -> PathBuf {
        match self.config.local_model_type.as_str() {
            "fasttext" => self.fasttext_model_path(),
            "hashlinear" => self.hashlinear_model_path(),
            _ => self.bow_model_path(),
        }
    }

    pub fn training_max_db_items(&self) -> Result<usize> {
        match self.config.local_model_type.as_str() {
            "fasttext" => Ok(self.config.fasttext_training.max_db_items),
            "hashlinear" => Ok(self.config.hashlinear_training.max_db_items),
            "bow" => Ok(self.config.bow_training.max_db_items),
            other => Err(anyhow::anyhow!("unsupported local_model_type: {other}")),
        }
    }
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests settings_default_scheduler_uses_single_worker_safe_values profile_training_model_path_matches_local_model_type -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config.rs src/profile.rs tests/scheduler_tests.rs
git commit -m "feat: add scheduler settings and profile helpers"
```

### Task 2: 补齐 sample RPC 的训练调度接口

**Files:**
- Modify: `src/sample_rpc.rs`
- Modify: `src/training.rs`
- Test: `tests/training_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖样本计数与清理 RPC**

```rust
#[tokio::test]
async fn sample_rpc_dispatches_cleanup_excess_samples_with_real_storage() {
    let rocks_dir = temp_rocks_dir("cleanup");
    seed_sample_db(&rocks_dir, &make_samples(8));

    let response = sample_rpc::dispatch_request_with_storage(
        sample_rpc::SampleRpcRequest::CleanupExcessSamples {
            profile: "default".to_string(),
            db_path: rocks_dir.display().to_string(),
            max_items: 4,
        },
    );

    assert!(response.ok);
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test training_tests sample_rpc_dispatches_cleanup_excess_samples_with_real_storage -- --nocapture`
Expected: FAIL with `sample RPC method not implemented yet`

- [ ] **Step 3: 实现 RPC 清理与训练侧辅助函数**

```rust
// src/sample_rpc.rs
SampleRpcRequest::CleanupExcessSamples { db_path, max_items, .. } => {
    let mut storage = crate::storage::SampleStorage::open_read_write(db_path)?;
    let removed = storage.cleanup_excess_samples(*max_items)?;
    Ok(serde_json::json!({ "removed": removed }))
}
```

```rust
// src/training.rs
pub async fn fetch_training_sample_count_via_rpc(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<usize> { /* use GetSampleCount */ }

pub async fn cleanup_training_samples_via_rpc(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<usize> { /* use CleanupExcessSamples */ }
```

- [ ] **Step 4: 运行训练/RPC 测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test training_tests -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/sample_rpc.rs src/training.rs tests/training_tests.rs
git commit -m "feat: add scheduler rpc helpers for sample count and cleanup"
```

### Task 3: 抽出单 profile 训练入口与状态文件协议

**Files:**
- Modify: `src/training.rs`
- Modify: `src/profile.rs`
- Test: `tests/scheduler_tests.rs`

- [ ] **Step 1: 写失败测试，固定训练状态文件协议**

```rust
#[test]
fn training_status_writer_persists_python_style_fields() {
    let profile = write_profile(
        "scheduler-status-shape",
        serde_json::json!({"local_model_type": "hashlinear"}),
    );

    training::write_training_status(
        &profile,
        &serde_json::json!({
            "status": "failed",
            "message": "previous run failed",
            "timestamp": 1_744_000_000u64,
            "profile": profile.profile_name,
            "model_type": "hashlinear"
        }),
    )
    .expect("write status");

    let status = profile.training_status().expect("training status");
    assert_eq!(status["status"], "failed");
    assert_eq!(status["message"], "previous run failed");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests training_status_writer_persists_python_style_fields -- --nocapture`
Expected: FAIL with missing `write_training_status`

- [ ] **Step 3: 增加单 profile 训练入口与状态读写**

```rust
// src/training.rs
pub fn write_training_status(profile: &ModerationProfile, status: &Value) -> Result<()> { /* atomic write */ }

pub async fn run_profile_training(
    rpc: &SampleRpcConfig,
    profile: &ModerationProfile,
) -> Result<HashlinearTrainingOutput> {
    let _ = cleanup_training_samples_via_rpc(rpc, profile).await?;
    let samples = fetch_training_samples_via_rpc(rpc, profile).await?;
    train_hashlinear_runtime(profile, &samples)
}
```

```rust
// status shape
{
  "status": "running|success|failed|skipped",
  "message": "human readable summary",
  "timestamp": 1744000000,
  "profile": "default",
  "model_type": "hashlinear",
  "sample_count": 256
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests training_status_writer_persists_python_style_fields -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training.rs src/profile.rs tests/scheduler_tests.rs
git commit -m "feat: add profile training entrypoint and status writer"
```

### Task 4: 实现 scheduler 核心判定与子进程命令拼装

**Files:**
- Create: `src/scheduler.rs`
- Modify: `src/config.rs`
- Test: `tests/scheduler_tests.rs`

- [ ] **Step 1: 写失败测试，覆盖 profile 扫描、失败冷却期、命令拼装**

```rust
#[test]
fn scheduler_skips_profile_when_failure_cooldown_not_elapsed() {
    let profile = write_profile("scheduler-cooldown", serde_json::json!({"local_model_type": "hashlinear"}));
    training::write_training_status(&profile, &serde_json::json!({
        "status": "failed",
        "message": "boom",
        "timestamp": current_unix_secs(),
        "profile": profile.profile_name,
        "model_type": "hashlinear"
    })).expect("write status");

    let decision = scheduler::cooldown_allows_training(&profile, 30, current_unix_secs()).unwrap();
    assert!(!decision);
}

#[test]
fn scheduler_command_uses_single_core_systemd_scope() {
    let cmd = scheduler::build_training_subprocess_command(
        "/services/apps/Prismguand-Rust",
        "default",
        "0",
    );
    assert_eq!(cmd.program, "systemd-run");
    assert!(cmd.args.iter().any(|arg| arg == "AllowedCPUs=0"));
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests scheduler_command_uses_single_core_systemd_scope -- --nocapture`
Expected: FAIL with missing `scheduler` module

- [ ] **Step 3: 实现调度核心，不执行真实训练**

```rust
// src/scheduler.rs
pub struct TrainingSubprocessCommand {
    pub program: String,
    pub args: Vec<String>,
}

pub fn list_profiles(root_dir: &Path) -> Result<Vec<String>> { /* scan configs/mod_profiles/*/profile.json */ }

pub fn cooldown_allows_training(
    profile: &ModerationProfile,
    cooldown_minutes: u64,
    now_unix_secs: u64,
) -> Result<bool> { /* inspect .train_status.json */ }

pub fn build_training_subprocess_command(
    root_dir: &str,
    profile_name: &str,
    allowed_cpus: &str,
) -> TrainingSubprocessCommand {
    TrainingSubprocessCommand {
        program: "systemd-run".to_string(),
        args: vec![
            "--scope".to_string(),
            "-p".to_string(),
            format!("AllowedCPUs={allowed_cpus}"),
            std::env::current_exe().expect("current exe").display().to_string(),
            "train-profile".to_string(),
            profile_name.to_string(),
        ],
    }
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests scheduler_skips_profile_when_failure_cooldown_not_elapsed scheduler_command_uses_single_core_systemd_scope -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scheduler.rs src/config.rs tests/scheduler_tests.rs
git commit -m "feat: add scheduler decision and subprocess command builder"
```

### Task 5: 接入 `main.rs` 的子进程训练入口和后台 loop

**Files:**
- Modify: `src/main.rs`
- Modify: `src/scheduler.rs`
- Modify: `src/training.rs`
- Test: `tests/scheduler_tests.rs`

- [ ] **Step 1: 写失败测试，固定 CLI 分发和 loop 行为**

```rust
#[tokio::test]
async fn scheduler_run_once_trains_only_eligible_profiles() {
    let root_dir = temp_root("scheduler-run-once");
    write_profile_into(&root_dir, "eligible", serde_json::json!({"local_model_type": "hashlinear"}));
    write_profile_into(&root_dir, "cooldown", serde_json::json!({"local_model_type": "hashlinear"}));

    let actions = scheduler::plan_training_round(&root_dir, &test_settings(), current_unix_secs())
        .await
        .expect("plan round");

    assert!(actions.iter().any(|item| item.profile_name == "eligible"));
    assert!(!actions.iter().any(|item| item.profile_name == "cooldown"));
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests scheduler_run_once_trains_only_eligible_profiles -- --nocapture`
Expected: FAIL with missing planner or background runner

- [ ] **Step 3: 在主程序接入分支与后台任务**

```rust
// src/main.rs
mod scheduler;

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.get(1).map(String::as_str) == Some("train-profile") {
        return training::run_training_subprocess_from_args(&args).await;
    }

    // existing HTTP + sample RPC startup...
    let scheduler_task = scheduler::start_scheduler_loop(settings.clone());
    // graceful shutdown aborts scheduler_task and sample_rpc_task
}
```

```rust
// src/scheduler.rs
pub fn start_scheduler_loop(settings: Settings) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let _ = run_scheduler_once(&settings).await;
            tokio::time::sleep(Duration::from_secs(settings.training_scheduler_interval_minutes * 60)).await;
        }
    })
}
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests scheduler_run_once_trains_only_eligible_profiles -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/main.rs src/scheduler.rs src/training.rs tests/scheduler_tests.rs
git commit -m "feat: start scheduler loop and training subprocess entrypoint"
```

### Task 6: 完成单 profile 子进程训练执行与端到端验证

**Files:**
- Modify: `src/training.rs`
- Modify: `src/scheduler.rs`
- Modify: `tests/training_tests.rs`
- Modify: `tests/scheduler_tests.rs`
- Modify: `docs/superpowers/specs/2026-04-03-rust-ai-moderation-runtime-design.md`

- [ ] **Step 1: 写失败测试，覆盖子进程训练核心协议**

```rust
#[tokio::test]
async fn profile_training_subprocess_marks_success_after_runtime_write() {
    let root_dir = temp_root("scheduler-subprocess-success");
    let profile = write_hashlinear_profile_with_seed_data(&root_dir, "default");
    let rpc = sample_rpc::SampleRpcConfig {
        enabled: true,
        transport: sample_rpc::SampleRpcTransport::Unix,
        unix_socket: root_dir.join("run").join("sample-store.sock"),
    };

    let output = training::run_profile_training(&rpc, &profile).await.expect("train");
    assert!(output.runtime_json_path.exists());

    let status = profile.training_status().expect("status");
    assert_eq!(status["status"], "success");
}
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests profile_training_subprocess_marks_success_after_runtime_write -- --nocapture`
Expected: FAIL because status transitions or subprocess training wrapper are incomplete

- [ ] **Step 3: 完成训练状态迁移和错误分支**

```rust
// src/training.rs
pub async fn run_training_subprocess_from_args(args: &[String]) -> Result<()> {
    // parse profile name
    // load settings/root dir from current dir
    // mark running
    // call run_profile_training
    // on success write success status
    // on error write failed status and return error
}
```

```rust
// src/scheduler.rs
async fn spawn_training_subprocess(...) -> Result<std::process::ExitStatus> {
    // use build_training_subprocess_command()
    // inherit stdout/stderr
    // do not spawn more than one subprocess per profile
}
```

- [ ] **Step 4: 运行最终验证**

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test training_tests -- --nocapture`
Expected: PASS

Run: `systemd-run --scope -p AllowedCPUs=0 cargo test --test scheduler_tests -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/training.rs src/scheduler.rs tests/training_tests.rs tests/scheduler_tests.rs docs/superpowers/specs/2026-04-03-rust-ai-moderation-runtime-design.md
git commit -m "feat: add rust training scheduler and subprocess runtime"
```

---

## Self-Review

- Spec 覆盖检查：
  当前计划只覆盖“训练调度器 + 子进程训练 + sample RPC 协议 + 状态文件”，不扩张到 fastText / BoW 真训练实现，也不碰主请求审核链路，符合当前拆分目标。
- 占位符检查：
  所有 task 都给出了明确文件、测试目标、命令和预期失败/通过状态，没有保留 TBD。
- 类型一致性检查：
  计划统一使用 `training_scheduler_*` 设置字段、`training_model_path()`、`run_profile_training()`、`run_training_subprocess_from_args()` 这些符号，后续实现需保持命名一致。

## 执行说明

- 所有本地编译、测试和任何重操作必须严格单核：
  `systemd-run --scope -p AllowedCPUs=0 ...`
- 子进程训练命令也必须固定通过 `systemd-run --scope -p AllowedCPUs=0` 启动，不能放宽到多核。
- 子 codex 只应修改本计划列出的文件；涉及共享文件时按 task 顺序串行合并，避免并发冲突。
