use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::config::Settings;
use crate::profile::ModerationProfile;
use crate::sample_rpc::SampleRpcConfig;
use crate::training::{evaluate_training_need, fetch_training_sample_count_via_rpc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrainingSubprocessCommand {
    pub program: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ScannedProfile {
    pub profile_name: String,
    pub profile: ModerationProfile,
}

#[derive(Debug, Clone)]
pub struct PlannedTrainingAction {
    pub profile_name: String,
    pub sample_count: usize,
    pub reason: &'static str,
}

pub fn list_profiles(root_dir: &Path) -> Result<Vec<String>> {
    let profiles_dir = root_dir.join("configs").join("mod_profiles");
    if !profiles_dir.exists() {
        return Ok(Vec::new());
    }

    let mut profiles = fs::read_dir(&profiles_dir)
        .with_context(|| format!("failed to read {}", profiles_dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let file_type = entry.file_type().ok()?;
            if !file_type.is_dir() {
                return None;
            }

            let profile_name = entry.file_name().to_string_lossy().into_owned();
            let profile_json = entry.path().join("profile.json");
            profile_json.exists().then_some(profile_name)
        })
        .collect::<Vec<_>>();
    profiles.sort();
    Ok(profiles)
}

pub fn scan_profiles(root_dir: &Path) -> Result<Vec<ScannedProfile>> {
    list_profiles(root_dir)?
        .into_iter()
        .map(|profile_name| {
            let profile = ModerationProfile::load(root_dir, &profile_name)?;
            Ok(ScannedProfile {
                profile_name,
                profile,
            })
        })
        .collect()
}

pub fn cooldown_allows_training(
    profile: &ModerationProfile,
    cooldown_minutes: u64,
    now_unix_secs: u64,
) -> Result<bool> {
    if cooldown_minutes == 0 {
        return Ok(true);
    }

    let Some(status) = profile.training_status() else {
        return Ok(true);
    };
    let Some(state) = status.get("status").and_then(|value| value.as_str()) else {
        return Ok(true);
    };
    if state != "failed" {
        return Ok(true);
    }

    let Some(timestamp) = status.get("timestamp").and_then(|value| value.as_u64()) else {
        return Ok(true);
    };

    let elapsed_secs = now_unix_secs.saturating_sub(timestamp);
    let cooldown_secs = Duration::from_secs(cooldown_minutes.saturating_mul(60)).as_secs();
    Ok(elapsed_secs >= cooldown_secs)
}

pub fn build_training_subprocess_command(
    root_dir: &str,
    profile_name: &str,
    allowed_cpus: &str,
) -> Result<TrainingSubprocessCommand> {
    let current_exe = std::env::current_exe().context("failed to resolve current executable")?;
    let working_dir = PathBuf::from(root_dir);

    Ok(TrainingSubprocessCommand {
        program: "systemd-run".to_string(),
        args: vec![
            "--scope".to_string(),
            "-p".to_string(),
            format!("AllowedCPUs={allowed_cpus}"),
            "--working-directory".to_string(),
            working_dir.display().to_string(),
            current_exe.display().to_string(),
            "train-profile".to_string(),
            profile_name.to_string(),
        ],
    })
}

pub async fn plan_training_round(
    root_dir: &Path,
    settings: &Settings,
    now_unix_secs: u64,
) -> Result<Vec<PlannedTrainingAction>> {
    let rpc = SampleRpcConfig::from_settings(settings)?;
    let mut planned = Vec::new();

    for scanned in scan_profiles(root_dir)? {
        if !cooldown_allows_training(
            &scanned.profile,
            settings.training_scheduler_failure_cooldown_minutes,
            now_unix_secs,
        )? {
            continue;
        }

        let sample_count = match fetch_training_sample_count_via_rpc(&rpc, &scanned.profile).await {
            Ok(sample_count) => sample_count,
            Err(err) => {
                warn!(
                    profile = %scanned.profile_name,
                    error = %err,
                    "skipping profile during scheduler round because training sample count is unavailable"
                );
                continue;
            }
        };
        let model_mtime = fs::metadata(scanned.profile.training_model_path())
            .ok()
            .and_then(|meta| meta.modified().ok());
        let now = std::time::UNIX_EPOCH + Duration::from_secs(now_unix_secs);
        let decision = evaluate_training_need(&scanned.profile, sample_count, model_mtime, now)?;
        if decision.should_train {
            planned.push(PlannedTrainingAction {
                profile_name: scanned.profile_name,
                sample_count,
                reason: decision.reason,
            });
        }
    }

    Ok(planned)
}

pub async fn spawn_training_subprocess(
    settings: &Settings,
    profile_name: &str,
) -> Result<std::process::ExitStatus> {
    let command = build_training_subprocess_command(
        &settings.root_dir.display().to_string(),
        profile_name,
        &settings.training_subprocess_allowed_cpus,
    )?;
    let status = std::process::Command::new(&command.program)
        .args(&command.args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .status()
        .with_context(|| format!("failed to run training subprocess for {}", profile_name))?;
    Ok(status)
}

pub async fn run_scheduler_once(settings: &Settings) -> Result<Vec<PlannedTrainingAction>> {
    let now_unix_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::ZERO)
        .as_secs();
    let planned = plan_training_round(&settings.root_dir, settings, now_unix_secs).await?;

    for action in &planned {
        info!(
            profile = %action.profile_name,
            sample_count = action.sample_count,
            reason = action.reason,
            "scheduler selected profile for training"
        );

        let status = spawn_training_subprocess(settings, &action.profile_name).await?;
        if !status.success() {
            warn!(
                profile = %action.profile_name,
                status = ?status.code(),
                "training subprocess exited unsuccessfully"
            );
        }
    }

    Ok(planned)
}

pub fn start_scheduler_loop(settings: Settings) -> Option<JoinHandle<()>> {
    if !settings.training_scheduler_enabled {
        return None;
    }

    Some(tokio::spawn(async move {
        let interval = Duration::from_secs(settings.training_scheduler_interval_minutes.max(1) * 60);
        loop {
            if let Err(err) = run_scheduler_once(&settings).await {
                warn!(error = %err, "training scheduler round failed");
            }
            tokio::time::sleep(interval).await;
        }
    }))
}
