use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

use crate::profile::ModerationProfile;

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
