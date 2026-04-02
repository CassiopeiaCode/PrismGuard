use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub debug: bool,
    pub log_level: String,
    pub access_log_file: String,
    pub moderation_log_file: String,
    pub training_log_file: String,
    #[serde(skip_serializing)]
    pub root_dir: PathBuf,
    #[serde(skip_serializing)]
    pub env_map: HashMap<String, String>,
}

impl Settings {
    pub fn load(root_dir: impl AsRef<Path>) -> Result<Self> {
        let root_dir = root_dir.as_ref().to_path_buf();
        let dotenv_path = root_dir.join(".env");
        if dotenv_path.exists() {
            load_env_file(&dotenv_path)?;
        }

        let env_map = env::vars().collect::<HashMap<_, _>>();

        Ok(Self {
            host: env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8000),
            debug: parse_bool_env("DEBUG", true),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "INFO".to_string()),
            access_log_file: env::var("ACCESS_LOG_FILE")
                .unwrap_or_else(|_| "logs/access.log".to_string()),
            moderation_log_file: env::var("MODERATION_LOG_FILE")
                .unwrap_or_else(|_| "logs/moderation.log".to_string()),
            training_log_file: env::var("TRAINING_LOG_FILE")
                .unwrap_or_else(|_| "logs/training.log".to_string()),
            root_dir,
            env_map,
        })
    }

    pub fn proxy_config_value(&self, key: &str) -> Option<&str> {
        self.env_map.get(key).map(String::as_str)
    }

    pub fn parse_proxy_config(&self, key: &str) -> Result<Option<Value>> {
        match self.proxy_config_value(key) {
            Some(raw) => {
                let value = serde_json::from_str(raw)
                    .with_context(|| format!("failed to parse env {} as JSON", key))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
}

fn parse_bool_env(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => default,
    }
}

fn load_env_file(path: &Path) -> Result<()> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to load {}", path.display()))?;

    for (idx, line) in raw.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let key = key.trim();
        if key.is_empty() {
            continue;
        }

        let value = value.trim().trim_matches('\r');
        env::set_var(key, value);
        let _ = idx;
    }

    Ok(())
}
