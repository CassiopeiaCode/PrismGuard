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
    pub training_data_rpc_enabled: bool,
    pub training_data_rpc_transport: String,
    pub training_data_rpc_unix_socket: String,
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
            training_data_rpc_enabled: parse_bool_env("TRAINING_DATA_RPC_ENABLED", true),
            training_data_rpc_transport: env::var("TRAINING_DATA_RPC_TRANSPORT")
                .unwrap_or_else(|_| "unix".to_string()),
            training_data_rpc_unix_socket: env::var("TRAINING_DATA_RPC_UNIX_SOCKET")
                .unwrap_or_else(|_| root_dir.join("run/sample-store.sock").display().to_string()),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn settings_default_training_rpc_uses_unix_socket() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let root_dir = PathBuf::from("/tmp/prismguard-config-defaults");
        std::env::remove_var("TRAINING_DATA_RPC_ENABLED");
        std::env::remove_var("TRAINING_DATA_RPC_TRANSPORT");
        std::env::remove_var("TRAINING_DATA_RPC_UNIX_SOCKET");

        let settings = Settings::load(&root_dir).expect("load settings");

        assert!(settings.training_data_rpc_enabled);
        assert_eq!(settings.training_data_rpc_transport, "unix");
        assert_eq!(
            settings.training_data_rpc_unix_socket,
            "/tmp/prismguard-config-defaults/run/sample-store.sock"
        );
    }

    #[test]
    fn settings_can_override_training_rpc_env() {
        let _guard = env_test_lock().lock().expect("env test lock");
        let root_dir = PathBuf::from("/tmp/prismguard-config-overrides");
        std::env::set_var("TRAINING_DATA_RPC_ENABLED", "0");
        std::env::set_var("TRAINING_DATA_RPC_TRANSPORT", "tcp");
        std::env::set_var("TRAINING_DATA_RPC_UNIX_SOCKET", "/tmp/custom.sock");

        let settings = Settings::load(&root_dir).expect("load settings");

        assert!(!settings.training_data_rpc_enabled);
        assert_eq!(settings.training_data_rpc_transport, "tcp");
        assert_eq!(settings.training_data_rpc_unix_socket, "/tmp/custom.sock");

        std::env::remove_var("TRAINING_DATA_RPC_ENABLED");
        std::env::remove_var("TRAINING_DATA_RPC_TRANSPORT");
        std::env::remove_var("TRAINING_DATA_RPC_UNIX_SOCKET");
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
