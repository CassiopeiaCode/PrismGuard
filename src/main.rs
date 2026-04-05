#![allow(non_snake_case)]

mod config;
mod format;
mod profile;
mod proxy;
mod response;
mod routes;
mod moderation;
mod scheduler;
mod sample_rpc;
#[cfg(feature = "storage-debug")]
mod storage;
mod streaming;
mod training;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use axum::{Router, Server};
use reqwest::Client;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::config::Settings;
use crate::routes::{router, AppState};
use crate::scheduler::start_scheduler_loop;
use crate::sample_rpc::{cleanup_stale_unix_socket, SampleRpcConfig, SampleRpcTransport};
#[cfg(feature = "storage-debug")]
use crate::sample_rpc::serve_storage_sample_rpc_until_shutdown;
use crate::training::run_training_subprocess_from_args;

enum StartupMode {
    Server,
    TrainProfile,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    match select_startup_mode(&args)? {
        StartupMode::Server => run_server().await,
        StartupMode::TrainProfile => run_training_subprocess_from_args(&args).await,
    }
}

async fn run_server() -> Result<()> {
    let root_dir = std::env::current_dir().context("failed to resolve current dir")?;
    let settings = Settings::load(&root_dir)?;
    init_tracing(&settings.log_level);
    lower_process_priority();
    prepare_sample_rpc(&settings)?;
    let sample_rpc_task = start_sample_rpc_server(&settings)?;
    let scheduler_task = start_scheduler_loop(settings.clone());

    let state = AppState {
        settings: Arc::new(settings.clone()),
        http_client: Client::builder()
            .tcp_keepalive(std::time::Duration::from_secs(30))
            .pool_idle_timeout(std::time::Duration::from_secs(30))
            .pool_max_idle_per_host(200)
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .context("failed to build reqwest client")?,
    };
    let app: Router = router(state);

    let addr: SocketAddr = format!("{}:{}", settings.host, settings.port)
        .parse()
        .with_context(|| format!("invalid listen address {}:{}", settings.host, settings.port))?;
    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {}", addr))?;

    info!(
        service = "Prismguand-Rust",
        root_dir = %root_dir.display(),
        %addr,
        "starting server"
    );

    let server_result = Server::from_tcp(listener.into_std()?)
        .context("failed to convert tokio listener to std listener")?
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server exited with error");

    if let Some(task) = sample_rpc_task {
        task.abort();
    }
    if let Some(task) = scheduler_task {
        task.abort();
    }

    server_result?;

    Ok(())
}

fn select_startup_mode(args: &[String]) -> Result<StartupMode> {
    match args.get(1).map(String::as_str) {
        None => Ok(StartupMode::Server),
        Some("train-profile") => {
            args.get(2)
                .ok_or_else(|| anyhow!("usage: {} train-profile <profile-name>", binary_name(args)))?;
            Ok(StartupMode::TrainProfile)
        }
        Some(other) => Err(anyhow!("unknown subcommand: {other}")),
    }
}

fn binary_name(args: &[String]) -> &str {
    args.first().map(String::as_str).unwrap_or("prismguard-rust")
}

fn prepare_sample_rpc(settings: &Settings) -> Result<()> {
    let rpc = SampleRpcConfig::from_settings(settings)?;
    rpc.prepare_runtime()?;
    if rpc.enabled && rpc.transport == SampleRpcTransport::Unix {
        cleanup_stale_unix_socket(&rpc.unix_socket)?;
    }
    Ok(())
}

fn start_sample_rpc_server(settings: &Settings) -> Result<Option<JoinHandle<()>>> {
    #[cfg(not(feature = "storage-debug"))]
    {
        let rpc = SampleRpcConfig::from_settings(settings)?;
        if rpc.enabled {
            warn!("sample RPC configured but storage-debug feature is disabled; server not started");
        }
        return Ok(None);
    }

    #[cfg(feature = "storage-debug")]
    {
    let rpc = SampleRpcConfig::from_settings(settings)?;
    if !rpc.enabled {
        return Ok(None);
    }

    match rpc.transport {
        SampleRpcTransport::Unix => {
            let socket_path = rpc.unix_socket.clone();
            let handle = tokio::spawn(async move {
                let _ = serve_storage_sample_rpc_until_shutdown(
                    &socket_path,
                    std::future::pending::<()>(),
                )
                .await;
            });
            Ok(Some(handle))
        }
        SampleRpcTransport::Tcp => Err(anyhow::anyhow!(
            "TRAINING_DATA_RPC_TRANSPORT=tcp is not implemented yet"
        )),
    }
    }
}

fn lower_process_priority() {
    let rc = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, 19) };
    if rc == 0 {
        info!("process priority set to nice=19");
    } else {
        let err = std::io::Error::last_os_error();
        warn!(error = %err, "failed to set process priority to nice=19");
    }
}

fn init_tracing(log_level: &str) {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .compact()
        .init();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut signal) = signal(SignalKind::terminate()) {
            signal.recv().await;
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}
