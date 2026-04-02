mod config;
mod profile;
mod proxy;
mod routes;
mod storage;

use std::net::SocketAddr;
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{Router, Server};
use reqwest::Client;
use tokio::net::TcpListener;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::config::Settings;
use crate::routes::{router, AppState};

#[tokio::main]
async fn main() -> Result<()> {
    let root_dir = std::env::current_dir().context("failed to resolve current dir")?;
    let settings = Settings::load(&root_dir)?;
    init_tracing(&settings.log_level);
    lower_process_priority();

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

    Server::from_tcp(listener.into_std()?)
        .context("failed to convert tokio listener to std listener")?
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("server exited with error")?;

    Ok(())
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
