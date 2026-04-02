use std::io::{self, Read};

use anyhow::Context;
use axum::body::{boxed, BoxBody, Body, StreamBody};
use axum::extract::{Path, State};
use axum::http::header::{ACCEPT_ENCODING, CONTENT_ENCODING, CONTENT_LENGTH, HOST};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, Request, StatusCode};
use axum::response::Response;
use futures_util::TryStreamExt;
use hyper::body::to_bytes;
use serde_json::Value;

use crate::routes::{parse_url_config, ApiError, AppState};

const HEADER_DENYLIST: &[&str] = &[
    "content-length",
    "transfer-encoding",
    "content-encoding",
    "set-cookie",
    "strict-transport-security",
    "content-security-policy",
    "content-security-policy-report-only",
    "x-frame-options",
    "x-content-type-options",
    "x-xss-protection",
    "permissions-policy",
    "referrer-policy",
];

pub async fn proxy_entry(
    State(state): State<AppState>,
    Path(cfg_and_upstream): Path<String>,
    request: Request<Body>,
) -> Result<Response, ApiError> {
    let parsed = parse_url_config(&state.settings, &cfg_and_upstream)?;
    let upstream = reqwest::Url::parse(&parsed.upstream).map_err(|e| {
        ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("invalid upstream url: {e}"),
        )
        .with_code("CONFIG_PARSE_ERROR")
        .with_error_type("config_error")
    })?;

    let upstream_base = format!(
        "{}://{}",
        upstream.scheme(),
        upstream
            .host_str()
            .ok_or_else(|| {
                ApiError::new(StatusCode::BAD_REQUEST, "upstream host is missing")
                    .with_code("CONFIG_PARSE_ERROR")
                    .with_error_type("config_error")
            })?
    );
    let final_url = upstream.to_string();
    let path = if upstream.path().is_empty() {
        "/".to_string()
    } else {
        upstream.path().to_string()
    };

    let (parts, body) = request.into_parts();
    let raw_body = to_bytes(body)
        .await
        .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, format!("failed to read body: {e}")))?;
    let request_json = parse_request_json(&parts.method, &parts.headers, &raw_body)?;
    let is_stream = request_json
        .as_ref()
        .and_then(|value| value.get("stream"))
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let mut upstream_request = state
        .http_client
        .request(parts.method.clone(), final_url)
        .headers(filtered_request_headers(&parts.headers));

    if !raw_body.is_empty() {
        upstream_request = upstream_request.body(raw_body.clone());
    }

    let upstream_response = upstream_request.send().await.map_err(|e| {
        ApiError::new(
            StatusCode::BAD_GATEWAY,
            format!("upstream request failed: {e}"),
        )
        .with_code("UPSTREAM_ERROR")
        .with_error_type("upstream_error")
    })?;

    tracing::info!(
        method = %parts.method,
        upstream_base,
        upstream_path = %path,
        stream = is_stream,
        status = upstream_response.status().as_u16(),
        config_source = %parsed.source,
        "proxied request"
    );

    build_proxy_response(upstream_response).await
}

fn parse_request_json(
    method: &Method,
    headers: &HeaderMap,
    raw_body: &[u8],
) -> Result<Option<Value>, ApiError> {
    if raw_body.is_empty() {
        return Ok(None);
    }

    if !matches!(*method, Method::POST | Method::PUT | Method::PATCH | Method::DELETE) {
        return Ok(None);
    }

    let decoded = decompress_body(headers, raw_body)?;
    match serde_json::from_slice::<Value>(&decoded) {
        Ok(value) => Ok(Some(value)),
        Err(_) => Ok(None),
    }
}

fn decompress_body(headers: &HeaderMap, raw_body: &[u8]) -> Result<Vec<u8>, ApiError> {
    let encoding = headers
        .get(CONTENT_ENCODING)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();

    match encoding.as_str() {
        "" | "identity" => Ok(raw_body.to_vec()),
        "gzip" => {
            let mut decoder = flate2::read::GzDecoder::new(raw_body);
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut out).map_err(|e| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("failed to decode gzip request body: {e}"),
                )
            })?;
            Ok(out)
        }
        "deflate" => {
            let mut decoder = flate2::read::ZlibDecoder::new(raw_body);
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut out).map_err(|e| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("failed to decode deflate request body: {e}"),
                )
            })?;
            Ok(out)
        }
        "br" => {
            let mut decoder = brotli::Decompressor::new(raw_body, 4096);
            let mut out = Vec::new();
            decoder.read_to_end(&mut out).map_err(|e| {
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    format!("failed to decode brotli request body: {e}"),
                )
            })?;
            Ok(out)
        }
        other => Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            format!("unsupported content-encoding: {other}"),
        )),
    }
}

fn filtered_request_headers(headers: &HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers.iter() {
        if name == HOST || name == CONTENT_LENGTH || name == ACCEPT_ENCODING {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    out.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    out
}

async fn build_proxy_response(upstream_response: reqwest::Response) -> Result<Response, ApiError> {
    let status = upstream_response.status();
    let headers = filtered_response_headers(upstream_response.headers());

    if is_stream_response(upstream_response.headers()) {
        let stream = upstream_response.bytes_stream().map_err(map_stream_error);
        let body = boxed(StreamBody::new(stream));
        build_response(status, &headers, body)
    } else {
        let body = upstream_response
            .bytes()
            .await
            .context("failed to read upstream body")
            .map_err(ApiError::from)?;
        build_response(status, &headers, boxed(Body::from(body)))
    }
}

fn build_response(
    status: reqwest::StatusCode,
    headers: &HeaderMap,
    body: BoxBody,
) -> Result<Response, ApiError> {
    let mut response = Response::builder().status(status);
    if let Some(target_headers) = response.headers_mut() {
        for (name, value) in headers.iter() {
            target_headers.append(name.clone(), value.clone());
        }
    }
    response
        .body(body)
        .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, format!("failed to build response: {e}")))
}

fn filtered_response_headers(headers: &reqwest::header::HeaderMap) -> HeaderMap {
    let mut out = HeaderMap::new();
    for (name, value) in headers.iter() {
        if HEADER_DENYLIST.contains(&name.as_str()) {
            continue;
        }
        if let Ok(header_name) = HeaderName::from_bytes(name.as_str().as_bytes()) {
            out.append(header_name, value.clone());
        }
    }
    out
}

fn is_stream_response(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.starts_with("text/event-stream"))
        .unwrap_or(false)
}

fn map_stream_error(error: reqwest::Error) -> io::Error {
    io::Error::new(io::ErrorKind::Other, error)
}
