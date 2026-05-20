#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
import argparse
import copy
import secrets
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse
import urllib.request
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"


@dataclass
class DiscoverResult:
    base_url: str
    trace: list[str]


@dataclass
class StreamCapture:
    status_code: int
    first_event_latency_ms: float
    total_latency_ms: float
    done_seen: bool
    lines: list[str]


@dataclass
class ProxyJsonResult:
    status_code: int
    total_latency_ms: float
    body: dict[str, object] | None
    error_body: str | None


@dataclass
class BenchmarkResult:
    mode: str
    status_code: int
    total_latency_ms: float
    proxy_overhead_ms: float
    diagnostics: list[str]
    body: dict[str, object] | None = None
    error_body: str | None = None
    first_event_latency_ms: float | None = None
    done_seen: bool | None = None
    stream_lines: list[str] | None = None
    request_nonce: str | None = None
    request_text_preview: str | None = None


@dataclass
class ProfileDiagnostics:
    profile_path: Path
    config: dict[str, Any] | None
    debug_profile: dict[str, Any] | None
    issues: list[str]


def build_inline_proxy_path(upstream_url: str, profile: str, stream: bool) -> str:
    config = {
        "target_format": "openai_chat",
        "delay_stream_header": True,
        "smart_moderation": {"enabled": True, "profile": profile},
        "metadata": {"benchmark": "inline_moderation", "stream": stream},
    }
    encoded = urllib.parse.quote(json.dumps(config, separators=(",", ":")), safe="")
    return f"/{encoded}${upstream_url}"


def normalize_host(raw_host: str | None) -> str | None:
    if not raw_host:
        return None
    host = raw_host.strip()
    if not host:
        return None
    if host == "0.0.0.0":
        return "127.0.0.1"
    return host


def find_prismguard_pids() -> list[int]:
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "Prismguand-Rust" in cmdline or "prismguard" in cmdline.lower():
            pids.append(int(entry.name))
    return pids


def read_proc_environ(pid: int) -> dict[str, str]:
    path = Path("/proc") / str(pid) / "environ"
    raw = path.read_bytes()
    env_map: dict[str, str] = {}
    for item in raw.split(b"\x00"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        env_map[key.decode("utf-8", errors="ignore")] = value.decode("utf-8", errors="ignore")
    return env_map


def load_dotenv_values(path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not path.exists():
        return env_map
    for line in path.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_map[key.strip()] = value.strip().strip("\r")
    return env_map


def probe_healthz(url: str, timeout_secs: float) -> tuple[bool, float]:
    request = urllib.request.Request(f"{url.rstrip('/')}/healthz", method="GET")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_secs) as response:
            ok = 200 <= response.status < 300
    except Exception:
        ok = False
    elapsed = time.perf_counter() - start
    return ok, elapsed


def discover_base_url(repo_root: Path, timeout_secs: float) -> DiscoverResult:
    trace: list[str] = []
    candidates: list[tuple[str, str]] = []

    for pid in find_prismguard_pids():
        try:
            env_map = read_proc_environ(pid)
        except OSError as exc:
            trace.append(f"proc:{pid} read_error={exc}")
            continue
        host = normalize_host(env_map.get("HOST"))
        port = env_map.get("PORT")
        if host and port:
            candidates.append((f"http://{host}:{port}", f"proc:{pid}"))

    dotenv_values = load_dotenv_values(repo_root / ".env")
    host = normalize_host(dotenv_values.get("HOST"))
    port = dotenv_values.get("PORT")
    if host and port:
        candidates.append((f"http://{host}:{port}", "dotenv"))

    candidates.append(("http://127.0.0.1:8000", "default"))

    for candidate, source in candidates:
        ok, latency_secs = probe_healthz(candidate, timeout_secs)
        trace.append(f"{source} -> {candidate} ok={ok} latency={latency_secs:.3f}s")
        if ok:
            return DiscoverResult(base_url=candidate, trace=trace)

    raise RuntimeError("failed to discover PrismGuard base URL; pass --base-url")


class _MockUpstreamHandler(BaseHTTPRequestHandler):
    server_version = "PrismGuardMockUpstream/1.0"

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            body = {}

        if body.get("stream"):
            self._send_stream()
        else:
            self._send_json()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _send_json(self) -> None:
        delay_ms = getattr(self.server, "json_delay_ms", 0)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
        payload = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "mock-ok"}}],
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()

    def _send_stream(self) -> None:
        first_delay_ms = getattr(self.server, "stream_first_token_delay_ms", 0)
        tail_delay_ms = getattr(self.server, "stream_tail_delay_ms", 0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        if first_delay_ms > 0:
            time.sleep(first_delay_ms / 1000.0)
        chunks = [
            "event: response.created\n"
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_mock\",\"model\":\"gpt-4.1-mini\",\"created_at\":1}}\n\n",
            "event: response.output_text.delta\n"
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"mock-ok\"}\n\n",
        ]
        for chunk in chunks:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        if tail_delay_ms > 0:
            time.sleep(tail_delay_ms / 1000.0)
        final_chunk = (
            "event: response.completed\n"
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_mock\",\"status\":\"completed\"}}\n\n"
            "data: [DONE]\n\n"
        )
        self.wfile.write(final_chunk.encode("utf-8"))
        self.wfile.flush()


class MockUpstreamServer:
    def __init__(self, json_delay_ms: int, stream_first_token_delay_ms: int, stream_tail_delay_ms: int):
        self.json_delay_ms = json_delay_ms
        self.stream_first_token_delay_ms = stream_first_token_delay_ms
        self.stream_tail_delay_ms = stream_tail_delay_ms
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.base_url = ""

    def __enter__(self) -> "MockUpstreamServer":
        server = ThreadingHTTPServer(("127.0.0.1", 0), _MockUpstreamHandler)
        server.json_delay_ms = self.json_delay_ms
        server.stream_first_token_delay_ms = self.stream_first_token_delay_ms
        server.stream_tail_delay_ms = self.stream_tail_delay_ms
        self._server = server
        self.base_url = f"http://127.0.0.1:{server.server_address[1]}"
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def fetch_json(url: str, payload: dict[str, object]) -> dict[str, object]:
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_sse(url: str, payload: dict[str, object], timeout_secs: float) -> StreamCapture:
    start = time.perf_counter()
    first_event_latency_ms = -1.0
    lines: list[str] = []
    done_seen = False
    with requests.post(url, json=payload, timeout=timeout_secs, stream=True) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line)
            if first_event_latency_ms < 0:
                first_event_latency_ms = (time.perf_counter() - start) * 1000.0
            if line == "data: [DONE]":
                done_seen = True
        total_latency_ms = (time.perf_counter() - start) * 1000.0
        return StreamCapture(
            status_code=response.status_code,
            first_event_latency_ms=max(0.0, first_event_latency_ms),
            total_latency_ms=total_latency_ms,
            done_seen=done_seen,
            lines=lines,
        )


def default_request_payload() -> dict[str, object]:
    return {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Please answer with exactly: ok"}],
    }


def load_request_payload(request_file: str | None) -> dict[str, object]:
    if request_file is None:
        return default_request_payload()
    return json.loads(Path(request_file).read_text(encoding="utf-8"))


def generate_request_nonce() -> str:
    return secrets.token_hex(6)


def _extract_user_text_preview(payload: dict[str, object]) -> str | None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content[:160]
    return None


def prepare_request_payload(
    payload: dict[str, object],
    randomize_user_text: bool,
    nonce: str | None = None,
) -> tuple[dict[str, object], str | None]:
    prepared = copy.deepcopy(payload)
    if not randomize_user_text:
        return prepared, None

    effective_nonce = nonce or generate_request_nonce()
    messages = prepared.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = f"{content}\n\nbenchmark_nonce={effective_nonce}"
                break
    return prepared, effective_nonce


def fetch_proxy_json(
    base_url: str,
    proxy_path: str,
    payload: dict[str, object],
    timeout_secs: float,
) -> ProxyJsonResult:
    url = f"{base_url.rstrip('/')}{proxy_path}"
    start = time.perf_counter()
    response = requests.post(url, json=payload, timeout=timeout_secs)
    total_latency_ms = (time.perf_counter() - start) * 1000.0
    try:
        body = response.json()
        error_body = None
    except ValueError:
        body = None
        error_body = response.text
    return ProxyJsonResult(
        status_code=response.status_code,
        total_latency_ms=total_latency_ms,
        body=body,
        error_body=error_body,
    )


def fetch_proxy_sse(
    base_url: str,
    proxy_path: str,
    payload: dict[str, object],
    timeout_secs: float,
) -> StreamCapture:
    url = f"{base_url.rstrip('/')}{proxy_path}"
    return fetch_sse(url, payload, timeout_secs)


def run_non_stream_benchmark(
    base_url: str,
    upstream_url: str,
    profile: str,
    request_file: str | None,
    timeout_secs: float,
    upstream_delay_ms: int,
    randomize_user_text: bool = True,
) -> BenchmarkResult:
    payload = load_request_payload(request_file)
    payload, request_nonce = prepare_request_payload(payload, randomize_user_text=randomize_user_text)
    payload["stream"] = False
    proxy_path = build_inline_proxy_path(upstream_url=upstream_url, profile=profile, stream=False)
    response = fetch_proxy_json(base_url, proxy_path, payload, timeout_secs)
    proxy_overhead_ms = max(0.0, response.total_latency_ms - float(upstream_delay_ms))
    return BenchmarkResult(
        mode="non-stream",
        status_code=response.status_code,
        total_latency_ms=response.total_latency_ms,
        proxy_overhead_ms=proxy_overhead_ms,
        diagnostics=[
            f"base_url={base_url}",
            f"upstream_url={upstream_url}",
            f"profile={profile}",
            f"status={response.status_code}",
        ],
        body=response.body,
        error_body=response.error_body,
        request_nonce=request_nonce,
        request_text_preview=_extract_user_text_preview(payload),
    )


def run_stream_benchmark(
    base_url: str,
    upstream_url: str,
    profile: str,
    request_file: str | None,
    timeout_secs: float,
    first_token_delay_ms: int,
    tail_delay_ms: int,
    randomize_user_text: bool = True,
) -> BenchmarkResult:
    payload = load_request_payload(request_file)
    payload, request_nonce = prepare_request_payload(payload, randomize_user_text=randomize_user_text)
    payload["stream"] = True
    proxy_path = build_inline_proxy_path(upstream_url=upstream_url, profile=profile, stream=True)
    response = fetch_proxy_sse(base_url, proxy_path, payload, timeout_secs)
    upstream_total_ms = float(first_token_delay_ms + tail_delay_ms)
    proxy_overhead_ms = max(0.0, response.total_latency_ms - upstream_total_ms)
    return BenchmarkResult(
        mode="stream",
        status_code=response.status_code,
        total_latency_ms=response.total_latency_ms,
        proxy_overhead_ms=proxy_overhead_ms,
        diagnostics=[
            f"base_url={base_url}",
            f"upstream_url={upstream_url}",
            f"profile={profile}",
            f"status={response.status_code}",
            f"done_seen={response.done_seen}",
        ],
        first_event_latency_ms=response.first_event_latency_ms,
        done_seen=response.done_seen,
        stream_lines=response.lines,
        request_nonce=request_nonce,
        request_text_preview=_extract_user_text_preview(payload),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inline PrismGuard requests with smart moderation and mock upstreams.",
    )
    parser.add_argument("--profile", required=True, help="Moderation profile name under configs/mod_profiles.")
    parser.add_argument("--base-url", help="PrismGuard base URL. Auto-discovered when omitted.")
    parser.add_argument("--request-file", help="Optional JSON file used as the proxy request body.")
    parser.add_argument(
        "--mode",
        choices=["all", "non-stream", "stream"],
        default="all",
        help="Which benchmark mode to run.",
    )
    parser.add_argument("--timeout-secs", type=float, default=15.0, help="HTTP timeout in seconds.")
    parser.add_argument("--upstream-delay-ms", type=int, default=40, help="Mock JSON upstream delay in milliseconds.")
    parser.add_argument(
        "--stream-first-token-delay-ms",
        type=int,
        default=40,
        help="Mock SSE first-event delay in milliseconds.",
    )
    parser.add_argument(
        "--stream-tail-delay-ms",
        type=int,
        default=20,
        help="Mock SSE tail delay in milliseconds.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print extra diagnostics.")
    parser.add_argument(
        "--disable-randomize-user-text",
        action="store_true",
        help="Keep user messages unchanged instead of appending a random benchmark nonce.",
    )
    return parser.parse_args()


def fetch_debug_profile(base_url: str, profile: str, timeout_secs: float) -> dict[str, Any] | None:
    url = f"{base_url.rstrip('/')}/debug/profile/{urllib.parse.quote(profile, safe='')}"
    try:
        response = requests.get(url, timeout=timeout_secs)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def load_profile_config(profile: str) -> tuple[Path, dict[str, Any] | None]:
    profile_path = CONFIGS_DIR / profile / "profile.json"
    if not profile_path.exists():
        return profile_path, None
    return profile_path, json.loads(profile_path.read_text(encoding="utf-8"))


def collect_profile_diagnostics(base_url: str, profile: str, timeout_secs: float) -> ProfileDiagnostics:
    profile_path, config = load_profile_config(profile)
    debug_profile = fetch_debug_profile(base_url, profile, timeout_secs)
    issues: list[str] = []
    if config is None:
        issues.append(f"profile config missing: {profile_path}")
        return ProfileDiagnostics(profile_path=profile_path, config=None, debug_profile=debug_profile, issues=issues)

    local_model_type = str(config.get("local_model_type") or "bow")
    profile_dir = profile_path.parent
    marker_paths = {
        "hashlinear": [profile_dir / "hashlinear_model.pkl", profile_dir / "hashlinear_runtime.json", profile_dir / "hashlinear_runtime.coef.f32"],
        "bow": [profile_dir / "bow_model.pkl", profile_dir / "bow_vectorizer.pkl", profile_dir / "bow_runtime.json", profile_dir / "bow_runtime.coef.f32"],
        "fasttext": [profile_dir / "fasttext_model.bin", profile_dir / "fasttext_runtime.json"],
    }
    for artifact in marker_paths.get(local_model_type, []):
        if not artifact.exists():
            issues.append(f"missing local model artifact: {artifact}")

    ai_cfg = config.get("ai") or {}
    if not ai_cfg.get("base_url"):
        issues.append("profile.ai.base_url is empty")
    if not ai_cfg.get("model"):
        issues.append("profile.ai.model is empty")
    if not ai_cfg.get("api_key_env"):
        issues.append("profile.ai.api_key_env is empty")

    if debug_profile is not None and not debug_profile.get("local_model_exists", False):
        issues.append("debug/profile reports local_model_exists=false, requests may fall back to LLM")

    return ProfileDiagnostics(
        profile_path=profile_path,
        config=config,
        debug_profile=debug_profile,
        issues=issues,
    )


def summarize_body(body: dict[str, Any] | None) -> str:
    if body is None:
        return "-"
    try:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))[:400]
    except Exception:
        return str(body)[:400]


def print_profile_diagnostics(diag: ProfileDiagnostics, verbose: bool) -> None:
    print("== Profile Diagnostics ==")
    print(f"profile_path: {diag.profile_path}")
    if diag.config is None:
        print("profile_config: missing")
    else:
        ai_cfg = diag.config.get("ai") or {}
        prob_cfg = diag.config.get("probability") or {}
        print(f"local_model_type: {diag.config.get('local_model_type')}")
        print(f"ai.base_url: {ai_cfg.get('base_url')}")
        print(f"ai.model: {ai_cfg.get('model')}")
        print(f"ai.api_key_env: {ai_cfg.get('api_key_env')}")
        print(f"probability.ai_review_rate: {prob_cfg.get('ai_review_rate')}")
        print(f"probability.low_risk_threshold: {prob_cfg.get('low_risk_threshold')}")
        print(f"probability.high_risk_threshold: {prob_cfg.get('high_risk_threshold')}")
    if diag.debug_profile is not None:
        print(f"debug.local_model_exists: {diag.debug_profile.get('local_model_exists')}")
        print(f"debug.live_sample_count: {diag.debug_profile.get('live_sample_count')}")
        decision = diag.debug_profile.get("training_decision") or {}
        if decision:
            print(f"debug.training_decision: {decision}")
    else:
        print("debug_profile: unavailable")
    if diag.issues:
        print("issues:")
        for item in diag.issues:
            print(f"- {item}")
    elif verbose:
        print("issues: none")


def explain_result(result: BenchmarkResult) -> list[str]:
    hints: list[str] = []
    if result.status_code >= 500:
        hints.append("proxy returned 5xx; inspect error body and moderation logs for upstream or parsing failures")
    elif result.status_code >= 400:
        hints.append("proxy returned 4xx; inspect inline config, profile name, and moderation_details in the body")

    if result.mode == "stream":
        if result.done_seen is False:
            hints.append("stream ended without [DONE]; check upstream SSE shape or proxy stream transformation")
        if result.first_event_latency_ms is not None and result.first_event_latency_ms > 1000:
            hints.append("time-to-first-event is high; bottleneck is likely before streaming starts, often moderation or upstream connect latency")
    if result.proxy_overhead_ms > 1000:
        hints.append("proxy overhead is high relative to the mock upstream delay; inspect moderation chain, LLM timeout, and local model availability")
    if not hints:
        hints.append("no obvious anomaly from latency shape; compare multiple runs and inspect moderation history if results still look suspicious")
    return hints


def print_benchmark_result(result: BenchmarkResult) -> None:
    print(f"== {result.mode} ==")
    print(f"status_code: {result.status_code}")
    print(f"total_latency_ms: {result.total_latency_ms:.2f}")
    print(f"proxy_overhead_ms: {result.proxy_overhead_ms:.2f}")
    if result.first_event_latency_ms is not None:
        print(f"first_event_latency_ms: {result.first_event_latency_ms:.2f}")
    if result.done_seen is not None:
        print(f"done_seen: {result.done_seen}")
    if result.body is not None:
        print(f"body: {summarize_body(result.body)}")
    if result.error_body:
        print(f"error_body: {result.error_body[:400]}")
    if result.request_nonce is not None:
        print(f"request_nonce: {result.request_nonce}")
    if result.request_text_preview is not None:
        print(f"request_text_preview: {result.request_text_preview}")
    if result.stream_lines:
        preview = " | ".join(result.stream_lines[:6])
        print(f"stream_preview: {preview}")
    print("diagnostics:")
    for item in result.diagnostics:
        print(f"- {item}")
    print("troubleshooting:")
    for item in explain_result(result):
        print(f"- {item}")


def resolve_base_url(cli_base_url: str | None, timeout_secs: float) -> DiscoverResult:
    if cli_base_url:
        ok, latency = probe_healthz(cli_base_url, timeout_secs)
        if not ok:
            raise RuntimeError(f"base URL {cli_base_url} failed /healthz probe")
        return DiscoverResult(
            base_url=cli_base_url.rstrip("/"),
            trace=[f"cli -> {cli_base_url.rstrip('/')} ok=True latency={latency:.3f}s"],
        )
    return discover_base_url(REPO_ROOT, timeout_secs)


def main() -> int:
    args = parse_args()
    try:
        discovered = resolve_base_url(args.base_url, args.timeout_secs)
    except Exception as exc:
        print(f"Failed to discover PrismGuard base URL: {exc}")
        return 2

    print("== Target Discovery ==")
    print(f"base_url: {discovered.base_url}")
    for item in discovered.trace:
        print(f"- {item}")

    profile_diag = collect_profile_diagnostics(discovered.base_url, args.profile, args.timeout_secs)
    print_profile_diagnostics(profile_diag, args.verbose)

    exit_code = 0
    with MockUpstreamServer(
        json_delay_ms=args.upstream_delay_ms,
        stream_first_token_delay_ms=args.stream_first_token_delay_ms,
        stream_tail_delay_ms=args.stream_tail_delay_ms,
    ) as server:
        upstream_url = f"{server.base_url}/v1/chat/completions"
        print("== Mock Upstream ==")
        print(f"upstream_url: {upstream_url}")
        print(f"json_delay_ms: {args.upstream_delay_ms}")
        print(f"stream_first_token_delay_ms: {args.stream_first_token_delay_ms}")
        print(f"stream_tail_delay_ms: {args.stream_tail_delay_ms}")

        if args.mode in {"all", "non-stream"}:
            try:
                non_stream = run_non_stream_benchmark(
                    base_url=discovered.base_url,
                    upstream_url=upstream_url,
                    profile=args.profile,
                    request_file=args.request_file,
                    timeout_secs=args.timeout_secs,
                    upstream_delay_ms=args.upstream_delay_ms,
                    randomize_user_text=not args.disable_randomize_user_text,
                )
                print_benchmark_result(non_stream)
                if non_stream.status_code >= 400:
                    exit_code = 1
            except Exception as exc:
                exit_code = 1
                print("== non-stream ==")
                print(f"benchmark_error: {exc}")

        if args.mode in {"all", "stream"}:
            try:
                stream = run_stream_benchmark(
                    base_url=discovered.base_url,
                    upstream_url=upstream_url,
                    profile=args.profile,
                    request_file=args.request_file,
                    timeout_secs=args.timeout_secs,
                    first_token_delay_ms=args.stream_first_token_delay_ms,
                    tail_delay_ms=args.stream_tail_delay_ms,
                    randomize_user_text=not args.disable_randomize_user_text,
                )
                print_benchmark_result(stream)
                if stream.status_code >= 400 or stream.done_seen is False:
                    exit_code = 1
            except Exception as exc:
                exit_code = 1
                print("== stream ==")
                print(f"benchmark_error: {exc}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
