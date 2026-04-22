#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"


def normalize_spaces(s: str) -> str:
    out = []
    prev_space = False
    for ch in s:
        is_space = ch.isspace()
        if is_space:
            if not prev_space:
                out.append(" ")
        else:
            out.append(ch)
        prev_space = is_space
    return "".join(out)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def decode_rocksdict_string(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    b = bytes(raw)
    if b[:1] == b"\x02":
        b = b[1:]
    return b.decode("utf-8", errors="strict")


def read_latest_sample_text(rocks_path: Path) -> Tuple[str, int]:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))
    next_id_raw = db.get(b"\x02meta:next_id") or db.get("meta:next_id")
    if next_id_raw is None:
        raise RuntimeError("missing meta:next_id in history.rocks")
    next_id = int(decode_rocksdict_string(next_id_raw))

    # IDs may not be perfectly contiguous; walk backwards until a sample exists.
    for candidate in range(next_id - 1, max(next_id - 2000, 0), -1):
        key = f"sample:{candidate:020}"
        raw = db.get(b"\x02" + key.encode("utf-8")) or db.get(key)
        if raw is None:
            continue
        obj = json.loads(decode_rocksdict_string(raw))
        text = str(obj.get("text") or "")
        label = int(obj.get("label") or 0)
        return text, label
    raise RuntimeError("failed to locate a recent sample:* record near meta:next_id")


def load_hashlinear_runtime(profile_dir: Path):
    meta_path = profile_dir / "hashlinear_runtime.json"
    coef_path = profile_dir / "hashlinear_runtime.coef.f32"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("runtime_version") != 1:
        raise RuntimeError(f"unsupported runtime_version={meta.get('runtime_version')}")
    cfg = meta.get("cfg") or {}
    n_features = int(meta.get("n_features") or 0)
    if n_features <= 0:
        raise RuntimeError("invalid n_features")
    import numpy as np  # type: ignore

    coef = np.frombuffer(coef_path.read_bytes(), dtype="<f4").astype("float32")
    if coef.shape[0] != n_features:
        raise RuntimeError(f"coef length mismatch: {coef.shape[0]} != {n_features}")
    return meta, cfg, coef


def bench_python_fast(text: str, meta: dict, cfg: dict, coef, iters: int, batch: int) -> float:
    import numpy as np  # type: ignore
    from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

    intercept = float(meta.get("intercept") or 0.0)
    n_features = int(meta.get("n_features") or 0)
    ngram = cfg.get("ngram_range") or [2, 4]
    ngram_range = (int(ngram[0]), int(ngram[1]))
    alternate_sign = bool(cfg.get("alternate_sign") or False)
    norm = "l2" if cfg.get("norm") == "l2" else None
    lowercase = bool(cfg.get("lowercase") if cfg.get("lowercase") is not None else True)

    def preprocess(s: str) -> str:
        s = s.replace("\r", " ").replace("\n", " ")
        if lowercase:
            s = s.lower()
        return normalize_spaces(s)

    vectorizer = HashingVectorizer(
        n_features=n_features,
        analyzer="char",
        ngram_range=ngram_range,
        alternate_sign=alternate_sign,
        norm=norm,
        lowercase=False,
        preprocessor=preprocess,
    )

    texts = [text] * max(1, batch)
    runs = max(1, iters // len(texts))
    t0 = time.perf_counter()
    for _ in range(runs):
        X = vectorizer.transform(texts)
        scores = X.dot(coef).astype(np.float64) + intercept
        _ = 1.0 / (1.0 + np.exp(-scores))
    elapsed = time.perf_counter() - t0
    total = runs * len(texts)
    return (elapsed / total) * 1000.0  # ms per text


def bench_python_slow(text: str, meta: dict, cfg: dict, coef, iters: int) -> float:
    # Slow pure-Python fallback identical in spirit to eval_profile.py's slow path:
    # char ngrams + murmurhash + accumulate
    intercept = float(meta.get("intercept") or 0.0)
    n_features = int(meta.get("n_features") or 0)
    ngram = cfg.get("ngram_range") or [2, 4]
    min_n, max_n = int(ngram[0]), int(ngram[1])
    alternate_sign = bool(cfg.get("alternate_sign") or False)
    norm = cfg.get("norm")
    lowercase = bool(cfg.get("lowercase") if cfg.get("lowercase") is not None else True)

    def murmurhash3_x86_32(data: bytes, seed: int = 0) -> int:
        c1 = 0xCC9E2D51
        c2 = 0x1B873593
        length = len(data)
        h1 = seed & 0xFFFFFFFF
        rounded_end = length & ~0x3
        for i in range(0, rounded_end, 4):
            k1 = int.from_bytes(data[i : i + 4], "little")
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
            k1 = (k1 * c2) & 0xFFFFFFFF
            h1 ^= k1
            h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
            h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF
        tail = data[rounded_end:]
        k1 = 0
        if len(tail) == 3:
            k1 ^= tail[2] << 16
            k1 ^= tail[1] << 8
            k1 ^= tail[0]
        elif len(tail) == 2:
            k1 ^= tail[1] << 8
            k1 ^= tail[0]
        elif len(tail) == 1:
            k1 ^= tail[0]
        if len(tail) > 0:
            k1 = (k1 * c1) & 0xFFFFFFFF
            k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
            k1 = (k1 * c2) & 0xFFFFFFFF
            h1 ^= k1
        h1 ^= length
        h1 ^= (h1 >> 16)
        h1 = (h1 * 0x85EBCA6B) & 0xFFFFFFFF
        h1 ^= (h1 >> 13)
        h1 = (h1 * 0xC2B2AE35) & 0xFFFFFFFF
        h1 ^= (h1 >> 16)
        if h1 & 0x80000000:
            return -((~h1 + 1) & 0xFFFFFFFF)
        return h1

    def iter_char_ngrams(s: str):
        chars = list(s)
        L = len(chars)
        for n in range(min_n, min(max_n, L) + 1):
            for i in range(0, L - n + 1):
                yield "".join(chars[i : i + n])

    def predict_one(s: str) -> float:
        s = s.replace("\r", " ").replace("\n", " ")
        if lowercase:
            s = s.lower()
        s = normalize_spaces(s)
        counts: dict[int, float] = {}
        for gram in iter_char_ngrams(s):
            h = murmurhash3_x86_32(gram.encode("utf-8"), 0)
            idx = abs(h) % n_features
            sign = -1.0 if (alternate_sign and h < 0) else 1.0
            counts[idx] = counts.get(idx, 0.0) + sign
        if norm == "l2" and counts:
            denom = math.sqrt(sum(v * v for v in counts.values()))
            if denom > 0:
                for k in list(counts.keys()):
                    counts[k] /= denom
        score = intercept
        for idx, val in counts.items():
            score += float(coef[idx]) * val
        return sigmoid(score)

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = predict_one(text)
    elapsed = time.perf_counter() - t0
    return (elapsed / iters) * 1000.0


def bench_rust_http_metrics(base_url: str, profile: str, iters: int, timeout_secs: float) -> Tuple[float, float, float]:
    """
    Approximate Rust per-text inference latency by calling the existing metrics endpoint with sample_size=1.
    This includes HTTP + DB open/read + JSON parsing overhead on the server, so it's an upper bound.
    """
    import requests  # type: ignore

    times: list[float] = []
    url = f"{base_url.rstrip('/')}/debug/profile/{profile}/metrics?sample_size=1&sampling=latest_full&threshold=0.5"
    for _ in range(iters):
        t0 = time.perf_counter()
        r = requests.get(url, timeout=timeout_secs)
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:2000]}")
        times.append(dt * 1000.0)
    return statistics.mean(times), statistics.median(times), max(times)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark single-text moderation latency (Python offline vs Rust HTTP).")
    ap.add_argument("--profile", required=True)
    ap.add_argument("--text", default=None, help="text to benchmark; default uses latest sample from history.rocks")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--py-fast-batch", type=int, default=128, help="batch size for python fast path")
    ap.add_argument("--rust-base-url", default="http://127.0.0.1:45271")
    ap.add_argument("--rust-iters", type=int, default=30)
    ap.add_argument("--rust-timeout-secs", type=float, default=60.0)
    args = ap.parse_args()

    profile = args.profile.strip()
    profile_dir = CONFIGS_DIR / profile
    if not profile_dir.exists():
        raise SystemExit(f"profile dir not found: {profile_dir}")

    cfg_json = json.loads((profile_dir / "profile.json").read_text(encoding="utf-8"))
    local_model_type = str(cfg_json.get("local_model_type") or "").strip().lower()
    if local_model_type != "hashlinear":
        raise SystemExit("bench_single currently supports hashlinear only (same as eval_profile offline fast path).")

    rocks_path = profile_dir / "history.rocks"
    if not rocks_path.exists():
        raise SystemExit(f"history.rocks not found: {rocks_path}")

    if args.text is None:
        text, label = read_latest_sample_text(rocks_path)
        text_src = f"latest sample (label={label})"
    else:
        text = args.text
        text_src = "cli --text"

    meta, cfg, coef = load_hashlinear_runtime(profile_dir)
    py_fast_ms = bench_python_fast(text, meta, cfg, coef, iters=args.iters, batch=args.py_fast_batch)
    py_slow_ms = bench_python_slow(text, meta, cfg, coef, iters=min(50, max(1, args.iters // 10)))
    rust_mean, rust_median, rust_max = bench_rust_http_metrics(
        base_url=args.rust_base_url,
        profile=profile,
        iters=args.rust_iters,
        timeout_secs=args.rust_timeout_secs,
    )

    print("Single-text latency benchmark")
    print(f"- profile: {profile} (local_model_type={local_model_type})")
    print(f"- text source: {text_src}")
    print(f"- text chars: {len(text)}")
    print()
    print("Python offline (hashlinear runtime)")
    print(f"- fast path (sklearn HashingVectorizer): {py_fast_ms:.4f} ms/text  (batch={args.py_fast_batch}, iters={args.iters})")
    print(f"- slow path (pure python ngrams/hash):   {py_slow_ms:.4f} ms/text  (iters~{min(50, max(1, args.iters // 10))})")
    print()
    print("Rust service (HTTP /debug/profile/<profile>/metrics?sample_size=1)")
    print(f"- mean:   {rust_mean:.2f} ms/request")
    print(f"- median: {rust_median:.2f} ms/request")
    print(f"- max:    {rust_max:.2f} ms/request  (iters={args.rust_iters})")
    print()
    print("Notes:")
    print("- Rust numbers include HTTP + server-side DB open/read + JSON encode/decode, so they are an upper bound for pure model inference.")
    print("- Python numbers are pure local inference cost (no HTTP, no DB scan).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

