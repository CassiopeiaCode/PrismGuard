#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"


@dataclass(frozen=True)
class FileInfo:
    path: Path
    exists: bool
    size: Optional[int]
    mtime: Optional[float]


def _human_bytes(n: Optional[int]) -> str:
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}" if u != "B" else f"{int(v)}B"
        v /= 1024.0
    return f"{v:.1f}PB"


def _fmt_time(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def file_info(path: Path) -> FileInfo:
    try:
        st = path.stat()
        return FileInfo(path=path, exists=True, size=st.st_size, mtime=st.st_mtime)
    except FileNotFoundError:
        return FileInfo(path=path, exists=False, size=None, mtime=None)


def print_file_report(title: str, paths: Iterable[Path]) -> None:
    print(f"\n== {title} ==")
    for p in paths:
        info = file_info(p)
        status = "OK" if info.exists else "MISSING"
        rel = str(p.relative_to(REPO_ROOT)) if p.is_absolute() else str(p)
        print(f"- {status:7} {rel}  size={_human_bytes(info.size):>8}  mtime={_fmt_time(info.mtime)}")


def load_profile_json(profile: str) -> Dict[str, Any]:
    profile_dir = CONFIGS_DIR / profile
    profile_path = profile_dir / "profile.json"
    if not profile_path.exists():
        raise SystemExit(f"profile config not found: {profile_path}")
    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"failed to parse {profile_path}: {e}")


def model_artifacts_for_type(profile_dir: Path, local_model_type: str) -> Tuple[List[Path], List[Path]]:
    local_model_type = local_model_type.strip().lower()
    if local_model_type == "hashlinear":
        runtime = [profile_dir / "hashlinear_runtime.json", profile_dir / "hashlinear_runtime.coef.f32"]
        marker = [profile_dir / "hashlinear_model.pkl"]
        return runtime, marker
    if local_model_type == "bow":
        runtime = [profile_dir / "bow_runtime.json", profile_dir / "bow_runtime.coef.f32"]
        marker = [profile_dir / "bow_model.pkl", profile_dir / "bow_vectorizer.pkl"]
        return runtime, marker
    if local_model_type == "fasttext":
        runtime = [profile_dir / "fasttext_runtime.json"]
        marker = [profile_dir / "fasttext_model.bin"]
        return runtime, marker
    raise SystemExit(f"unsupported local_model_type: {local_model_type}")


def parse_thresholds(raw: str) -> List[float]:
    if not raw.strip():
        return []
    out: List[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    seen = set()
    deduped: List[float] = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        deduped.append(v)
    return deduped


def sigmoid(x: float) -> float:
    # stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def murmurhash3_x86_32(data: bytes, seed: int = 0) -> int:
    # Ported from Rust implementation in src/moderation/hashlinear.rs
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

    # Convert to signed i32
    if h1 & 0x80000000:
        return -((~h1 + 1) & 0xFFFFFFFF)
    return h1


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


def iter_char_ngrams(text: str, min_n: int, max_n: int) -> Iterator[str]:
    chars = list(text)
    text_len = len(chars)
    current_min = min_n
    if current_min == 1:
        for ch in chars:
            yield ch
        current_min += 1
    for n in range(current_min, min(max_n, text_len) + 1):
        for start in range(0, text_len - n + 1):
            yield "".join(chars[start : start + n])


def _read_f32_le(path: Path) -> "List[float]":
    import numpy as np  # type: ignore

    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise SystemExit(f"invalid coef file length (not multiple of 4): {path} bytes={len(data)}")
    arr = np.frombuffer(data, dtype="<f4")
    return arr.astype("float32").tolist()


@dataclass
class HashlinearRuntime:
    intercept: float
    n_features: int
    analyzer: str
    ngram_range: Tuple[int, int]
    alternate_sign: bool
    norm: Optional[str]
    lowercase: bool
    use_jieba: bool
    coef: List[float]

    @staticmethod
    def load(meta_path: Path, coef_path: Path) -> "HashlinearRuntime":
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("runtime_version") != 1:
            raise SystemExit(f"unsupported hashlinear runtime_version: {meta.get('runtime_version')}")
        if meta.get("classes") != [0, 1]:
            raise SystemExit(f"unsupported hashlinear classes: {meta.get('classes')}")
        cfg = meta.get("cfg") or {}
        n_features = int(meta.get("n_features") or 0)
        if n_features <= 0:
            raise SystemExit("hashlinear n_features must be > 0")
        if int(cfg.get("n_features") or 0) != n_features:
            raise SystemExit(f"hashlinear n_features mismatch: cfg={cfg.get('n_features')} meta={n_features}")
        ngram = cfg.get("ngram_range") or [2, 4]
        if not (isinstance(ngram, list) and len(ngram) == 2 and int(ngram[0]) > 0 and int(ngram[0]) <= int(ngram[1])):
            raise SystemExit(f"invalid hashlinear ngram_range: {ngram}")
        coef = _read_f32_le(coef_path)
        if len(coef) != n_features:
            raise SystemExit(f"hashlinear coef length mismatch: {len(coef)} != {n_features}")
        return HashlinearRuntime(
            intercept=float(meta.get("intercept") or 0.0),
            n_features=n_features,
            analyzer=str(cfg.get("analyzer") or "char"),
            ngram_range=(int(ngram[0]), int(ngram[1])),
            alternate_sign=bool(cfg.get("alternate_sign") or False),
            norm=cfg.get("norm"),
            lowercase=bool(cfg.get("lowercase") if cfg.get("lowercase") is not None else True),
            use_jieba=bool(cfg.get("use_jieba") or False),
            coef=coef,
        )

    def _prepare_text(self, text: str) -> str:
        clean = text.replace("\r", " ").replace("\n", " ")
        lowered = clean.lower() if self.lowercase else clean
        if self.analyzer == "char":
            return normalize_spaces(lowered)
        if self.analyzer == "word":
            if not self.use_jieba:
                return lowered
            import jieba  # type: ignore
            return " ".join(jieba.cut(lowered, cut_all=False))
        raise SystemExit(f"unsupported hashlinear analyzer: {self.analyzer}")

    def predict_proba(self, text: str) -> float:
        prepared = self._prepare_text(text)
        min_n, max_n = self.ngram_range
        counts: Dict[int, float] = {}
        if self.analyzer == "char":
            grams = iter_char_ngrams(prepared, min_n, max_n)
        else:
            # word analyzer: use whitespace tokens, build n-grams similar to Rust
            tokens = prepared.split()
            grams_list: List[str] = []
            cur_min = min_n
            if cur_min == 1:
                grams_list.extend(tokens)
                cur_min += 1
            for n in range(cur_min, min(max_n, len(tokens)) + 1):
                for start in range(0, len(tokens) - n + 1):
                    grams_list.append(" ".join(tokens[start : start + n]))
            grams = iter(grams_list)

        for gram in grams:
            h = murmurhash3_x86_32(gram.encode("utf-8"), 0)
            idx = (abs(h) % self.n_features)
            sign = -1.0 if (self.alternate_sign and h < 0) else 1.0
            counts[idx] = counts.get(idx, 0.0) + sign

        if self.norm == "l2" and counts:
            norm = math.sqrt(sum(v * v for v in counts.values()))
            if norm > 0:
                for k in list(counts.keys()):
                    counts[k] = counts[k] / norm

        score = self.intercept
        for idx, value in counts.items():
            score += float(self.coef[idx]) * value
        return sigmoid(score)


def decode_rocksdict_string(raw: bytes) -> str:
    if raw[:1] == b"\x02":
        raw = raw[1:]
    return raw.decode("utf-8", errors="strict")


def decode_rocksdict_value(raw: Any) -> str:
    """
    rocksdict may return values as `bytes` or `str` depending on the underlying DB.
    Our Rust code writes values as "rocksdict strings" with an optional 0x02 prefix.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (bytes, bytearray, memoryview)):
        b = bytes(raw)
        if b[:1] == b"\x02":
            b = b[1:]
        return b.decode("utf-8", errors="strict")
    return str(raw)


def iter_samples_from_rocksdb(rocks_path: Path, limit: Optional[int]) -> Iterator[Tuple[str, int]]:
    try:
        from rocksdict import Rdict  # type: ignore
    except Exception as e:
        raise SystemExit(
            "missing dependency 'rocksdict'. In py-scripts run:\n"
            "  uv venv && uv pip install -e .\n"
            f"Import error: {e}"
        )

    # Rdict is a dict-like wrapper over RocksDB, typically opens in read-only.
    db = Rdict(str(rocks_path))
    count = 0
    for k, v in db.items():
        try:
            if isinstance(k, str):
                key = k
            else:
                key = decode_rocksdict_string(bytes(k))
        except Exception:
            continue
        if not key.startswith("sample:"):
            continue
        try:
            raw = decode_rocksdict_value(v)
            obj = json.loads(raw)
        except Exception:
            continue
        text = obj.get("text") or ""
        label = int(obj.get("label") or 0)
        yield (str(text), label)
        count += 1
        if limit is not None and count >= limit:
            break


def chunks(items: List[Any], size: int) -> Iterator[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def compute_confusion(prob: List[float], actual: List[bool], threshold: float) -> Tuple[int, int, int, int]:
    tp = tn = fp = fn = 0
    for p, a in zip(prob, actual):
        pred = p >= threshold
        if pred and a:
            tp += 1
        elif (not pred) and (not a):
            tn += 1
        elif pred and (not a):
            fp += 1
        else:
            fn += 1
    return tp, tn, fp, fn


def compute_counts_streaming(
    probabilities: List[float],
    actual: List[bool],
    thresholds: List[float],
) -> List[Tuple[int, int, int, int]]:
    # For each threshold, keep tp/tn/fp/fn.
    tps = [0] * len(thresholds)
    tns = [0] * len(thresholds)
    fps = [0] * len(thresholds)
    fns = [0] * len(thresholds)
    for p, a in zip(probabilities, actual):
        for i, t in enumerate(thresholds):
            pred = p >= t
            if pred and a:
                tps[i] += 1
            elif (not pred) and (not a):
                tns[i] += 1
            elif pred and (not a):
                fps[i] += 1
            else:
                fns[i] += 1
    return list(zip(tps, tns, fps, fns))


def ratio(n: float, d: float) -> float:
    return 0.0 if d == 0 else (n / d)


def f1_from_counts(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    accuracy = ratio(tp + tn, tp + tn + fp + fn)
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def print_metrics_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    cols = [
        ("threshold", "thr"),
        ("f1", "f1"),
        ("precision", "prec"),
        ("recall", "rec"),
        ("accuracy", "acc"),
        ("tp", "tp"),
        ("fp", "fp"),
        ("tn", "tn"),
        ("fn", "fn"),
        ("evaluated", "n"),
        ("elapsed_secs", "secs"),
    ]

    def fmt(k: str, v: Any) -> str:
        if v is None:
            return "-"
        if k == "threshold":
            return f"{float(v):.3f}"
        if k in {"f1", "precision", "recall", "accuracy"}:
            return f"{float(v):.4f}"
        if k == "elapsed_secs":
            return f"{float(v):.3f}"
        return str(v)

    headers = [short for _, short in cols]
    table = [headers]
    for r in rows:
        table.append([fmt(k, r.get(k)) for k, _ in cols])
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]

    def join(row: List[str]) -> str:
        return "  ".join(row[i].rjust(widths[i]) for i in range(len(row)))

    print("\n== Metrics (offline, per threshold) ==")
    print(join(table[0]))
    print("  ".join("-" * w for w in widths))
    for row in table[1:]:
        print(join(row))


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a moderation profile offline (read-only RocksDB + local runtime inference).")
    ap.add_argument("--profile", required=True, help="profile name under configs/mod_profiles/<profile>")
    ap.add_argument(
        "--thresholds",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="comma-separated thresholds in [0,1]",
    )
    ap.add_argument("--limit", type=int, default=0, help="limit samples for quick runs (0 = no limit)")
    ap.add_argument("--batch-size", type=int, default=256, help="batch size for vectorized inference")
    ap.add_argument("--out-json", default=None, help="write full JSON report to this path")
    args = ap.parse_args()

    profile = args.profile.strip()
    profile_dir = CONFIGS_DIR / profile
    if not profile_dir.exists():
        raise SystemExit(f"profile dir not found: {profile_dir}")

    thresholds = parse_thresholds(args.thresholds)
    for t in thresholds:
        if t < 0.0 or t > 1.0:
            raise SystemExit(f"threshold out of range [0,1]: {t}")

    profile_json = load_profile_json(profile)
    local_model_type = ((profile_json.get("local_model_type") or "").strip().lower() if isinstance(profile_json, dict) else "")
    if not local_model_type:
        raise SystemExit(f"missing local_model_type in {profile_dir / 'profile.json'}")

    runtime_files, marker_files = model_artifacts_for_type(profile_dir, local_model_type)
    extra_report = [
        profile_dir / ".train_status.json",
        profile_dir / "train.log",
        profile_dir / "history.rocks",
        profile_dir / "history.db.bak",
        profile_dir / "keywords.txt",
    ]

    print("PrismGuard profile evaluation (offline)")
    print(f"- repo_root:  {REPO_ROOT}")
    print(f"- profile:    {profile}")
    print(f"- model_type: {local_model_type}")
    print(f"- thresholds: {', '.join(f'{t:.3f}' for t in thresholds)}")
    if args.limit and args.limit > 0:
        print(f"- limit:      {args.limit}")

    print_file_report("Runtime files (required for inference)", runtime_files)
    print_file_report("Training marker files (existence indicates trained artifacts)", marker_files)
    print_file_report("Other files", extra_report)

    if local_model_type != "hashlinear":
        raise SystemExit(f"offline inference currently implemented for hashlinear only (got {local_model_type})")

    meta_path = profile_dir / "hashlinear_runtime.json"
    coef_path = profile_dir / "hashlinear_runtime.coef.f32"
    runtime = HashlinearRuntime.load(meta_path, coef_path)

    rocks_path = profile_dir / "history.rocks"
    if not rocks_path.exists():
        raise SystemExit(f"history.rocks not found: {rocks_path}")

    limit = args.limit if args.limit and args.limit > 0 else None

    # Fast path: scikit-learn HashingVectorizer (C-accelerated hashing + sparse dot)
    # This avoids recomputing inference for every threshold and avoids Python-level ngram loops.
    try:
        import numpy as np  # type: ignore
        from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore

        def preprocess(text: str) -> str:
            s = text.replace("\r", " ").replace("\n", " ")
            if runtime.lowercase:
                s = s.lower()
            return normalize_spaces(s)

        vectorizer = HashingVectorizer(
            n_features=runtime.n_features,
            analyzer="char" if runtime.analyzer == "char" else "word",
            ngram_range=runtime.ngram_range,
            alternate_sign=runtime.alternate_sign,
            norm="l2" if runtime.norm == "l2" else None,
            lowercase=False,  # handled in preprocess to match Rust ordering
            preprocessor=preprocess,
        )
        coef = np.asarray(runtime.coef, dtype=np.float32)

        started = time.time()
        n = 0
        pos = 0
        neg = 0
        # per-threshold counts
        tps = np.zeros(len(thresholds), dtype=np.int64)
        tns = np.zeros(len(thresholds), dtype=np.int64)
        fps = np.zeros(len(thresholds), dtype=np.int64)
        fns = np.zeros(len(thresholds), dtype=np.int64)

        batch_texts: List[str] = []
        batch_actual: List[bool] = []

        def flush_batch() -> None:
            nonlocal n, pos, neg, batch_texts, batch_actual, tps, tns, fps, fns
            if not batch_texts:
                return
            X = vectorizer.transform(batch_texts)
            scores = X.dot(coef).astype(np.float64) + float(runtime.intercept)
            # sigmoid vectorized
            probs = 1.0 / (1.0 + np.exp(-scores))
            actual_arr = np.asarray(batch_actual, dtype=bool)
            for i, thr in enumerate(thresholds):
                pred = probs >= thr
                tps[i] += np.sum(pred & actual_arr)
                tns[i] += np.sum((~pred) & (~actual_arr))
                fps[i] += np.sum(pred & (~actual_arr))
                fns[i] += np.sum((~pred) & actual_arr)
            n += len(batch_texts)
            pos += int(np.sum(actual_arr))
            neg += int(len(batch_texts) - np.sum(actual_arr))
            batch_texts = []
            batch_actual = []

        for text, label in iter_samples_from_rocksdb(rocks_path, limit=limit):
            batch_texts.append(text)
            is_pos = (label != 0)
            batch_actual.append(is_pos)
            if len(batch_texts) >= max(1, int(args.batch_size)):
                flush_batch()
        flush_batch()

        elapsed_pred = time.time() - started

        if n == 0:
            raise SystemExit("no samples found in history.rocks (sample:* keys)")

        print("\n== Dataset ==")
        print(f"- evaluated: {n}")
        print(f"- label_1 (violation): {pos}")
        print(f"- label_0 (pass): {neg}")
        print(f"- inference_elapsed_secs: {elapsed_pred:.3f}")
        print(f"- avg_secs_per_sample: {elapsed_pred / n:.6f}")

        rows: List[Dict[str, Any]] = []
        for i, t in enumerate(thresholds):
            metrics = f1_from_counts(int(tps[i]), int(tns[i]), int(fps[i]), int(fns[i]))
            rows.append(
                {
                    "threshold": t,
                    "tp": int(tps[i]),
                    "tn": int(tns[i]),
                    "fp": int(fps[i]),
                    "fn": int(fns[i]),
                    "evaluated": n,
                    "elapsed_secs": 0.0,
                    **metrics,
                }
            )

        print_metrics_table(rows)

        report = {
            "profile": profile,
            "local_model_type": local_model_type,
            "thresholds": thresholds,
            "dataset": {"evaluated": n, "label_1": pos, "label_0": neg},
            "inference_elapsed_secs": elapsed_pred,
            "metrics": rows,
            "engine": "sklearn_hashingvectorizer",
        }

        if args.out_json:
            out_path = Path(args.out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nWrote: {out_path}")

        return 0

    except Exception as e:
        print(f"\n!! fast path unavailable, falling back to slow pure-python inference: {e}")

    # Slow fallback path: pure Python per-sample prediction and store probabilities.
    probabilities: List[float] = []
    actual: List[bool] = []

    started = time.time()
    n = 0
    pos = 0
    neg = 0
    for text, label in iter_samples_from_rocksdb(rocks_path, limit=limit):
        n += 1
        is_pos = (label != 0)
        if is_pos:
            pos += 1
        else:
            neg += 1
        probabilities.append(runtime.predict_proba(text))
        actual.append(is_pos)
    elapsed_pred = time.time() - started

    if n == 0:
        raise SystemExit("no samples found in history.rocks (sample:* keys)")

    print("\n== Dataset ==")
    print(f"- evaluated: {n}")
    print(f"- label_1 (violation): {pos}")
    print(f"- label_0 (pass): {neg}")
    print(f"- inference_elapsed_secs: {elapsed_pred:.3f}")
    print(f"- avg_secs_per_sample: {elapsed_pred / n:.6f}")

    rows: List[Dict[str, Any]] = []
    for t in thresholds:
        t0 = time.time()
        tp, tn, fp, fn = compute_confusion(probabilities, actual, threshold=t)
        metrics = f1_from_counts(tp, tn, fp, fn)
        rows.append(
            {
                "threshold": t,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "evaluated": n,
                "elapsed_secs": round(time.time() - t0, 6),
                **metrics,
            }
        )

    print_metrics_table(rows)

    report = {
        "profile": profile,
        "local_model_type": local_model_type,
        "thresholds": thresholds,
        "dataset": {"evaluated": n, "label_1": pos, "label_0": neg},
        "inference_elapsed_secs": elapsed_pred,
        "metrics": rows,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
