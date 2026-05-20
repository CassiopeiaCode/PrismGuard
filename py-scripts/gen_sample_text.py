#!/usr/bin/env python3
"""Print a random sample text from history.rocks, optionally filtered by violation label and category."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"

_LOCK_FD: Optional[int] = None


def decode_rocksdict_string(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    b = bytes(raw)
    if b[:1] == b"\x02":
        b = b[1:]
    return b.decode("utf-8", errors="strict")


def iter_items(db) -> Iterator[Tuple[str, Any]]:
    for k, v in db.items():
        try:
            key = decode_rocksdict_string(k)
        except Exception:
            continue
        yield key, v


def try_acquire_rocks_lock(rocks_path: Path) -> None:
    lock_path = rocks_path / "LOCK"
    if not lock_path.exists():
        raise SystemExit(f"LOCK file not found: {lock_path} (is this a valid RocksDB dir?)")

    if os.name != "posix":
        raise SystemExit("lock check only implemented for POSIX systems")

    import fcntl

    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise SystemExit(
                f"RocksDB appears to be in use (failed to lock {lock_path}: {e}).\n"
                "Stop the service/process that is using this RocksDB directory and retry."
            )
    finally:
        global _LOCK_FD
        _LOCK_FD = fd


def is_sample_key(key: str) -> bool:
    return key.startswith("sample:")


def load_samples(
    rocks_path: Path,
    label: Optional[int],
    category: Optional[str],
    limit: Optional[int],
) -> list[dict]:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))
    samples: list[dict] = []
    seen: set[str] = set()

    for key, raw_value in iter_items(db):
        if not is_sample_key(key):
            continue
        try:
            obj = json.loads(decode_rocksdict_string(raw_value))
        except Exception:
            continue

        if label is not None and int(obj.get("label") or 0) != label:
            continue

        if category is not None:
            cat = (obj.get("category") or "").strip()
            if cat.lower() != category.strip().lower():
                continue

        text = (obj.get("text") or "").strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        samples.append(obj)

        if limit is not None and len(samples) >= limit:
            break

    return samples


def _label_name(label: int) -> str:
    return "violation" if label == 1 else "non-violation"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print a random sample text from a profile's history.rocks, "
        "optionally filtered by violation label and category."
    )
    ap.add_argument(
        "--profile",
        default="4claudecode",
        help="profile name under configs/mod_profiles/<profile> (default: 4claudecode)",
    )
    ap.add_argument(
        "--rocks-path",
        default=None,
        help="path to history.rocks directory (overrides --profile)",
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--violation",
        action="store_true",
        dest="violation",
        default=None,
        help="only pick violation samples (label=1)",
    )
    group.add_argument(
        "--no-violation",
        action="store_false",
        dest="violation",
        default=None,
        help="only pick non-violation samples (label=0)",
    )
    ap.add_argument(
        "--category",
        default=None,
        help="filter by category (e.g. 色情, 政治, 越狱). Only meaningful with --violation.",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="output as JSON with metadata",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed for reproducibility",
    )
    ap.add_argument(
        "--scan-limit",
        type=int,
        default=0,
        help="max samples to scan from DB (0 = no limit, scans all)",
    )
    args = ap.parse_args()

    if args.rocks_path:
        rocks_path = Path(args.rocks_path).resolve()
    else:
        rocks_path = (CONFIGS_DIR / args.profile.strip() / "history.rocks").resolve()

    if not rocks_path.exists():
        raise SystemExit(f"history.rocks not found: {rocks_path}")

    if args.seed is not None:
        random.seed(args.seed)

    # Determine label filter
    if args.violation is None:
        label = random.choice([0, 1])
    else:
        label = 1 if args.violation else 0

    scan_limit = args.scan_limit if args.scan_limit and args.scan_limit > 0 else None

    # If user didn't explicitly pick violation, resolve random choice now
    # so we can filter by label from the start
    try_acquire_rocks_lock(rocks_path)

    samples = load_samples(
        rocks_path,
        label=label,
        category=args.category if label == 1 else None,
        limit=scan_limit,
    )

    if not samples:
        filters = []
        filters.append(f"label={label} ({_label_name(label)})")
        if args.category:
            filters.append(f"category={args.category}")
        raise SystemExit(
            f"No samples found matching: {', '.join(filters)} "
            f"in {rocks_path}"
        )

    chosen = random.choice(samples)

    if args.json:
        out = {
            "profile": args.profile,
            "rocks_path": str(rocks_path),
            "label": chosen.get("label"),
            "category": chosen.get("category"),
            "text": chosen.get("text"),
            "id": chosen.get("id"),
            "created_at": chosen.get("created_at"),
            "matched_samples": len(samples),
        }
        if args.seed is not None:
            out["seed"] = args.seed
        json.dump(out, ensure_ascii=False, indent=2, fp=sys.stdout)
        print()
    else:
        print(chosen.get("text", ""))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
