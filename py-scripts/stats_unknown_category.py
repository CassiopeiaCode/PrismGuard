#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"


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


def is_sample_key(key: str) -> bool:
    return key.startswith("sample:")


def try_acquire_rocks_lock(rocks_path: Path) -> None:
    lock_path = rocks_path / "LOCK"
    if not lock_path.exists():
        raise SystemExit(f"LOCK file not found: {lock_path} (is this a valid RocksDB dir?)")

    if os.name != "posix":
        raise SystemExit("lock check only implemented for POSIX systems")

    import fcntl  # POSIX only

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
        global _LOCK_FD  # noqa: PLW0603
        _LOCK_FD = fd


_LOCK_FD: Optional[int] = None


@dataclass
class UnknownStats:
    total_samples: int
    unknown_category: int
    missing_category: int
    other_category: int


def compute_unknown_stats(rocks_path: Path, limit: Optional[int]) -> UnknownStats:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))
    total = 0
    unknown = 0
    missing = 0
    other = 0

    for key, raw_value in iter_items(db):
        if not is_sample_key(key):
            continue
        try:
            obj = json.loads(decode_rocksdict_string(raw_value))
        except Exception:
            continue

        total += 1
        category = obj.get("category", None)
        if category is None:
            missing += 1
        elif isinstance(category, str) and category.strip().lower() == "unknown":
            unknown += 1
        else:
            other += 1

        if limit is not None and total >= limit:
            break

    return UnknownStats(
        total_samples=total,
        unknown_category=unknown,
        missing_category=missing,
        other_category=other,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Count samples where category == 'unknown' in history.rocks.")
    ap.add_argument("--profile", default=None, help="profile name under configs/mod_profiles/<profile>")
    ap.add_argument("--rocks-path", default=None, help="path to history.rocks directory (overrides --profile)")
    ap.add_argument("--limit", type=int, default=0, help="limit scanned samples (0 = no limit)")
    args = ap.parse_args()

    if not args.profile and not args.rocks_path:
        raise SystemExit("must provide --profile or --rocks-path")

    if args.rocks_path:
        rocks_path = Path(args.rocks_path).resolve()
    else:
        profile_dir = CONFIGS_DIR / str(args.profile).strip()
        rocks_path = (profile_dir / "history.rocks").resolve()

    if not rocks_path.exists():
        raise SystemExit(f"history.rocks not found: {rocks_path}")

    # Avoid racing with the running service.
    try_acquire_rocks_lock(rocks_path)

    limit = args.limit if args.limit and args.limit > 0 else None
    stats = compute_unknown_stats(rocks_path, limit=limit)
    scanned = stats.total_samples
    print("Unknown category stats")
    print(f"- rocks_path: {rocks_path}")
    print(f"- scanned_samples: {scanned}{'' if limit is None else f' (limit={limit})'}")
    if scanned > 0:
        print(f"- category_unknown: {stats.unknown_category} ({stats.unknown_category / scanned:.2%})")
        print(f"- category_missing: {stats.missing_category} ({stats.missing_category / scanned:.2%})")
        print(f"- category_other:   {stats.other_category} ({stats.other_category / scanned:.2%})")
    else:
        print("- no samples found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

