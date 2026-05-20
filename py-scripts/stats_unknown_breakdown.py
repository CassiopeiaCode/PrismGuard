#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import Counter, defaultdict
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


def is_sample_key(key: str) -> bool:
    return key.startswith("sample:")


def parse_created_at(raw: Any) -> Optional[dt.datetime]:
    if not isinstance(raw, str) or not raw.strip():
        return None
    # Format from src/storage.rs current_local_timestamp(): "YYYY-MM-DD HH:MM:SS"
    try:
        return dt.datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


@dataclass
class Breakdown:
    total_samples: int
    unknown_samples: int
    unknown_label0: int
    unknown_label1: int
    unknown_missing_created_at: int
    unknown_first: Optional[dt.datetime]
    unknown_last: Optional[dt.datetime]
    unknown_by_day: Counter[str]
    total_by_day: Counter[str]


def scan_breakdown(rocks_path: Path, limit: Optional[int]) -> Breakdown:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))

    total_samples = 0
    unknown_samples = 0
    unknown_label0 = 0
    unknown_label1 = 0
    unknown_missing_created_at = 0
    unknown_first: Optional[dt.datetime] = None
    unknown_last: Optional[dt.datetime] = None

    unknown_by_day: Counter[str] = Counter()
    total_by_day: Counter[str] = Counter()

    for key, raw_value in iter_items(db):
        if not is_sample_key(key):
            continue
        try:
            obj = json.loads(decode_rocksdict_string(raw_value))
        except Exception:
            continue

        total_samples += 1

        created = parse_created_at(obj.get("created_at"))
        if created is not None:
            total_by_day[created.date().isoformat()] += 1

        category = obj.get("category", None)
        is_unknown = isinstance(category, str) and category.strip().lower() == "unknown"
        if is_unknown:
            unknown_samples += 1
            label = int(obj.get("label") or 0)
            if label == 0:
                unknown_label0 += 1
            else:
                unknown_label1 += 1

            if created is None:
                unknown_missing_created_at += 1
            else:
                day = created.date().isoformat()
                unknown_by_day[day] += 1
                unknown_first = created if unknown_first is None else min(unknown_first, created)
                unknown_last = created if unknown_last is None else max(unknown_last, created)

        if limit is not None and total_samples >= limit:
            break

    return Breakdown(
        total_samples=total_samples,
        unknown_samples=unknown_samples,
        unknown_label0=unknown_label0,
        unknown_label1=unknown_label1,
        unknown_missing_created_at=unknown_missing_created_at,
        unknown_first=unknown_first,
        unknown_last=unknown_last,
        unknown_by_day=unknown_by_day,
        total_by_day=total_by_day,
    )


def print_top_days(title: str, counter: Counter[str], total_by_day: Counter[str], limit: int) -> None:
    print(f"\n== {title} ==")
    for day, cnt in counter.most_common(limit):
        denom = total_by_day.get(day, 0)
        ratio = (cnt / denom) if denom else 0.0
        print(f"- {day}: unknown={cnt} / total={denom} ({ratio:.2%})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Break down category=='unknown' samples by created_at time buckets to infer whether they come from old fallback logic."
    )
    ap.add_argument("--profile", required=True, help="profile name under configs/mod_profiles/<profile>")
    ap.add_argument("--limit", type=int, default=0, help="limit scanned samples (0 = no limit)")
    ap.add_argument("--top-days", type=int, default=20, help="top N days to display")
    ap.add_argument("--lookback-days", type=int, default=7, help="report unknown counts in the last N days")
    args = ap.parse_args()

    profile = args.profile.strip()
    rocks_path = (CONFIGS_DIR / profile / "history.rocks").resolve()
    if not rocks_path.exists():
        raise SystemExit(f"history.rocks not found: {rocks_path}")

    try_acquire_rocks_lock(rocks_path)

    limit = args.limit if args.limit and args.limit > 0 else None
    breakdown = scan_breakdown(rocks_path, limit=limit)

    now = dt.datetime.now()
    lookback_start = (now - dt.timedelta(days=max(1, int(args.lookback_days)))).date().isoformat()
    unknown_recent = sum(cnt for day, cnt in breakdown.unknown_by_day.items() if day >= lookback_start)
    total_recent = sum(cnt for day, cnt in breakdown.total_by_day.items() if day >= lookback_start)

    print("Unknown category breakdown")
    print(f"- profile: {profile}")
    print(f"- rocks_path: {rocks_path}")
    print(f"- scanned_samples: {breakdown.total_samples}{'' if limit is None else f' (limit={limit})'}")
    if breakdown.total_samples > 0:
        print(f"- unknown_samples: {breakdown.unknown_samples} ({breakdown.unknown_samples / breakdown.total_samples:.2%})")
    print(f"- unknown_label0: {breakdown.unknown_label0}")
    print(f"- unknown_label1: {breakdown.unknown_label1}")
    print(f"- unknown_missing_created_at: {breakdown.unknown_missing_created_at}")
    print(f"- unknown_first: {breakdown.unknown_first}")
    print(f"- unknown_last:  {breakdown.unknown_last}")
    if total_recent > 0:
        print(
            f"- last_{args.lookback_days}d: unknown={unknown_recent} / total={total_recent} ({unknown_recent / total_recent:.2%})"
        )

    print_top_days(
        title=f"Top days by unknown count (top {args.top_days})",
        counter=breakdown.unknown_by_day,
        total_by_day=breakdown.total_by_day,
        limit=max(1, int(args.top_days)),
    )

    # Also show days with highest unknown ratio (but require enough samples).
    ratios: list[Tuple[str, float, int, int]] = []
    for day, unk in breakdown.unknown_by_day.items():
        total = breakdown.total_by_day.get(day, 0)
        if total < 50:
            continue
        ratios.append((day, unk / total, unk, total))
    ratios.sort(key=lambda t: t[1], reverse=True)
    print(f"\n== Top days by unknown ratio (min_total=50, top {args.top_days}) ==")
    for day, ratio, unk, total in ratios[: max(1, int(args.top_days))]:
        print(f"- {day}: unknown={unk} / total={total} ({ratio:.2%})")

    print("\nInference hint:")
    print("- If unknown samples cluster in older dates and drop to ~0 after deployment, they likely come from the old keyword fallback.")
    print("- If unknown continues appearing recently, LLM might be returning JSON with category='unknown' (or upstream formatting issues persist).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

