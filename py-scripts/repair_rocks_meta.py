#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs" / "mod_profiles"


def encode_rocksdict_string(s: str) -> bytes:
    # Mirror Rust: prefix 0x02 + UTF-8 bytes.
    return b"\x02" + s.encode("utf-8")


def decode_rocksdict_string(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    b = bytes(raw)
    if b[:1] == b"\x02":
        b = b[1:]
    return b.decode("utf-8", errors="strict")


def iter_db_items(db) -> Iterator[Tuple[str, Any]]:
    for k, v in db.items():
        try:
            key = decode_rocksdict_string(k)
        except Exception:
            continue
        yield key, v


def parse_sample_id(key: str) -> Optional[int]:
    if not key.startswith("sample:"):
        return None
    suffix = key.split("sample:", 1)[1]
    # Format is sample:{id:020}, but be tolerant.
    try:
        return int(suffix)
    except Exception:
        return None


def md5_hex(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def try_acquire_rocks_lock(rocks_path: Path) -> None:
    """
    Refuse to run if the RocksDB lock is held by another process.

    RocksDB uses fcntl locks; on Linux, `fcntl.lockf` maps to fcntl locking.
    """
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
        # Keep lock only for the duration of this script run by holding fd open.
        # Caller is expected to keep process alive while operating on DB.
        # NOTE: We intentionally do NOT close fd here; we return it via global keeper.
        global _LOCK_FD  # noqa: PLW0603
        _LOCK_FD = fd


_LOCK_FD: Optional[int] = None


@dataclass
class RebuildResult:
    sample_count: int
    count_0: int
    count_1: int
    max_sample_id: int
    pointer_count: int


def rebuild_meta_and_pointers(rocks_path: Path, fix_pointers: bool) -> Tuple[Dict[str, str], Dict[str, str], RebuildResult]:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))

    sample_count = 0
    count_0 = 0
    count_1 = 0
    max_sample_id = 0
    pointers: Dict[str, int] = {}

    for key, raw_value in iter_db_items(db):
        sample_id = parse_sample_id(key)
        if sample_id is None:
            continue

        sample_count += 1
        if sample_id > max_sample_id:
            max_sample_id = sample_id

        try:
            value_text = decode_rocksdict_string(raw_value)
            obj = json.loads(value_text)
        except Exception:
            # If sample is corrupted, skip it rather than stopping the repair.
            continue

        label = int(obj.get("label") or 0)
        if label == 0:
            count_0 += 1
        else:
            count_1 += 1

        if fix_pointers:
            text_hash = obj.get("text_hash")
            if not isinstance(text_hash, str) or not text_hash:
                text = obj.get("text") or ""
                text_hash = md5_hex(str(text))
            current = pointers.get(text_hash)
            if current is None or sample_id > current:
                pointers[text_hash] = sample_id

    next_id = 1 if max_sample_id <= 0 else (max_sample_id + 1)
    meta_updates = {
        "meta:next_id": str(next_id),
        "meta:count": str(sample_count),
        "meta:count:0": str(count_0),
        "meta:count:1": str(count_1),
    }

    pointer_updates: Dict[str, str] = {}
    if fix_pointers:
        for text_hash, sid in pointers.items():
            pointer_updates[f"text_latest:{text_hash}"] = str(sid)

    return (
        meta_updates,
        pointer_updates,
        RebuildResult(
            sample_count=sample_count,
            count_0=count_0,
            count_1=count_1,
            max_sample_id=max_sample_id,
            pointer_count=len(pointer_updates),
        ),
    )


def apply_updates(rocks_path: Path, meta_updates: Dict[str, str], pointer_updates: Dict[str, str]) -> None:
    from rocksdict import Rdict  # type: ignore

    db = Rdict(str(rocks_path))
    for k, v in meta_updates.items():
        db[encode_rocksdict_string(k)] = encode_rocksdict_string(v)
    for k, v in pointer_updates.items():
        db[encode_rocksdict_string(k)] = encode_rocksdict_string(v)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Repair PrismGuard RocksDB metadata (meta:* and optionally text_latest:*). Refuses to run if DB is in use."
    )
    ap.add_argument("--profile", default=None, help="profile name under configs/mod_profiles/<profile>")
    ap.add_argument("--rocks-path", default=None, help="path to history.rocks directory (overrides --profile)")
    ap.add_argument("--fix-pointers", action="store_true", help="rebuild text_latest:* pointers from sample records")
    ap.add_argument("--apply", action="store_true", help="actually write changes (default is dry-run)")
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

    # Occupancy check first: fail fast if another process holds the lock.
    try_acquire_rocks_lock(rocks_path)

    meta_updates, pointer_updates, result = rebuild_meta_and_pointers(
        rocks_path=rocks_path,
        fix_pointers=bool(args.fix_pointers),
    )

    print("RocksDB repair plan")
    print(f"- rocks_path: {rocks_path}")
    print(f"- samples: {result.sample_count} (label0={result.count_0}, label1={result.count_1}, max_id={result.max_sample_id})")
    print(f"- fix_pointers: {bool(args.fix_pointers)} (pointer_updates={result.pointer_count})")
    print(f"- mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print("\nMeta updates:")
    for k, v in meta_updates.items():
        print(f"- {k} = {v}")
    if args.fix_pointers:
        print("\nPointer updates:")
        # Avoid dumping huge lists.
        shown = 0
        for k, v in pointer_updates.items():
            if shown >= 10:
                print(f"- ... (+{len(pointer_updates) - 10} more)")
                break
            print(f"- {k} = {v}")
            shown += 1

    if args.apply:
        apply_updates(rocks_path, meta_updates, pointer_updates)
        print("\nApplied updates successfully.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

