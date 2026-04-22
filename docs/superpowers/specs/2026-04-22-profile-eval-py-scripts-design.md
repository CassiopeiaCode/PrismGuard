# Profile Eval (Python scripts via running PrismGuard) - Design

**Date:** 2026-04-22

## Goal

Provide a fast-to-iterate evaluation tool that:

1. Accepts a moderation `profile`
2. Reports local model artifacts for that profile (runtime + training marker files)
3. Uses the **already running** PrismGuard Rust service to:
   - Open the profile database in read-only mode (server-side)
   - Run predictions/metrics for the profile's `local_model_type`
   - Produce F1 scores across multiple confidence thresholds

This intentionally avoids compiling or testing the Rust codebase, and only depends on the existing binary/service.

## Non-Goals

- Reimplement local inference in Python (hashlinear/bow/fasttext)
- Read/iterate `history.rocks` directly in Python
- Add new Rust endpoints or change existing service behavior

## Approach

### Python tool layout

- Directory: `py-scripts/`
- Dependency management: `uv` virtualenv in `py-scripts/.venv`
- Script: `py-scripts/eval_profile.py`

### Service discovery

The script determines the service base URL in this order:

1. `--base-url` CLI override
2. `PRISMGUARD_BASE_URL` or `BASE_URL` env var
3. Repository root `.env` values: `HOST` + `PORT`
4. Fallback: `http://127.0.0.1:8000`

If `.env` binds to `0.0.0.0`, the client uses `127.0.0.1`.

### Data & metrics retrieval

The script calls:

- `GET /debug/profile/<profile>` to retrieve:
  - `live_sample_count` (used for full-dataset evaluation size)
  - `history_rocks_path` (reported)
  - `training_status` (reported)
  - plus other debug metadata

- `GET /debug/profile/<profile>/metrics?...` per threshold:
  - `sample_size=<live_sample_count>`
  - `sampling=latest_full|random_full|balanced`
  - `threshold=<t>`

The service performs DB open/read-only behavior internally; the script reports the output as evidence.

### Model selection

Only the profile's configured `local_model_type` is evaluated.

The script still reports missing/present runtime + marker files for that model type to explain failures quickly.

### Output

- Human-readable artifact report (exists/size/mtime)
- Human-readable metrics table per threshold:
  - `precision`, `recall`, `f1`, `accuracy`
  - `tp`, `tn`, `fp`, `fn`, `evaluated`
  - elapsed seconds per request
- Optional `--out-json` for machine-readable output

## Risks / Notes

- Full-dataset metrics may be slow (depends on sample count, model type, and server load).
- The `/debug/profile/<profile>/metrics` route can be feature-gated; the tool should error clearly on non-200 responses.

