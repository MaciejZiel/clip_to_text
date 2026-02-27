# Clip to Text Pro

Web application for local video-to-text transcription:
- upload `.mp4`, `.mov`, `.mkv` files,
- extract audio with `ffmpeg`,
- transcribe locally with `faster-whisper`,
- choose output format: `txt` or `txt + srt`,
- live progress (SSE + polling fallback),
- heartbeat SSE for long-running jobs,
- persistent job history in SQLite,
- transcript cache based on file hash,
- optional `.srt` subtitle export,
- queue limits (`429` when too many active jobs),
- polished UI with light/dark mode.

## Requirements
- Python `3.10+`
- `ffmpeg` (and optionally `ffprobe` if video duration limits are enabled)

Example `ffmpeg` installation:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optional: cp .env.example .env and tune limits/profiles
```

## Run
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

App URL: `http://127.0.0.1:8000`

## Main Endpoints
- `POST /api/jobs` - create a new transcription job
- `GET /api/jobs` - list recent jobs
- `GET /api/jobs/{job_id}` - get job status
- `GET /api/jobs/{job_id}/events` - SSE status stream
- `POST /api/jobs/{job_id}/cancel` - cancel a job
- `GET /api/jobs/{job_id}/result` - get result as JSON
- `GET /api/jobs/{job_id}/subtitle` - get SRT subtitles as plain text
- `GET /api/jobs/{job_id}/download?format=txt|srt` - download `.txt` or `.srt`
- `POST /api/transcribe` - simple sync mode (no job history), supports `output_format=txt|txt_srt`
- `GET /health` - health + runtime metrics

## Configuration (ENV)
- `MAX_UPLOAD_MB` (default: `500`)
- `MAX_VIDEO_DURATION_SECONDS` (default: `0`, disabled)
- `MAX_PENDING_JOBS` (default: `64`)

- `WHISPER_FAST_MODEL` (default: `tiny`)
- `WHISPER_ACCURATE_MODEL` (default: `medium`)
- `WHISPER_DEVICE` (default: `auto`)
- `WHISPER_COMPUTE_TYPE` (default: `int8`)
- `WHISPER_CPU_THREADS` (default: `0`)
- `WHISPER_NUM_WORKERS` (default: `1`)
- `WHISPER_BATCH_SIZE` (default: `1`)
- `WHISPER_FAST_VAD_FILTER` (default: `0`)
- `WHISPER_FAST_WITHOUT_TIMESTAMPS` (default: `1`)
- `WHISPER_ACCURATE_WITHOUT_TIMESTAMPS` (default: `1`)

- `TRANSCRIBE_WORKERS` (default: half CPU cores, max 4)
- `PRELOAD_FAST_MODEL` (default: `1`)
- `PRELOAD_ACCURATE_MODEL` (default: `0`)

- `JOB_TTL_SECONDS` (default: `7200`, in-memory job TTL)
- `JOB_RETENTION_SECONDS` (default: `604800`, persisted job retention in SQLite)
- `TRANSCRIPT_CACHE_TTL_SECONDS` (default: `86400`)
- `TRANSCRIPT_CACHE_MAX_ITEMS` (default: `256`)
- `JOBS_DB_PATH` (default: `data/jobs.sqlite3`)

- `SSE_POLL_INTERVAL_SECONDS` (default: `0.8`)
- `SSE_MAX_SECONDS` (default: `3600`)
- `SSE_HEARTBEAT_SECONDS` (default: `12`)
- `MAINTENANCE_INTERVAL_SECONDS` (default: `300`)

- `FFMPEG_BIN` (default: `ffmpeg`)
- `FFPROBE_BIN` (default: `ffprobe`)
- `FFMPEG_THREADS` (default: `0`, auto)
- `LOG_LEVEL` (default: `INFO`)

## Quick Performance Profiles
Note: when `output_format=txt_srt` is selected, timestamps are enabled and processing is usually slower than plain `txt`.

CPU (higher throughput):
```bash
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
export WHISPER_FAST_MODEL=tiny
export WHISPER_CPU_THREADS=8
export WHISPER_NUM_WORKERS=2
export TRANSCRIBE_WORKERS=2
export PRELOAD_FAST_MODEL=1
```

GPU (fastest single-job latency):
```bash
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE_TYPE=float16
export WHISPER_FAST_MODEL=tiny
export WHISPER_BATCH_SIZE=8
export TRANSCRIBE_WORKERS=1
export PRELOAD_FAST_MODEL=1
```

## Smoke Tests
If `pytest` is available:
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Handled Errors
- invalid file format,
- unsupported MIME type,
- file too large,
- recording duration limit exceeded (optional),
- no audio track,
- missing `ffmpeg`/`ffprobe`,
- missing `faster-whisper`.
