# Clip to Text Pro

Webowa aplikacja do transkrypcji wideo na tekst:
- upload plików `.mp4`, `.mov`, `.mkv`,
- ekstrakcja audio przez `ffmpeg`,
- transkrypcja lokalnie przez `faster-whisper`,
- progres live (SSE + fallback do pollingu),
- heartbeat SSE (stabilniejsze połączenie przy długich jobach),
- historia jobów zapisywana w SQLite,
- cache transkrypcji po hashu pliku,
- limity kolejki (`429`, gdy za dużo aktywnych zadań),
- UI z trybem jasnym/ciemnym.

## Wymagania
- Python `3.10+`
- `ffmpeg` (oraz opcjonalnie `ffprobe` jeśli włączysz limit długości nagrania)

Przykładowa instalacja `ffmpeg`:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS (Homebrew): `brew install ffmpeg`

## Instalacja
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# opcjonalnie: cp .env.example .env i ustaw własne limity/profile
```

## Uruchomienie
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Aplikacja: `http://127.0.0.1:8000`

## Najważniejsze endpointy
- `POST /api/jobs` - tworzy nowe zadanie transkrypcji
- `GET /api/jobs` - lista ostatnich zadań
- `GET /api/jobs/{job_id}` - status zadania
- `GET /api/jobs/{job_id}/events` - stream statusu (SSE)
- `POST /api/jobs/{job_id}/cancel` - anulowanie zadania
- `GET /api/jobs/{job_id}/result` - gotowy tekst (JSON)
- `GET /api/jobs/{job_id}/download` - pobranie `.txt`
- `POST /api/transcribe` - tryb prosty (sync, bez job history)
- `GET /health` - health + metryki runtime

## Konfiguracja (ENV)
- `MAX_UPLOAD_MB` (domyślnie: `500`)
- `MAX_VIDEO_DURATION_SECONDS` (domyślnie: `0`, wyłączone)
- `MAX_PENDING_JOBS` (domyślnie: `64`)

- `WHISPER_FAST_MODEL` (domyślnie: `tiny`)
- `WHISPER_ACCURATE_MODEL` (domyślnie: `medium`)
- `WHISPER_DEVICE` (domyślnie: `auto`)
- `WHISPER_COMPUTE_TYPE` (domyślnie: `int8`)
- `WHISPER_CPU_THREADS` (domyślnie: `0`)
- `WHISPER_NUM_WORKERS` (domyślnie: `1`)
- `WHISPER_BATCH_SIZE` (domyślnie: `1`)
- `WHISPER_FAST_VAD_FILTER` (domyślnie: `0`)
- `WHISPER_FAST_WITHOUT_TIMESTAMPS` (domyślnie: `1`)
- `WHISPER_ACCURATE_WITHOUT_TIMESTAMPS` (domyślnie: `1`)

- `TRANSCRIBE_WORKERS` (domyślnie: połowa rdzeni CPU, max 4)
- `PRELOAD_FAST_MODEL` (domyślnie: `1`)
- `PRELOAD_ACCURATE_MODEL` (domyślnie: `0`)

- `JOB_TTL_SECONDS` (domyślnie: `7200`, TTL jobów w pamięci)
- `JOB_RETENTION_SECONDS` (domyślnie: `604800`, retencja jobów w SQLite)
- `TRANSCRIPT_CACHE_TTL_SECONDS` (domyślnie: `86400`)
- `TRANSCRIPT_CACHE_MAX_ITEMS` (domyślnie: `256`)
- `JOBS_DB_PATH` (domyślnie: `data/jobs.sqlite3`)

- `SSE_POLL_INTERVAL_SECONDS` (domyślnie: `0.8`)
- `SSE_MAX_SECONDS` (domyślnie: `3600`)
- `SSE_HEARTBEAT_SECONDS` (domyślnie: `12`)
- `MAINTENANCE_INTERVAL_SECONDS` (domyślnie: `300`)

- `FFMPEG_BIN` (domyślnie: `ffmpeg`)
- `FFPROBE_BIN` (domyślnie: `ffprobe`)
- `FFMPEG_THREADS` (domyślnie: `0`, auto)
- `LOG_LEVEL` (domyślnie: `INFO`)

## Szybkie profile wydajności
CPU (większy throughput):
```bash
export WHISPER_DEVICE=cpu
export WHISPER_COMPUTE_TYPE=int8
export WHISPER_FAST_MODEL=tiny
export WHISPER_CPU_THREADS=8
export WHISPER_NUM_WORKERS=2
export TRANSCRIBE_WORKERS=2
export PRELOAD_FAST_MODEL=1
```

GPU (najlepsza szybkość pojedynczego zadania):
```bash
export WHISPER_DEVICE=cuda
export WHISPER_COMPUTE_TYPE=float16
export WHISPER_FAST_MODEL=tiny
export WHISPER_BATCH_SIZE=8
export TRANSCRIBE_WORKERS=1
export PRELOAD_FAST_MODEL=1
```

## Testy (smoke)
Jeśli masz `pytest`:
```bash
pip install -r requirements-dev.txt
pytest -q
```

## Obsłużone błędy
- niepoprawny format pliku,
- nieobsługiwany MIME,
- zbyt duży plik,
- przekroczony limit długości nagrania (opcjonalnie),
- brak ścieżki audio,
- brak `ffmpeg`/`ffprobe`,
- brak `faster-whisper`.
