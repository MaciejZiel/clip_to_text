"""Settings read from the environment, shared type aliases and logging."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv"}
ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-matroska",
    "video/webm",
    "application/octet-stream",
}
MKV_SIGNATURE = b"\x1a\x45\xdf\xa3"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = max(0, int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "0")))
MAX_PENDING_JOBS = max(1, int(os.getenv("MAX_PENDING_JOBS", "64")))

JOB_MEMORY_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "7200"))
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "604800"))
TRANSCRIPT_CACHE_TTL_SECONDS = int(os.getenv("TRANSCRIPT_CACHE_TTL_SECONDS", "86400"))
TRANSCRIPT_CACHE_MAX_ITEMS = max(1, int(os.getenv("TRANSCRIPT_CACHE_MAX_ITEMS", "256")))

DEFAULT_TRANSCRIBE_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
TRANSCRIBE_WORKERS = max(
    1, int(os.getenv("TRANSCRIBE_WORKERS", str(DEFAULT_TRANSCRIBE_WORKERS)))
)

PRELOAD_FAST_MODEL = os.getenv("PRELOAD_FAST_MODEL", "1") == "1"
PRELOAD_ACCURATE_MODEL = os.getenv("PRELOAD_ACCURATE_MODEL", "0") == "1"

JOBS_DIR = Path(tempfile.gettempdir()) / "clip_to_text_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOBS_DB_PATH = Path(os.getenv("JOBS_DB_PATH", str(DATA_DIR / "jobs.sqlite3")))
JOBS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SSE_POLL_INTERVAL_SECONDS = float(os.getenv("SSE_POLL_INTERVAL_SECONDS", "0.8"))
SSE_MAX_SECONDS = int(os.getenv("SSE_MAX_SECONDS", "3600"))
SSE_HEARTBEAT_SECONDS = max(3.0, float(os.getenv("SSE_HEARTBEAT_SECONDS", "12")))
MAINTENANCE_INTERVAL_SECONDS = max(
    30, int(os.getenv("MAINTENANCE_INTERVAL_SECONDS", "300"))
)

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_THREADS = max(0, int(os.getenv("FFMPEG_THREADS", "0")))
FFMPEG_AVAILABLE = shutil.which(FFMPEG_BIN) is not None
FFPROBE_AVAILABLE = shutil.which(FFPROBE_BIN) is not None

ETA_STATS_WINDOW = max(10, int(os.getenv("ETA_STATS_WINDOW", "120")))
ETA_MIN_SAMPLES = max(1, int(os.getenv("ETA_MIN_SAMPLES", "3")))
ETA_CACHE_TTL_SECONDS = max(5, int(os.getenv("ETA_CACHE_TTL_SECONDS", "30")))

WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = max(0, int(os.getenv("WHISPER_CPU_THREADS", "0")))
WHISPER_NUM_WORKERS = max(1, int(os.getenv("WHISPER_NUM_WORKERS", "1")))
WHISPER_BATCH_SIZE = max(1, int(os.getenv("WHISPER_BATCH_SIZE", "1")))

MODE_CONFIG: dict[str, dict[str, Any]] = {
    "fast": {
        "model_size": os.getenv("WHISPER_FAST_MODEL", "tiny"),
        "beam_size": 1,
        "best_of": 1,
        "vad_filter": os.getenv("WHISPER_FAST_VAD_FILTER", "0") == "1",
        "condition_on_previous_text": False,
        "without_timestamps": os.getenv("WHISPER_FAST_WITHOUT_TIMESTAMPS", "1") == "1",
    },
    "accurate": {
        "model_size": os.getenv("WHISPER_ACCURATE_MODEL", "medium"),
        "beam_size": 5,
        "best_of": 5,
        "vad_filter": True,
        "condition_on_previous_text": True,
        "without_timestamps": os.getenv("WHISPER_ACCURATE_WITHOUT_TIMESTAMPS", "1") == "1",
    },
}

JobState = Literal["queued", "processing", "done", "error"]
LanguageOption = Literal["pl", "en", "auto"]
OutputFormat = Literal["txt", "txt_srt"]
logger = logging.getLogger("clip_to_text")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
logger.setLevel(LOG_LEVEL)
logging.getLogger("httpx").setLevel(logging.WARNING)
