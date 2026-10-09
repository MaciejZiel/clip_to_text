import asyncio
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from math import ceil
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None  # type: ignore[assignment]


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


class InvalidFileTypeError(Exception):
    pass


class FileTooLargeError(Exception):
    pass


class NoAudioStreamError(Exception):
    pass


class VideoTooLongError(Exception):
    pass


class TranscriptionRuntimeError(Exception):
    pass


class TranscriptionCancelledError(Exception):
    pass


@dataclass
class UploadMetadata:
    size: int
    content_hash: str
    header: bytes


@dataclass
class CachedTranscript:
    transcript: str
    subtitle_srt: str | None
    detected_language: str | None
    detected_language_probability: float | None
    created_at: float


@dataclass
class TranscriptionJob:
    id: str
    filename: str
    language: LanguageOption
    mode: Literal["fast", "accurate"]
    output_format: OutputFormat
    tmp_dir: Path
    video_path: Path
    audio_path: Path
    created_at: float
    updated_at: float
    state: JobState = "queued"
    stage: str = "queued"
    progress: float = 20.0
    message: str = "File uploaded. Job is queued."
    transcript: str | None = None
    subtitle_srt: str | None = None
    detected_language: str | None = None
    detected_language_probability: float | None = None
    error: str | None = None
    content_hash: str | None = None
    completed_at: float | None = None


class TranscriptionResponse(BaseModel):
    transcript: str
    filename: str
    subtitle_srt: str | None = None
    has_subtitles: bool = False
    detected_language: str | None = None
    detected_language_probability: float | None = None


class JobStartResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    language: LanguageOption
    mode: Literal["fast", "accurate"]
    output_format: OutputFormat
    state: JobState
    stage: str
    progress: float
    message: str
    ready: bool
    has_subtitles: bool
    detected_language: str | None = None
    detected_language_probability: float | None = None
    queue_position: int | None = None
    queue_size: int = 0
    estimated_wait_seconds: float | None = None
    estimated_total_seconds: float | None = None
    created_at: float
    updated_at: float
    error: str | None = None


class JobSummary(BaseModel):
    job_id: str
    filename: str
    language: LanguageOption
    mode: Literal["fast", "accurate"]
    output_format: OutputFormat
    state: JobState
    stage: str
    progress: float
    message: str
    has_subtitles: bool
    detected_language: str | None = None
    detected_language_probability: float | None = None
    queue_position: int | None = None
    queue_size: int = 0
    estimated_wait_seconds: float | None = None
    estimated_total_seconds: float | None = None
    created_at: float
    updated_at: float
    error: str | None = None


class JobsListResponse(BaseModel):
    jobs: list[JobSummary]


app = FastAPI(title="Clip to Text Pro")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

_MODEL_CACHE: dict[tuple[str, str, str, int, int], Any] = {}
_PIPELINE_CACHE: dict[tuple[str, str, str, int, int, int], Any] = {}
_MODEL_LOCK = Lock()

_TRANSCRIPT_CACHE: dict[tuple[str, str, str, str, str, str, str], CachedTranscript] = {}
_TRANSCRIPT_CACHE_LOCK = Lock()

_JOBS: dict[str, TranscriptionJob] = {}
_JOBS_LOCK = Lock()

_JOB_FUTURES: dict[str, Future[Any]] = {}
_JOB_FUTURES_LOCK = Lock()

_CANCELLED_JOBS: set[str] = set()
_CANCELLED_JOBS_LOCK = Lock()

_WORKER_POOL = ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS)
_MAINTENANCE_STOP = Event()
_MAINTENANCE_THREAD: Thread | None = None

_DB_CONNECTION: sqlite3.Connection | None = None
_DB_LOCK = RLock()


@dataclass
class EtaStats:
    average_runtime_seconds: float
    samples: int
    created_at: float


_ETA_CACHE: dict[tuple[str, str, str], EtaStats] = {}
_ETA_CACHE_LOCK = Lock()


def _db_conn() -> sqlite3.Connection:
    global _DB_CONNECTION
    if _DB_CONNECTION is None:
        connection = sqlite3.connect(str(JOBS_DB_PATH), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        _DB_CONNECTION = connection
    return _DB_CONNECTION


def _db_fetchone(query: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
    with _DB_LOCK:
        row = _db_conn().execute(query, params).fetchone()
    return row


def _db_fetchall(query: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
    with _DB_LOCK:
        rows = _db_conn().execute(query, params).fetchall()
    return rows


def _db_execute(query: str, params: tuple[Any, ...] = ()) -> None:
    with _DB_LOCK:
        connection = _db_conn()
        connection.execute(query, params)
        connection.commit()


def _ensure_table_column(table: str, column: str, definition: str) -> None:
    columns = {str(row["name"]) for row in _db_fetchall(f"PRAGMA table_info({table})")}
    if column in columns:
        return
    _db_execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _init_db() -> None:
    _db_execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            language TEXT NOT NULL,
            mode TEXT NOT NULL,
            output_format TEXT NOT NULL DEFAULT 'txt',
            state TEXT NOT NULL,
            stage TEXT NOT NULL,
            progress REAL NOT NULL,
            message TEXT NOT NULL,
            transcript TEXT,
            subtitle_srt TEXT,
            detected_language TEXT,
            detected_language_probability REAL,
            error TEXT,
            content_hash TEXT,
            video_path TEXT,
            audio_path TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL
        )
        """
    )
    _db_execute(
        """
        CREATE TABLE IF NOT EXISTS transcript_cache (
            cache_key TEXT PRIMARY KEY,
            transcript TEXT NOT NULL,
            subtitle_srt TEXT,
            detected_language TEXT,
            detected_language_probability REAL,
            created_at REAL NOT NULL
        )
        """
    )
    _ensure_table_column("jobs", "output_format", "TEXT NOT NULL DEFAULT 'txt'")
    _ensure_table_column("jobs", "subtitle_srt", "TEXT")
    _ensure_table_column("jobs", "detected_language", "TEXT")
    _ensure_table_column("jobs", "detected_language_probability", "REAL")
    _ensure_table_column("transcript_cache", "subtitle_srt", "TEXT")
    _ensure_table_column("transcript_cache", "detected_language", "TEXT")
    _ensure_table_column("transcript_cache", "detected_language_probability", "REAL")
    _db_execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)")
    _db_execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state)")
    _db_execute("CREATE INDEX IF NOT EXISTS idx_jobs_completed_at ON jobs(completed_at)")
    _db_execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_profile ON jobs(mode, output_format, language, state, completed_at DESC)"
    )


def _close_db() -> None:
    global _DB_CONNECTION
    with _DB_LOCK:
        if _DB_CONNECTION is not None:
            _DB_CONNECTION.close()
            _DB_CONNECTION = None


def _safe_filename_stem(filename: str) -> str:
    stem = Path(filename).stem or "transcription"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "transcription"


def _validate_extension(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise InvalidFileTypeError("Supported formats: .mp4, .mov, .mkv.")
    return suffix


def _validate_content_type(upload: UploadFile) -> None:
    content_type = (upload.content_type or "").strip().lower()
    if not content_type:
        return
    if content_type in ALLOWED_CONTENT_TYPES or content_type.startswith("video/"):
        return
    raise InvalidFileTypeError(f"Unsupported MIME type: {content_type}")


def _validate_magic_header(header: bytes, suffix: str) -> None:
    if len(header) < 12:
        raise InvalidFileTypeError("The file is corrupted or has an invalid format.")

    if suffix in {".mp4", ".mov"}:
        if b"ftyp" not in header[4:16]:
            raise InvalidFileTypeError("The file does not look like a valid MP4/MOV container.")
        return

    if suffix == ".mkv":
        if not header.startswith(MKV_SIGNATURE):
            raise InvalidFileTypeError("The file does not look like a valid MKV container.")
        return


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise TranscriptionRuntimeError(
            f"Required tool not found in PATH: {command[0]}. Install ffmpeg and restart the app."
        ) from exc


def _save_upload(upload: UploadFile, destination: Path, suffix: str) -> UploadMetadata:
    total_size = 0
    digest = sha256()
    header = b""

    with destination.open("wb") as output_file:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break

            if len(header) < 32:
                need = 32 - len(header)
                header += chunk[:need]

            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                raise FileTooLargeError(
                    f"The file is too large. Maximum size is {MAX_UPLOAD_MB} MB."
                )

            output_file.write(chunk)
            digest.update(chunk)

    _validate_magic_header(header, suffix)
    return UploadMetadata(size=total_size, content_hash=digest.hexdigest(), header=header)


def _probe_duration_seconds(video_path: Path) -> float | None:
    if MAX_VIDEO_DURATION_SECONDS <= 0:
        return None

    command = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-nostdin",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    probe = _run_command(command)
    if probe.returncode != 0:
        stderr = probe.stderr.strip() or "Failed to read video duration."
        raise TranscriptionRuntimeError(stderr)

    output = (probe.stdout or "").strip()
    if not output:
        return None

    try:
        duration = float(output)
    except ValueError:
        return None

    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise VideoTooLongError(
            f"The recording is too long ({int(duration)} s). Limit: {MAX_VIDEO_DURATION_SECONDS} s."
        )
    return duration


def _extract_audio_to_wav(video_path: Path, audio_path: Path) -> None:
    command = [FFMPEG_BIN, "-v", "error", "-nostdin", "-y"]
    if FFMPEG_THREADS > 0:
        command.extend(["-threads", str(FFMPEG_THREADS)])
    command.extend(
        [
            "-i",
            str(video_path),
            "-map",
            "a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(audio_path),
        ]
    )
    ffmpeg = _run_command(command)
    if ffmpeg.returncode != 0:
        stderr = ffmpeg.stderr.strip() or "Failed to extract audio from the video."
        normalized = stderr.lower()
        no_audio_patterns = ("stream map", "matches no streams", "does not contain any stream")
        if any(pattern in normalized for pattern in no_audio_patterns):
            raise NoAudioStreamError("No audio track found in the file.")
        raise TranscriptionRuntimeError(stderr)


def _model_cache_key(mode: Literal["fast", "accurate"]) -> tuple[str, str, str, int, int]:
    config = MODE_CONFIG[mode]
    return (
        str(config["model_size"]),
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
        WHISPER_CPU_THREADS,
        WHISPER_NUM_WORKERS,
    )


def _get_whisper_model(mode: Literal["fast", "accurate"]) -> Any:
    if WhisperModel is None:
        raise TranscriptionRuntimeError(
            "Missing faster-whisper package. Install dependencies from requirements.txt."
        )

    cache_key = _model_cache_key(mode)
    model_size = cache_key[0]

    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is None:
            model = WhisperModel(
                model_size_or_path=model_size,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
                cpu_threads=WHISPER_CPU_THREADS,
                num_workers=WHISPER_NUM_WORKERS,
            )
            _MODEL_CACHE[cache_key] = model
    return model


def _get_batched_pipeline(mode: Literal["fast", "accurate"], model: Any) -> Any | None:
    if BatchedInferencePipeline is None or WHISPER_BATCH_SIZE <= 1:
        return None

    cache_key = _model_cache_key(mode) + (WHISPER_BATCH_SIZE,)
    with _MODEL_LOCK:
        pipeline = _PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            pipeline = BatchedInferencePipeline(model=model)
            _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def _is_job_cancelled(job_id: str) -> bool:
    with _CANCELLED_JOBS_LOCK:
        return job_id in _CANCELLED_JOBS


def _mark_job_cancelled(job_id: str) -> None:
    with _CANCELLED_JOBS_LOCK:
        _CANCELLED_JOBS.add(job_id)


def _clear_job_cancelled(job_id: str) -> None:
    with _CANCELLED_JOBS_LOCK:
        _CANCELLED_JOBS.discard(job_id)


def _raise_if_cancelled(job_id: str) -> None:
    if _is_job_cancelled(job_id):
        raise TranscriptionCancelledError("Job has been cancelled.")


def _transcribe_audio(
    audio_path: Path,
    language: LanguageOption,
    mode: Literal["fast", "accurate"],
    output_format: OutputFormat,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> tuple[str, str | None, str | None, float | None]:
    if cancel_callback is not None:
        cancel_callback()

    if progress_callback is not None:
        progress_callback(45.0, "Loading model and starting transcription...")

    model = _get_whisper_model(mode)
    config = MODE_CONFIG[mode]

    transcribe_kwargs = {
        "language": None if language == "auto" else language,
        "beam_size": int(config["beam_size"]),
        "best_of": int(config["best_of"]),
        "vad_filter": bool(config["vad_filter"]),
        "condition_on_previous_text": bool(config["condition_on_previous_text"]),
        # Timings are required for SRT generation, otherwise keep fast profile.
        "without_timestamps": bool(config["without_timestamps"]) if output_format == "txt" else False,
        "temperature": 0.0,
    }

    pipeline = _get_batched_pipeline(mode, model)
    if pipeline is None:
        segments, info = model.transcribe(str(audio_path), **transcribe_kwargs)
    else:
        segments, info = pipeline.transcribe(
            str(audio_path),
            batch_size=WHISPER_BATCH_SIZE,
            **transcribe_kwargs,
        )

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    pieces: list[str] = []
    subtitle_segments: list[tuple[float, float, str]] = []
    last_progress = 45.0

    for segment in segments:
        if cancel_callback is not None:
            cancel_callback()

        text = (getattr(segment, "text", "") or "").strip()
        if text:
            pieces.append(text)
            if output_format == "txt_srt":
                segment_start = getattr(segment, "start", None)
                segment_end = getattr(segment, "end", None)
                if segment_start is not None and segment_end is not None:
                    start_value = max(0.0, float(segment_start))
                    end_value = max(start_value, float(segment_end))
                    subtitle_segments.append((start_value, end_value, text))

        if progress_callback is None:
            continue

        segment_end = getattr(segment, "end", None)
        if duration > 0.0 and segment_end is not None:
            ratio = max(0.0, min(float(segment_end) / duration, 1.0))
            progress = 45.0 + ratio * 50.0
        else:
            progress = min(95.0, last_progress + max(0.8, (95.0 - last_progress) * 0.18))

        if progress - last_progress >= 1.0:
            progress_callback(progress, "Transcribing audio...")
            last_progress = progress

    if progress_callback is not None:
        progress_callback(98.0, "Finalizing output...")

    transcript = _normalize_transcript(" ".join(pieces))
    if not transcript:
        raise TranscriptionRuntimeError(
            "Transcription returned an empty result. Check audio quality in the file."
        )

    detected_language_raw = getattr(info, "language", None)
    detected_language = str(detected_language_raw) if detected_language_raw else None
    detected_probability_raw = getattr(info, "language_probability", None)
    detected_probability: float | None = None
    if detected_probability_raw is not None:
        try:
            detected_probability = max(0.0, min(1.0, float(detected_probability_raw)))
        except (TypeError, ValueError):
            detected_probability = None

    subtitle_srt = _build_srt(subtitle_segments) if output_format == "txt_srt" else None
    return transcript, subtitle_srt, detected_language, detected_probability


def _normalize_transcript(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _build_srt(segments: list[tuple[float, float, str]]) -> str | None:
    if not segments:
        return None

    lines: list[str] = []
    for index, (start, end, text) in enumerate(segments, start=1):
        clean_text = _normalize_transcript(text)
        if not clean_text:
            continue
        if end <= start:
            end = start + 0.4

        lines.extend(
            [
                str(index),
                f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}",
                clean_text,
                "",
            ]
        )

    subtitle = "\n".join(lines).strip()
    return subtitle or None


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return


def _remove_job_files(job: TranscriptionJob) -> None:
    for path in (job.video_path, job.audio_path):
        if path and str(path) not in {"", "."}:
            _remove_file(path)

    if job.tmp_dir and str(job.tmp_dir) not in {"", "."}:
        try:
            job.tmp_dir.rmdir()
        except OSError:
            pass


def _remove_temp_dir(temp_dir: Path) -> None:
    for path in temp_dir.glob("*"):
        _remove_file(path)
    try:
        temp_dir.rmdir()
    except OSError:
        pass


def _row_to_job(row: sqlite3.Row) -> TranscriptionJob:
    video_path = Path(row["video_path"]) if row["video_path"] else Path("")
    audio_path = Path(row["audio_path"]) if row["audio_path"] else Path("")
    tmp_dir = video_path.parent if row["video_path"] else Path("")
    output_format: OutputFormat = "txt_srt" if row["output_format"] == "txt_srt" else "txt"
    language: LanguageOption = "auto" if row["language"] == "auto" else ("en" if row["language"] == "en" else "pl")

    return TranscriptionJob(
        id=row["id"],
        filename=row["filename"],
        language=language,
        mode=row["mode"],
        output_format=output_format,
        tmp_dir=tmp_dir,
        video_path=video_path,
        audio_path=audio_path,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        state=row["state"],
        stage=row["stage"],
        progress=float(row["progress"]),
        message=row["message"],
        transcript=row["transcript"],
        subtitle_srt=row["subtitle_srt"],
        detected_language=row["detected_language"],
        detected_language_probability=row["detected_language_probability"],
        error=row["error"],
        content_hash=row["content_hash"],
        completed_at=row["completed_at"],
    )


def _upsert_job(job: TranscriptionJob) -> None:
    _db_execute(
        """
        INSERT INTO jobs (
            id, filename, language, mode, output_format, state, stage, progress, message,
            transcript, subtitle_srt, detected_language, detected_language_probability, error, content_hash, video_path, audio_path,
            created_at, updated_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            filename=excluded.filename,
            language=excluded.language,
            mode=excluded.mode,
            output_format=excluded.output_format,
            state=excluded.state,
            stage=excluded.stage,
            progress=excluded.progress,
            message=excluded.message,
            transcript=excluded.transcript,
            subtitle_srt=excluded.subtitle_srt,
            detected_language=excluded.detected_language,
            detected_language_probability=excluded.detected_language_probability,
            error=excluded.error,
            content_hash=excluded.content_hash,
            video_path=excluded.video_path,
            audio_path=excluded.audio_path,
            updated_at=excluded.updated_at,
            completed_at=excluded.completed_at
        """,
        (
            job.id,
            job.filename,
            job.language,
            job.mode,
            job.output_format,
            job.state,
            job.stage,
            job.progress,
            job.message,
            job.transcript,
            job.subtitle_srt,
            job.detected_language,
            job.detected_language_probability,
            job.error,
            job.content_hash,
            str(job.video_path) if str(job.video_path) else None,
            str(job.audio_path) if str(job.audio_path) else None,
            job.created_at,
            job.updated_at,
            job.completed_at,
        ),
    )


def _get_job_from_db(job_id: str) -> TranscriptionJob | None:
    row = _db_fetchone("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if row is None:
        return None
    return _row_to_job(row)


def _update_job(
    job_id: str,
    *,
    state: JobState | None = None,
    stage: str | None = None,
    progress: float | None = None,
    message: str | None = None,
    transcript: str | None = None,
    subtitle_srt: str | None = None,
    detected_language: str | None = None,
    detected_language_probability: float | None = None,
    error: str | None = None,
) -> None:
    now = time.time()

    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    if job is None:
        job = _get_job_from_db(job_id)
        if job is None:
            return

    if state is not None:
        job.state = state
    if stage is not None:
        job.stage = stage
    if progress is not None:
        job.progress = max(0.0, min(100.0, float(progress)))
    if message is not None:
        job.message = message
    if transcript is not None:
        job.transcript = transcript
    if subtitle_srt is not None:
        job.subtitle_srt = subtitle_srt
    if detected_language is not None:
        job.detected_language = detected_language
    if detected_language_probability is not None:
        job.detected_language_probability = detected_language_probability
    job.error = error
    job.updated_at = now

    if job.state in {"done", "error"} and job.completed_at is None:
        job.completed_at = now

    with _JOBS_LOCK:
        _JOBS[job_id] = job

    _upsert_job(job)

    if job.state == "done":
        with _ETA_CACHE_LOCK:
            _ETA_CACHE.clear()


def _get_job_snapshot(job_id: str) -> TranscriptionJob | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            return replace(job)
    return _get_job_from_db(job_id)


def _require_job_snapshot(job_id: str) -> TranscriptionJob:
    job = _get_job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


def _queue_stats(job_id: str) -> tuple[int | None, int]:
    with _JOBS_LOCK:
        queued_jobs = sorted(
            (job for job in _JOBS.values() if job.state == "queued"),
            key=lambda job: job.created_at,
        )

    queue_size = len(queued_jobs)
    for index, job in enumerate(queued_jobs, start=1):
        if job.id == job_id:
            return index, queue_size
    return None, queue_size


def _active_jobs_count() -> int:
    with _JOBS_LOCK:
        return sum(1 for job in _JOBS.values() if job.state in {"queued", "processing"})


def _estimate_runtime_for_profile(mode: str, output_format: str, language: str | None) -> EtaStats | None:
    normalized_language = language if language is not None else "*"
    cache_key = (mode, output_format, normalized_language)
    now = time.time()

    with _ETA_CACHE_LOCK:
        cached = _ETA_CACHE.get(cache_key)
        if cached is not None and now - cached.created_at <= ETA_CACHE_TTL_SECONDS:
            return cached

    if language is None:
        row = _db_fetchone(
            """
            SELECT AVG(completed_at - created_at) AS avg_runtime, COUNT(1) AS samples
            FROM (
                SELECT created_at, completed_at
                FROM jobs
                WHERE state = 'done'
                  AND completed_at IS NOT NULL
                  AND created_at IS NOT NULL
                  AND mode = ?
                  AND output_format = ?
                ORDER BY completed_at DESC
                LIMIT ?
            )
            """,
            (mode, output_format, ETA_STATS_WINDOW),
        )
    else:
        row = _db_fetchone(
            """
            SELECT AVG(completed_at - created_at) AS avg_runtime, COUNT(1) AS samples
            FROM (
                SELECT created_at, completed_at
                FROM jobs
                WHERE state = 'done'
                  AND completed_at IS NOT NULL
                  AND created_at IS NOT NULL
                  AND mode = ?
                  AND output_format = ?
                  AND language = ?
                ORDER BY completed_at DESC
                LIMIT ?
            )
            """,
            (mode, output_format, language, ETA_STATS_WINDOW),
        )
    if row is None or row["avg_runtime"] is None:
        return None

    samples = int(row["samples"] or 0)
    if samples < ETA_MIN_SAMPLES:
        return None

    avg_runtime = max(1.0, float(row["avg_runtime"]))
    stats = EtaStats(average_runtime_seconds=avg_runtime, samples=samples, created_at=now)
    with _ETA_CACHE_LOCK:
        _ETA_CACHE[cache_key] = stats
    return stats


def _get_eta_stats(mode: str, output_format: str, language: LanguageOption) -> EtaStats | None:
    preferred = _estimate_runtime_for_profile(mode, output_format, language)
    if preferred is not None:
        return preferred
    return _estimate_runtime_for_profile(mode, output_format, None)


def _estimate_eta_seconds(
    job: TranscriptionJob,
    queue_position: int | None,
) -> tuple[float | None, float | None]:
    stats = _get_eta_stats(job.mode, job.output_format, job.language)
    if stats is None:
        return None, None

    per_job_seconds = stats.average_runtime_seconds
    wait_seconds = 0.0

    if job.state == "queued" and queue_position is not None and queue_position > 1:
        jobs_ahead = queue_position - 1
        waves = ceil(jobs_ahead / max(1, TRANSCRIBE_WORKERS))
        wait_seconds = max(0.0, waves * per_job_seconds)

    if job.state == "queued":
        total_seconds = wait_seconds + per_job_seconds
        return round(wait_seconds, 1), round(total_seconds, 1)

    if job.state == "processing":
        remaining = max(0.0, (100.0 - job.progress) / 100.0 * per_job_seconds)
        return 0.0, round(remaining, 1)

    return 0.0, 0.0


def _build_job_status(job: TranscriptionJob) -> JobStatusResponse:
    queue_position, queue_size = _queue_stats(job.id)
    estimated_wait_seconds, estimated_total_seconds = _estimate_eta_seconds(job, queue_position)
    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        language=job.language,
        mode=job.mode,
        output_format=job.output_format,
        state=job.state,
        stage=job.stage,
        progress=round(job.progress, 1),
        message=job.message,
        ready=job.state == "done",
        has_subtitles=bool(job.subtitle_srt),
        detected_language=job.detected_language,
        detected_language_probability=job.detected_language_probability,
        queue_position=queue_position,
        queue_size=queue_size,
        estimated_wait_seconds=estimated_wait_seconds,
        estimated_total_seconds=estimated_total_seconds,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
    )


def _build_job_summary(job: TranscriptionJob) -> JobSummary:
    queue_position, queue_size = _queue_stats(job.id)
    estimated_wait_seconds, estimated_total_seconds = _estimate_eta_seconds(job, queue_position)
    return JobSummary(
        job_id=job.id,
        filename=job.filename,
        language=job.language,
        mode=job.mode,
        output_format=job.output_format,
        state=job.state,
        stage=job.stage,
        progress=round(job.progress, 1),
        message=job.message,
        has_subtitles=bool(job.subtitle_srt),
        detected_language=job.detected_language,
        detected_language_probability=job.detected_language_probability,
        queue_position=queue_position,
        queue_size=queue_size,
        estimated_wait_seconds=estimated_wait_seconds,
        estimated_total_seconds=estimated_total_seconds,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
    )


def _transcript_cache_key(
    content_hash: str,
    language: LanguageOption,
    mode: Literal["fast", "accurate"],
    output_format: OutputFormat,
) -> tuple[str, str, str, str, str, str, str]:
    config = MODE_CONFIG[mode]
    return (
        content_hash,
        language,
        mode,
        output_format,
        str(config["model_size"]),
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
    )


def _cleanup_transcript_cache() -> None:
    now = time.time()
    threshold = now - TRANSCRIPT_CACHE_TTL_SECONDS if TRANSCRIPT_CACHE_TTL_SECONDS > 0 else now

    with _TRANSCRIPT_CACHE_LOCK:
        if TRANSCRIPT_CACHE_TTL_SECONDS <= 0:
            _TRANSCRIPT_CACHE.clear()
        else:
            stale = [
                key
                for key, value in _TRANSCRIPT_CACHE.items()
                if value.created_at < threshold
            ]
            for key in stale:
                _TRANSCRIPT_CACHE.pop(key, None)

            if len(_TRANSCRIPT_CACHE) > TRANSCRIPT_CACHE_MAX_ITEMS:
                sorted_items = sorted(_TRANSCRIPT_CACHE.items(), key=lambda item: item[1].created_at)
                overflow = len(_TRANSCRIPT_CACHE) - TRANSCRIPT_CACHE_MAX_ITEMS
                for index in range(overflow):
                    _TRANSCRIPT_CACHE.pop(sorted_items[index][0], None)

    if TRANSCRIPT_CACHE_TTL_SECONDS <= 0:
        _db_execute("DELETE FROM transcript_cache")
        return

    _db_execute("DELETE FROM transcript_cache WHERE created_at < ?", (threshold,))

    rows = _db_fetchall(
        "SELECT cache_key FROM transcript_cache ORDER BY created_at DESC LIMIT -1 OFFSET ?",
        (TRANSCRIPT_CACHE_MAX_ITEMS,),
    )
    for row in rows:
        _db_execute("DELETE FROM transcript_cache WHERE cache_key = ?", (row["cache_key"],))


def _get_cached_transcript(
    content_hash: str,
    language: LanguageOption,
    mode: Literal["fast", "accurate"],
    output_format: OutputFormat,
) -> CachedTranscript | None:
    if TRANSCRIPT_CACHE_TTL_SECONDS <= 0:
        return None

    cache_key = _transcript_cache_key(content_hash, language, mode, output_format)

    with _TRANSCRIPT_CACHE_LOCK:
        cached = _TRANSCRIPT_CACHE.get(cache_key)
        if cached is not None and time.time() - cached.created_at <= TRANSCRIPT_CACHE_TTL_SECONDS:
            return cached

    row = _db_fetchone(
        """
        SELECT transcript, subtitle_srt, detected_language, detected_language_probability, created_at
        FROM transcript_cache
        WHERE cache_key = ?
        """,
        (json.dumps(cache_key),),
    )
    if row is None:
        return None

    if time.time() - float(row["created_at"]) > TRANSCRIPT_CACHE_TTL_SECONDS:
        _db_execute("DELETE FROM transcript_cache WHERE cache_key = ?", (json.dumps(cache_key),))
        return None

    cached = CachedTranscript(
        transcript=str(row["transcript"]),
        subtitle_srt=row["subtitle_srt"],
        detected_language=row["detected_language"],
        detected_language_probability=row["detected_language_probability"],
        created_at=float(row["created_at"]),
    )
    with _TRANSCRIPT_CACHE_LOCK:
        _TRANSCRIPT_CACHE[cache_key] = cached
    return cached


def _set_cached_transcript(
    content_hash: str | None,
    language: LanguageOption,
    mode: Literal["fast", "accurate"],
    output_format: OutputFormat,
    transcript: str,
    subtitle_srt: str | None,
    detected_language: str | None,
    detected_language_probability: float | None,
) -> None:
    if not content_hash or TRANSCRIPT_CACHE_TTL_SECONDS <= 0:
        return

    cache_key = _transcript_cache_key(content_hash, language, mode, output_format)
    now = time.time()

    with _TRANSCRIPT_CACHE_LOCK:
        _TRANSCRIPT_CACHE[cache_key] = CachedTranscript(
            transcript=transcript,
            subtitle_srt=subtitle_srt,
            detected_language=detected_language,
            detected_language_probability=detected_language_probability,
            created_at=now,
        )

    serialized_key = json.dumps(cache_key)
    _db_execute(
        """
        INSERT INTO transcript_cache (
            cache_key, transcript, subtitle_srt, detected_language, detected_language_probability, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            transcript=excluded.transcript,
            subtitle_srt=excluded.subtitle_srt,
            detected_language=excluded.detected_language,
            detected_language_probability=excluded.detected_language_probability,
            created_at=excluded.created_at
        """,
        (
            serialized_key,
            transcript,
            subtitle_srt,
            detected_language,
            detected_language_probability,
            now,
        ),
    )


def _preload_model(mode: Literal["fast", "accurate"]) -> None:
    try:
        _get_whisper_model(mode)
    except Exception:
        return


def _cleanup_memory_jobs() -> None:
    now = time.time()
    expired: list[TranscriptionJob] = []

    with _JOBS_LOCK:
        for job_id, job in list(_JOBS.items()):
            if job.state not in {"done", "error"}:
                continue
            if now - job.updated_at <= JOB_MEMORY_TTL_SECONDS:
                continue
            expired.append(_JOBS.pop(job_id))

    for job in expired:
        _remove_job_files(job)


def _cleanup_persisted_jobs() -> None:
    if JOB_RETENTION_SECONDS <= 0:
        return

    cutoff = time.time() - JOB_RETENTION_SECONDS
    rows = _db_fetchall(
        """
        SELECT video_path, audio_path
        FROM jobs
        WHERE completed_at IS NOT NULL AND completed_at < ?
        """,
        (cutoff,),
    )

    _db_execute(
        "DELETE FROM jobs WHERE completed_at IS NOT NULL AND completed_at < ?",
        (cutoff,),
    )

    for row in rows:
        video_path = Path(row["video_path"]) if row["video_path"] else None
        audio_path = Path(row["audio_path"]) if row["audio_path"] else None

        if video_path is not None:
            _remove_file(video_path)
            try:
                video_path.parent.rmdir()
            except OSError:
                pass
        if audio_path is not None:
            _remove_file(audio_path)


def _cleanup_expired_jobs() -> None:
    _cleanup_memory_jobs()
    _cleanup_persisted_jobs()
    _cleanup_transcript_cache()


def _maintenance_loop() -> None:
    while not _MAINTENANCE_STOP.wait(MAINTENANCE_INTERVAL_SECONDS):
        try:
            _cleanup_expired_jobs()
        except Exception:
            logger.exception("Background maintenance loop failed.")


def _mark_stale_jobs_after_restart() -> None:
    rows = _db_fetchall(
        "SELECT * FROM jobs WHERE state IN ('queued', 'processing')"
    )
    now = time.time()

    for row in rows:
        job = _row_to_job(row)
        job.state = "error"
        job.stage = "interrupted"
        job.progress = max(job.progress, 100.0)
        job.message = "Job interrupted by app restart."
        job.error = "Interrupted during server restart."
        job.updated_at = now
        job.completed_at = now
        _upsert_job(job)
        _remove_job_files(job)


def _run_transcription_job(job_id: str) -> None:
    job = _get_job_snapshot(job_id)
    if job is None:
        return

    logger.info("Start job %s (%s, %s)", job_id, job.language, job.mode)

    try:
        _raise_if_cancelled(job_id)

        _update_job(
            job_id,
            state="processing",
            stage="validating",
            progress=25.0,
            message="Preparing transcription...",
            error=None,
        )

        _raise_if_cancelled(job_id)
        _update_job(
            job_id,
            state="processing",
            stage="extracting_audio",
            progress=35.0,
            message="Extracting audio from video...",
            error=None,
        )
        _extract_audio_to_wav(job.video_path, job.audio_path)

        def on_progress(progress: float, message: str) -> None:
            _update_job(
                job_id,
                state="processing",
                stage="transcribing",
                progress=progress,
                message=message,
                error=None,
            )

        transcript, subtitle_srt, detected_language, detected_probability = _transcribe_audio(
            job.audio_path,
            job.language,
            job.mode,
            job.output_format,
            progress_callback=on_progress,
            cancel_callback=lambda: _raise_if_cancelled(job_id),
        )

        _set_cached_transcript(
            job.content_hash,
            job.language,
            job.mode,
            job.output_format,
            transcript,
            subtitle_srt,
            detected_language,
            detected_probability,
        )
        _update_job(
            job_id,
            state="done",
            stage="done",
            progress=100.0,
            message="Done.",
            transcript=transcript,
            subtitle_srt=subtitle_srt,
            detected_language=detected_language,
            detected_language_probability=detected_probability,
            error=None,
        )
        logger.info("Job %s finished successfully", job_id)
    except TranscriptionCancelledError as exc:
        _update_job(
            job_id,
            state="error",
            stage="cancelled",
            progress=100.0,
            message="Job cancelled.",
            error=str(exc),
        )
        logger.info("Job %s cancelled", job_id)
    except (
        InvalidFileTypeError,
        FileTooLargeError,
        VideoTooLongError,
        NoAudioStreamError,
        TranscriptionRuntimeError,
    ) as exc:
        _update_job(
            job_id,
            state="error",
            stage="error",
            progress=100.0,
            message="Transcription failed.",
            error=str(exc),
        )
        logger.warning("Job %s failed: %s", job_id, exc)
    except Exception:
        _update_job(
            job_id,
            state="error",
            stage="error",
            progress=100.0,
            message="An unexpected error occurred during transcription.",
            error="Unexpected processing error.",
        )
        logger.exception("Job %s failed with unexpected error", job_id)
    finally:
        finished = _get_job_snapshot(job_id)
        if finished is not None:
            _remove_job_files(finished)

        _clear_job_cancelled(job_id)
        with _JOB_FUTURES_LOCK:
            _JOB_FUTURES.pop(job_id, None)


def _process_upload(
    upload: UploadFile,
    language: LanguageOption,
    mode: Literal["fast", "accurate"],
    output_format: OutputFormat,
) -> tuple[str, str | None, str | None, float | None]:
    if not upload.filename:
        raise InvalidFileTypeError("Missing filename.")

    _validate_content_type(upload)
    suffix = _validate_extension(upload.filename)

    with tempfile.TemporaryDirectory(prefix="clip_to_text_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        video_path = tmp_path / f"input{suffix}"
        audio_path = tmp_path / "audio.wav"

        metadata = _save_upload(upload, video_path, suffix)
        _probe_duration_seconds(video_path)
        cached_result = _get_cached_transcript(metadata.content_hash, language, mode, output_format)
        if cached_result is not None:
            return (
                cached_result.transcript,
                cached_result.subtitle_srt,
                cached_result.detected_language,
                cached_result.detected_language_probability,
            )

        _extract_audio_to_wav(video_path, audio_path)

        transcript, subtitle_srt, detected_language, detected_probability = _transcribe_audio(
            audio_path,
            language,
            mode,
            output_format,
        )
        _set_cached_transcript(
            metadata.content_hash,
            language,
            mode,
            output_format,
            transcript,
            subtitle_srt,
            detected_language,
            detected_probability,
        )
        return transcript, subtitle_srt, detected_language, detected_probability


def _serialize_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[return-value]
    return model.dict()  # type: ignore[return-value]


@app.on_event("startup")
def startup_warmup() -> None:
    global _MAINTENANCE_THREAD

    _init_db()
    _mark_stale_jobs_after_restart()
    _cleanup_expired_jobs()

    if not FFMPEG_AVAILABLE:
        logger.warning("ffmpeg not found in PATH (%s).", FFMPEG_BIN)
    if MAX_VIDEO_DURATION_SECONDS > 0 and not FFPROBE_AVAILABLE:
        logger.warning("ffprobe not found in PATH (%s).", FFPROBE_BIN)

    _MAINTENANCE_STOP.clear()
    if _MAINTENANCE_THREAD is None or not _MAINTENANCE_THREAD.is_alive():
        _MAINTENANCE_THREAD = Thread(target=_maintenance_loop, daemon=True)
        _MAINTENANCE_THREAD.start()

    if PRELOAD_FAST_MODEL:
        Thread(target=_preload_model, args=("fast",), daemon=True).start()
    if PRELOAD_ACCURATE_MODEL:
        Thread(target=_preload_model, args=("accurate",), daemon=True).start()


@app.on_event("shutdown")
def shutdown_workers() -> None:
    _MAINTENANCE_STOP.set()
    _WORKER_POOL.shutdown(wait=False, cancel_futures=True)
    _close_db()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"max_upload_mb": MAX_UPLOAD_MB},
    )


@app.get("/api/jobs", response_model=JobsListResponse)
async def list_jobs(limit: int = Query(12, ge=1, le=100)) -> JobsListResponse:
    _cleanup_expired_jobs()

    rows = _db_fetchall(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    jobs = [_build_job_summary(_row_to_job(row)) for row in rows]
    return JobsListResponse(jobs=jobs)


@app.post("/api/jobs", response_model=JobStartResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    file: UploadFile = File(...),
    language: LanguageOption = Form("pl"),
    mode: Literal["fast", "accurate"] = Form("fast"),
    output_format: OutputFormat = Form("txt"),
) -> JobStartResponse:
    _cleanup_expired_jobs()

    if _active_jobs_count() >= MAX_PENDING_JOBS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                "Queue is full. Try again in a moment "
                f"(max {MAX_PENDING_JOBS} active jobs)."
            ),
        )

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename.")

    try:
        _validate_content_type(file)
        suffix = _validate_extension(file.filename)
    except InvalidFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    safe_name = _safe_filename_stem(file.filename)

    job_id = uuid.uuid4().hex
    temp_dir = JOBS_DIR / job_id
    video_path = temp_dir / f"input{suffix}"
    audio_path = temp_dir / "audio.wav"
    temp_dir.mkdir(parents=True, exist_ok=False)

    metadata: UploadMetadata | None = None
    try:
        metadata = await run_in_threadpool(_save_upload, file, video_path, suffix)
    except (FileTooLargeError, InvalidFileTypeError) as exc:
        _remove_temp_dir(temp_dir)
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if isinstance(exc, FileTooLargeError)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        _remove_temp_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from exc
    finally:
        await file.close()

    assert metadata is not None
    now = time.time()

    try:
        await run_in_threadpool(_probe_duration_seconds, video_path)
    except VideoTooLongError as exc:
        _remove_temp_dir(temp_dir)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except TranscriptionRuntimeError as exc:
        _remove_temp_dir(temp_dir)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    cached_result = _get_cached_transcript(metadata.content_hash, language, mode, output_format)
    if cached_result is not None:
        _remove_temp_dir(temp_dir)
        cached_job = TranscriptionJob(
            id=job_id,
            filename=safe_name,
            language=language,
            mode=mode,
            output_format=output_format,
            tmp_dir=temp_dir,
            video_path=video_path,
            audio_path=audio_path,
            created_at=now,
            updated_at=now,
            state="done",
            stage="done",
            progress=100.0,
            message="Done (cache hit).",
            transcript=cached_result.transcript,
            subtitle_srt=cached_result.subtitle_srt,
            detected_language=cached_result.detected_language,
            detected_language_probability=cached_result.detected_language_probability,
            error=None,
            content_hash=metadata.content_hash,
            completed_at=now,
        )
        with _JOBS_LOCK:
            _JOBS[job_id] = cached_job
        _upsert_job(cached_job)
        return JobStartResponse(job_id=job_id)

    job = TranscriptionJob(
        id=job_id,
        filename=safe_name,
        language=language,
        mode=mode,
        output_format=output_format,
        tmp_dir=temp_dir,
        video_path=video_path,
        audio_path=audio_path,
        created_at=now,
        updated_at=now,
        content_hash=metadata.content_hash,
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = job
    _upsert_job(job)
    logger.info("Queued job %s (%s, %s, %s)", job_id, language, mode, output_format)

    future = _WORKER_POOL.submit(_run_transcription_job, job_id)
    with _JOB_FUTURES_LOCK:
        _JOB_FUTURES[job_id] = future

    return JobStartResponse(job_id=job_id)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    _cleanup_expired_jobs()
    job = _require_job_snapshot(job_id)
    return _build_job_status(job)


@app.post("/api/jobs/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: str) -> JobStatusResponse:
    _cleanup_expired_jobs()
    job = _require_job_snapshot(job_id)

    if job.state in {"done", "error"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Job is already finished.",
        )

    _mark_job_cancelled(job_id)

    cancelled_before_start = False
    with _JOB_FUTURES_LOCK:
        future = _JOB_FUTURES.get(job_id)
        if future is not None and future.cancel():
            cancelled_before_start = True
            _JOB_FUTURES.pop(job_id, None)

    if cancelled_before_start:
        _update_job(
            job_id,
            state="error",
            stage="cancelled",
            progress=100.0,
            message="Job cancelled.",
            error="Job was cancelled before it started.",
        )
        finished = _get_job_snapshot(job_id)
        if finished is not None:
            _remove_job_files(finished)
            with _JOBS_LOCK:
                _JOBS[job_id] = finished
        _clear_job_cancelled(job_id)
    else:
        _update_job(
            job_id,
            state="processing",
            stage="cancelling",
            message="Attempting to cancel job...",
            error=None,
        )

    return _build_job_status(_require_job_snapshot(job_id))


@app.get("/api/jobs/{job_id}/events")
async def stream_job_events(job_id: str) -> StreamingResponse:
    async def event_generator() -> Any:
        last_payload = ""
        started_at = time.time()
        last_heartbeat = started_at

        yield "retry: 3000\n\n"

        while True:
            job = _get_job_snapshot(job_id)
            if job is None:
                payload = {
                    "job_id": job_id,
                    "state": "error",
                    "stage": "error",
                    "progress": 100,
                    "message": "Job not found.",
                    "ready": False,
                    "error": "Job not found.",
                }
                yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
                break

            status_payload = _serialize_model(_build_job_status(job))
            serialized = json.dumps(status_payload, ensure_ascii=False)
            if serialized != last_payload:
                yield f"data: {serialized}\n\n"
                last_payload = serialized
                last_heartbeat = time.time()

            if job.state in {"done", "error"}:
                break

            now = time.time()
            if now - last_heartbeat >= SSE_HEARTBEAT_SECONDS:
                yield ": ping\n\n"
                last_heartbeat = now

            if time.time() - started_at > SSE_MAX_SECONDS:
                timeout_payload = {
                    "job_id": job.id,
                    "state": job.state,
                    "stage": job.stage,
                    "progress": job.progress,
                    "message": "SSE connection timed out. Refresh status manually.",
                    "ready": False,
                    "error": None,
                }
                yield (
                    "event: timeout\n"
                    f"data: {json.dumps(timeout_payload, ensure_ascii=False)}\n\n"
                )
                break

            await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/jobs/{job_id}/result", response_model=TranscriptionResponse)
async def get_job_result(job_id: str) -> TranscriptionResponse:
    _cleanup_expired_jobs()
    job = _require_job_snapshot(job_id)

    if job.state == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=job.error or "Transcription failed.",
        )
    if job.state != "done" or not job.transcript:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Result is not ready yet.",
        )

    return TranscriptionResponse(
        transcript=job.transcript,
        filename=job.filename,
        subtitle_srt=job.subtitle_srt,
        has_subtitles=bool(job.subtitle_srt),
        detected_language=job.detected_language,
        detected_language_probability=job.detected_language_probability,
    )


@app.get("/api/jobs/{job_id}/download")
async def download_job_result(
    job_id: str,
    file_format: Literal["txt", "srt"] = Query("txt", alias="format"),
) -> PlainTextResponse:
    result = await get_job_result(job_id)
    if file_format == "srt":
        if not result.subtitle_srt:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This job does not contain SRT subtitles.",
            )
        headers = {"Content-Disposition": f'attachment; filename="{result.filename}.srt"'}
        return PlainTextResponse(result.subtitle_srt, headers=headers)

    headers = {"Content-Disposition": f'attachment; filename="{result.filename}.txt"'}
    return PlainTextResponse(result.transcript, headers=headers)


@app.get("/api/jobs/{job_id}/subtitle")
async def get_job_subtitle(job_id: str) -> PlainTextResponse:
    result = await get_job_result(job_id)
    if not result.subtitle_srt:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This job does not contain SRT subtitles.",
        )
    return PlainTextResponse(result.subtitle_srt)


@app.post("/api/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    language: LanguageOption = Form("pl"),
    mode: Literal["fast", "accurate"] = Form("fast"),
    output_format: OutputFormat = Form("txt"),
) -> TranscriptionResponse:
    safe_name = _safe_filename_stem(file.filename or "transcription")

    try:
        transcript, subtitle_srt, detected_language, detected_probability = await run_in_threadpool(
            _process_upload,
            file,
            language,
            mode,
            output_format,
        )
    except InvalidFileTypeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except VideoTooLongError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except NoAudioStreamError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except TranscriptionRuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        await file.close()

    return TranscriptionResponse(
        transcript=transcript,
        filename=safe_name,
        subtitle_srt=subtitle_srt,
        has_subtitles=bool(subtitle_srt),
        detected_language=detected_language,
        detected_language_probability=detected_probability,
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    _cleanup_expired_jobs()

    with _JOBS_LOCK:
        active_jobs = sum(1 for job in _JOBS.values() if job.state in {"queued", "processing"})
        queued_jobs = sum(1 for job in _JOBS.values() if job.state == "queued")
        processing_jobs = sum(1 for job in _JOBS.values() if job.state == "processing")

    with _TRANSCRIPT_CACHE_LOCK:
        cache_size = len(_TRANSCRIPT_CACHE)
    with _ETA_CACHE_LOCK:
        eta_cache_entries = len(_ETA_CACHE)

    db_row = _db_fetchone("SELECT COUNT(1) AS count FROM jobs")
    persisted_jobs = int(db_row["count"]) if db_row else 0

    return {
        "status": "ok",
        "active_jobs": active_jobs,
        "queued_jobs": queued_jobs,
        "processing_jobs": processing_jobs,
        "cache_size": cache_size,
        "persisted_jobs": persisted_jobs,
        "db_path": str(JOBS_DB_PATH),
        "workers": TRANSCRIBE_WORKERS,
        "max_pending_jobs": MAX_PENDING_JOBS,
        "ffmpeg_available": FFMPEG_AVAILABLE,
        "ffprobe_available": FFPROBE_AVAILABLE,
        "eta_cache_entries": eta_cache_entries,
        "eta_stats_window": ETA_STATS_WINDOW,
        "eta_min_samples": ETA_MIN_SAMPLES,
    }
