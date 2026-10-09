"""SQLite persistence, the transcript cache and upload/temp file handling."""

import json
import re
import sqlite3
import time
from hashlib import sha256
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Literal

from fastapi import UploadFile

from app.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    JOBS_DB_PATH,
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_MB,
    MKV_SIGNATURE,
    MODE_CONFIG,
    TRANSCRIPT_CACHE_MAX_ITEMS,
    TRANSCRIPT_CACHE_TTL_SECONDS,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    LanguageOption,
    OutputFormat,
)
from app.schemas import (
    CachedTranscript,
    FileTooLargeError,
    InvalidFileTypeError,
    TranscriptionJob,
    UploadMetadata,
)


_TRANSCRIPT_CACHE: dict[tuple[str, str, str, str, str, str, str], CachedTranscript] = {}
_TRANSCRIPT_CACHE_LOCK = Lock()

_DB_CONNECTION: sqlite3.Connection | None = None
_DB_LOCK = RLock()


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
