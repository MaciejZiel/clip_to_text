"""Background job queue: state, ETA estimates, cancellation and maintenance."""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from math import ceil
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from app.config import (
    ETA_CACHE_TTL_SECONDS,
    ETA_MIN_SAMPLES,
    ETA_STATS_WINDOW,
    FFMPEG_AVAILABLE,
    FFMPEG_BIN,
    FFPROBE_AVAILABLE,
    FFPROBE_BIN,
    JOB_MEMORY_TTL_SECONDS,
    JOB_RETENTION_SECONDS,
    MAINTENANCE_INTERVAL_SECONDS,
    MAX_VIDEO_DURATION_SECONDS,
    PRELOAD_ACCURATE_MODEL,
    PRELOAD_FAST_MODEL,
    TRANSCRIBE_WORKERS,
    JobState,
    LanguageOption,
    logger,
)
from app.schemas import (
    EtaStats,
    FileTooLargeError,
    InvalidFileTypeError,
    JobStatusResponse,
    JobSummary,
    NoAudioStreamError,
    TranscriptionCancelledError,
    TranscriptionJob,
    TranscriptionRuntimeError,
    VideoTooLongError,
)
from app.storage import (
    _cleanup_transcript_cache,
    _close_db,
    _db_execute,
    _db_fetchall,
    _db_fetchone,
    _get_job_from_db,
    _init_db,
    _remove_file,
    _remove_job_files,
    _row_to_job,
    _set_cached_transcript,
    _upsert_job,
)
from app.transcription import _extract_audio_to_wav, _preload_model, _transcribe_audio


_JOBS: dict[str, TranscriptionJob] = {}
_JOBS_LOCK = Lock()

_JOB_FUTURES: dict[str, Future[Any]] = {}
_JOB_FUTURES_LOCK = Lock()

_CANCELLED_JOBS: set[str] = set()
_CANCELLED_JOBS_LOCK = Lock()

_WORKER_POOL = ThreadPoolExecutor(max_workers=TRANSCRIBE_WORKERS)
_MAINTENANCE_STOP = Event()
_MAINTENANCE_THREAD: Thread | None = None


_ETA_CACHE: dict[tuple[str, str, str], EtaStats] = {}
_ETA_CACHE_LOCK = Lock()


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


def shutdown_workers() -> None:
    _MAINTENANCE_STOP.set()
    _WORKER_POOL.shutdown(wait=False, cancel_futures=True)
    _close_db()
