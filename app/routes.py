"""HTTP endpoints: the page, the job API, SSE progress and the synchronous API."""

import asyncio
import json
import time
import uuid
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import (
    ETA_MIN_SAMPLES,
    ETA_STATS_WINDOW,
    FFMPEG_AVAILABLE,
    FFPROBE_AVAILABLE,
    JOBS_DB_PATH,
    JOBS_DIR,
    MAX_PENDING_JOBS,
    MAX_UPLOAD_MB,
    PROJECT_ROOT,
    SSE_HEARTBEAT_SECONDS,
    SSE_MAX_SECONDS,
    SSE_POLL_INTERVAL_SECONDS,
    TRANSCRIBE_WORKERS,
    LanguageOption,
    OutputFormat,
    logger,
)
from app.jobs import (
    _ETA_CACHE,
    _ETA_CACHE_LOCK,
    _JOB_FUTURES,
    _JOB_FUTURES_LOCK,
    _JOBS,
    _JOBS_LOCK,
    _WORKER_POOL,
    _active_jobs_count,
    _build_job_status,
    _build_job_summary,
    _cleanup_expired_jobs,
    _clear_job_cancelled,
    _get_job_snapshot,
    _mark_job_cancelled,
    _run_transcription_job,
    _update_job,
)
from app.schemas import (
    FileTooLargeError,
    InvalidFileTypeError,
    JobsListResponse,
    JobStartResponse,
    JobStatusResponse,
    NoAudioStreamError,
    TranscriptionJob,
    TranscriptionResponse,
    TranscriptionRuntimeError,
    UploadMetadata,
    VideoTooLongError,
)
from app.storage import (
    _TRANSCRIPT_CACHE,
    _TRANSCRIPT_CACHE_LOCK,
    _db_fetchall,
    _db_fetchone,
    _get_cached_transcript,
    _remove_job_files,
    _remove_temp_dir,
    _row_to_job,
    _safe_filename_stem,
    _save_upload,
    _upsert_job,
    _validate_content_type,
    _validate_extension,
)
from app.transcription import _probe_duration_seconds, _process_upload

router = APIRouter()
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


def _require_job_snapshot(job_id: str) -> TranscriptionJob:
    job = _get_job_snapshot(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
    return job


def _serialize_model(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()  # type: ignore[return-value]
    return model.dict()  # type: ignore[return-value]


@router.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"max_upload_mb": MAX_UPLOAD_MB},
    )


@router.get("/api/jobs", response_model=JobsListResponse)
async def list_jobs(limit: int = Query(12, ge=1, le=100)) -> JobsListResponse:
    _cleanup_expired_jobs()

    rows = _db_fetchall(
        "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    jobs = [_build_job_summary(_row_to_job(row)) for row in rows]
    return JobsListResponse(jobs=jobs)


@router.post("/api/jobs", response_model=JobStartResponse, status_code=status.HTTP_202_ACCEPTED)
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


@router.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    _cleanup_expired_jobs()
    job = _require_job_snapshot(job_id)
    return _build_job_status(job)


@router.post("/api/jobs/{job_id}/cancel", response_model=JobStatusResponse)
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


@router.get("/api/jobs/{job_id}/events")
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


@router.get("/api/jobs/{job_id}/result", response_model=TranscriptionResponse)
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


@router.get("/api/jobs/{job_id}/download")
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


@router.get("/api/jobs/{job_id}/subtitle")
async def get_job_subtitle(job_id: str) -> PlainTextResponse:
    result = await get_job_result(job_id)
    if not result.subtitle_srt:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This job does not contain SRT subtitles.",
        )
    return PlainTextResponse(result.subtitle_srt)


@router.post("/api/transcribe", response_model=TranscriptionResponse)
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


@router.get("/health")
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
