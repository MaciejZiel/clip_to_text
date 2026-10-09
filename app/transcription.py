"""Audio extraction with ffmpeg and speech recognition with faster-whisper."""

import re
import subprocess
import tempfile
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal

from fastapi import UploadFile

from app.config import (
    FFMPEG_BIN,
    FFMPEG_THREADS,
    FFPROBE_BIN,
    MAX_VIDEO_DURATION_SECONDS,
    MODE_CONFIG,
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_NUM_WORKERS,
    LanguageOption,
    OutputFormat,
)
from app.schemas import (
    InvalidFileTypeError,
    NoAudioStreamError,
    TranscriptionRuntimeError,
    VideoTooLongError,
)
from app.storage import (
    _get_cached_transcript,
    _save_upload,
    _set_cached_transcript,
    _validate_content_type,
    _validate_extension,
)

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None  # type: ignore[assignment]

try:
    from faster_whisper import BatchedInferencePipeline
except ImportError:
    BatchedInferencePipeline = None  # type: ignore[assignment]


_MODEL_CACHE: dict[tuple[str, str, str, int, int], Any] = {}
_PIPELINE_CACHE: dict[tuple[str, str, str, int, int, int], Any] = {}
_MODEL_LOCK = Lock()


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


def _preload_model(mode: Literal["fast", "accurate"]) -> None:
    try:
        _get_whisper_model(mode)
    except Exception:
        return


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
