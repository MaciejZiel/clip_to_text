"""Errors, internal records and API response models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.config import JobState, LanguageOption, OutputFormat


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


@dataclass
class EtaStats:
    average_runtime_seconds: float
    samples: int
    created_at: float
