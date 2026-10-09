"""End-to-end job tests with ffmpeg and Whisper replaced by fakes."""

import asyncio
import atexit
import importlib
import os
import time
import uuid
from types import ModuleType
from typing import Any, Callable

import httpx
import pytest

os.environ.setdefault("PRELOAD_FAST_MODEL", "0")
os.environ.setdefault("PRELOAD_ACCURATE_MODEL", "0")
os.environ.setdefault("JOBS_DB_PATH", "/tmp/clip_to_text_test.sqlite3")
from app.main import app, shutdown_workers, startup_warmup  # noqa: E402

startup_warmup()
atexit.register(shutdown_workers)

_CANDIDATE_MODULES = ("app.jobs", "app.transcription", "app.storage", "app.main")


def _find(name: str) -> Any:
    for module_name in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)


def _worker_module() -> ModuleType:
    return importlib.import_module(_find("_run_transcription_job").__module__)


def request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def _fake_mp4() -> bytes:
    # Minimal ISO BMFF header followed by unique bytes so the transcript cache
    # never hits by accident across test runs.
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + uuid.uuid4().bytes * 4


def _fake_transcribe(
    audio_path: Any,
    language: str,
    mode: str,
    output_format: str,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_callback: Callable[[], None] | None = None,
) -> tuple[str, str | None, str | None, float | None]:
    if cancel_callback is not None:
        cancel_callback()
    if progress_callback is not None:
        progress_callback(60.0, "Transcribing audio...")
    subtitle = "1\n00:00:00,000 --> 00:00:01,500\nHello world." if output_format == "txt_srt" else None
    return "Hello world.", subtitle, "en", 0.93


def _wait_for_finish(job_id: str, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = request("GET", f"/api/jobs/{job_id}").json()
        if payload["state"] in {"done", "error"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish in {timeout} s")


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _worker_module()
    monkeypatch.setattr(module, "_extract_audio_to_wav", lambda video, audio: None)
    monkeypatch.setattr(module, "_transcribe_audio", _fake_transcribe)


def _upload(content: bytes, output_format: str = "txt_srt", name: str = "clip one.mp4") -> httpx.Response:
    files = {"file": (name, content, "video/mp4")}
    data = {"language": "auto", "mode": "fast", "output_format": output_format}
    return request("POST", "/api/jobs", files=files, data=data)


def test_job_runs_to_completion_and_exposes_results(fake_engine: None) -> None:
    response = _upload(_fake_mp4())
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_payload = _wait_for_finish(job_id)
    assert status_payload["state"] == "done"
    assert status_payload["stage"] == "done"
    assert status_payload["progress"] == 100.0
    assert status_payload["ready"] is True
    assert status_payload["has_subtitles"] is True
    assert status_payload["filename"] == "clip_one"
    assert status_payload["detected_language"] == "en"
    assert status_payload["detected_language_probability"] == 0.93

    result = request("GET", f"/api/jobs/{job_id}/result").json()
    assert result["transcript"] == "Hello world."
    assert result["subtitle_srt"].startswith("1\n00:00:00,000 --> 00:00:01,500")

    txt = request("GET", f"/api/jobs/{job_id}/download", params={"format": "txt"})
    assert txt.status_code == 200
    assert txt.text == "Hello world."
    assert txt.headers["content-disposition"] == 'attachment; filename="clip_one.txt"'

    srt = request("GET", f"/api/jobs/{job_id}/download", params={"format": "srt"})
    assert srt.headers["content-disposition"] == 'attachment; filename="clip_one.srt"'
    assert request("GET", f"/api/jobs/{job_id}/subtitle").text == srt.text

    events = request("GET", f"/api/jobs/{job_id}/events")
    assert events.headers["content-type"].startswith("text/event-stream")
    assert events.text.startswith("retry: 3000")
    assert '"state": "done"' in events.text

    listed = request("GET", "/api/jobs", params={"limit": 100}).json()["jobs"]
    assert any(job["job_id"] == job_id for job in listed)

    cancel = request("POST", f"/api/jobs/{job_id}/cancel")
    assert cancel.status_code == 409


def test_same_upload_is_served_from_cache(fake_engine: None) -> None:
    content = _fake_mp4()
    first = _upload(content, output_format="txt")
    first_status = _wait_for_finish(first.json()["job_id"])
    assert first_status["state"] == "done"
    assert first_status["has_subtitles"] is False

    second = _upload(content, output_format="txt")
    second_status = request("GET", f"/api/jobs/{second.json()['job_id']}").json()
    assert second_status["state"] == "done"
    assert second_status["message"] == "Done (cache hit)."

    srt = request("GET", f"/api/jobs/{second.json()['job_id']}/download", params={"format": "srt"})
    assert srt.status_code == 409


def test_failed_transcription_marks_job_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _worker_module()
    runtime_error = _find("TranscriptionRuntimeError")

    def failing(*args: Any, **kwargs: Any) -> None:
        raise runtime_error("ffmpeg exploded")

    monkeypatch.setattr(module, "_extract_audio_to_wav", failing)

    job_id = _upload(_fake_mp4()).json()["job_id"]
    status_payload = _wait_for_finish(job_id)
    assert status_payload["state"] == "error"
    assert status_payload["stage"] == "error"
    assert status_payload["message"] == "Transcription failed."
    assert status_payload["error"] == "ffmpeg exploded"

    result = request("GET", f"/api/jobs/{job_id}/result")
    assert result.status_code == 400
    assert result.json()["detail"] == "ffmpeg exploded"


def test_upload_with_wrong_container_is_rejected() -> None:
    response = _upload(b"\x00" * 64)
    assert response.status_code == 400
    assert "MP4/MOV" in response.json()["detail"]


def test_srt_and_transcript_helpers() -> None:
    build_srt = _find("_build_srt")
    normalize = _find("_normalize_transcript")

    assert normalize("  Hello ,   world  !  ") == "Hello, world!"
    assert build_srt([]) is None
    assert build_srt([(0.0, 0.0, " Hi "), (3661.5, 3662.25, "there")]) == (
        "1\n00:00:00,000 --> 00:00:00,400\nHi\n\n2\n01:01:01,500 --> 01:01:02,250\nthere"
    )
