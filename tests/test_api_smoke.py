import asyncio
import atexit
import os

import httpx

os.environ.setdefault("PRELOAD_FAST_MODEL", "0")
os.environ.setdefault("PRELOAD_ACCURATE_MODEL", "0")
os.environ.setdefault("JOBS_DB_PATH", "/tmp/clip_to_text_test.sqlite3")
from app.main import app, shutdown_workers, startup_warmup

startup_warmup()
atexit.register(shutdown_workers)


def request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def test_home_page_loads() -> None:
    response = request("GET", "/")
    assert response.status_code == 200
    assert "Clip to Text" in response.text


def test_health_endpoint_shape() -> None:
    response = request("GET", "/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "active_jobs" in payload
    assert "queued_jobs" in payload
    assert "processing_jobs" in payload
    assert "cache_size" in payload
    assert "persisted_jobs" in payload
    assert "max_pending_jobs" in payload


def test_list_jobs_endpoint() -> None:
    response = request("GET", "/api/jobs?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert "jobs" in payload
    assert isinstance(payload["jobs"], list)


def test_job_status_for_missing_job() -> None:
    response = request("GET", "/api/jobs/does-not-exist")
    assert response.status_code == 404


def test_cancel_missing_job_returns_not_found() -> None:
    response = request("POST", "/api/jobs/does-not-exist/cancel")
    assert response.status_code == 404


def test_create_job_rejects_invalid_extension() -> None:
    files = {"file": ("notes.txt", b"hello", "text/plain")}
    data = {"language": "pl", "mode": "fast"}
    response = request("POST", "/api/jobs", files=files, data=data)
    assert response.status_code == 400
