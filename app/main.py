"""Application entry point: builds the FastAPI app from the modules in this package.

- ``config``: settings read from the environment
- ``schemas``: errors, internal records and response models
- ``storage``: SQLite persistence, transcript cache, upload and temp files
- ``transcription``: ffmpeg audio extraction and faster-whisper
- ``jobs``: background job queue, ETA estimates and maintenance
- ``routes``: HTTP endpoints
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT
from app.jobs import shutdown_workers, startup_warmup
from app.routes import router

app = FastAPI(title="Clip to Text Pro")
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
app.include_router(router)
app.on_event("startup")(startup_warmup)
app.on_event("shutdown")(shutdown_workers)

__all__ = ["app", "shutdown_workers", "startup_warmup"]
