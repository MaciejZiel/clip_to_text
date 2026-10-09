# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-10

First tagged release.

### Added

- Web app that transcribes `.mp4`, `.mov` and `.mkv` recordings locally with
  faster-whisper after extracting audio with ffmpeg.
- Background jobs: uploads return `202 Accepted` with a job id, a thread pool
  processes them, and the browser follows progress over Server-Sent Events with
  heartbeats and a polling fallback. Jobs can be cancelled.
- "Faster" (`tiny`) and "More accurate" (`medium`, beam search and VAD)
  profiles; Polish, English or automatic language detection with the detected
  language and its probability.
- Plain-text output and `.srt` subtitles built from segment timestamps,
  downloadable or available as JSON.
- Job history in SQLite that survives restarts, ETA estimates from recent jobs
  of the same profile, and a transcript cache keyed by the file's SHA-256 and
  the model settings.
- Upload checks (extension, MIME type, container magic bytes, streaming size
  limit), a bounded queue that returns `429` when full, and jobs interrupted by
  a restart marked as such.
- Light and dark theme for the page.
- API smoke tests and job-flow tests with ffmpeg and Whisper replaced by
  fakes, run in GitHub Actions on Python 3.10 and 3.12.
- CodeQL code scanning and Dependabot updates for pip and GitHub Actions.

### Changed

- The backend is split from a single `app/main.py` into `config`, `schemas`,
  `storage`, `transcription`, `jobs` and `routes` modules, with no behaviour
  change.
- The page headline describes the app as local transcription with Whisper
  instead of "production-ready".

### Fixed

- PyAV is capped below 19, which removed an argument faster-whisper passes and
  broke every transcription on a fresh install.

[1.0.0]: https://github.com/MaciejZiel/clip_to_text/releases/tag/v1.0.0
