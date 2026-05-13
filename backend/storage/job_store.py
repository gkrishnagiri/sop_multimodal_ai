import json
from pathlib import Path

from backend.config.settings import settings


JOBS_FILE = Path(settings.data_dir) / "jobs" / "jobs.json"


def ensure_jobs_file():
    JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not JOBS_FILE.exists():
        JOBS_FILE.write_text("{}", encoding="utf-8")


def load_jobs() -> dict:
    ensure_jobs_file()

    with JOBS_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jobs(jobs: dict) -> None:
    ensure_jobs_file()

    with JOBS_FILE.open("w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)