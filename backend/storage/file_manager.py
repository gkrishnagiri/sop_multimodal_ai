from pathlib import Path
from uuid import uuid4
from fastapi import UploadFile

from backend.config.settings import settings


def save_uploaded_video(file: UploadFile) -> dict:
    videos_dir = Path(settings.videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    job_id = str(uuid4())
    original_name = file.filename or "uploaded_video.mp4"
    extension = Path(original_name).suffix or ".mp4"

    stored_filename = f"{job_id}{extension}"
    stored_path = videos_dir / stored_filename

    with stored_path.open("wb") as buffer:
        buffer.write(file.file.read())

    return {
        "job_id": job_id,
        "original_filename": original_name,
        "stored_filename": stored_filename,
        "stored_path": str(stored_path),
    }