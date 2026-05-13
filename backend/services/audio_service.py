import os
import subprocess
from pathlib import Path

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job

def extract_audio_from_video(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    video_path = job["video"]["stored_path"]

    audio_dir = Path(settings.data_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_path = audio_dir / f"{job_id}.wav"

    # ffmpeg command
    command = [
        "ffmpeg",
        "-y",  # overwrite if exists
        "-i", video_path,
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",  # 16kHz (good for speech models later)
        "-ac", "1",  # mono
        str(audio_path),
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")

    update_job(job_id, {
        "audio_path": str(audio_path),
        "status": "audio_extracted",
    })

    return {
        "job_id": job_id,
        "status": job["status"],
        "audio_path": str(audio_path),
    }