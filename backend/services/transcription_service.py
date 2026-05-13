import os
from pathlib import Path

from faster_whisper import WhisperModel

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job

if settings.hf_token:
    os.environ["HF_TOKEN"] = settings.hf_token

model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    audio_path = job.get("audio_path")
    if not audio_path:
        raise ValueError("Audio not found. Run extract-audio first.")

    segments, info = model.transcribe(audio_path, beam_size=5)

    transcript_lines = []

    for segment in segments:
        line = f"[{segment.start:.2f} - {segment.end:.2f}] {segment.text.strip()}"
        transcript_lines.append(line)

    transcript_text = "\n".join(transcript_lines)

    transcript_dir = Path(settings.data_dir) / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcript_dir / f"{job_id}.txt"

    with transcript_path.open("w", encoding="utf-8") as f:
        f.write(transcript_text)

    update_job(job_id, {
        "transcript_path": str(transcript_path),
        "status": "transcribed",
    })

    return {
        "job_id": job_id,
        "status": job["status"],
        "language": info.language,
        "language_probability": info.language_probability,
        "transcript_path": str(transcript_path),
        "text_preview": transcript_text[:500],
    }