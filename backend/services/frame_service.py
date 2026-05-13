from pathlib import Path

import cv2

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


def extract_frames_from_video(job_id: str, interval_seconds: int = 5):
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    video_path = job["video"]["stored_path"]

    frames_dir = Path(settings.data_dir) / "frames" / job_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise RuntimeError("Unable to open video file")

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Unable to determine video FPS")

    frame_interval = int(fps * interval_seconds)
    saved_frames = []

    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if frame_number % frame_interval == 0:
            timestamp_seconds = frame_number / fps
            frame_filename = f"frame_{frame_number}_ts_{timestamp_seconds:.2f}.jpg"
            frame_path = frames_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            saved_frames.append({
                "frame_number": frame_number,
                "timestamp_seconds": round(timestamp_seconds, 2),
                "path": str(frame_path),
            })

        frame_number += 1

    video.release()

    update_job(job_id, {
        "frames_dir": str(frames_dir),
        "frames": saved_frames,
        "status": "frames_extracted",
    })

    return {
        "job_id": job_id,
        "status": job["status"],
        "fps": fps,
        "total_frames": total_frames,
        "interval_seconds": interval_seconds,
        "frames_saved": len(saved_frames),
        "frames_dir": str(frames_dir),
        "sample_frames": saved_frames[:5],
    }