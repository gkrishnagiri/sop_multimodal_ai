import json
import re
from pathlib import Path

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


TIMESTAMP_PATTERN = re.compile(r"\[(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)\]\s*(.*)")


def _parse_transcript(transcript_path: str) -> list[dict]:
    segments = []

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            match = TIMESTAMP_PATTERN.match(line)

            if match:
                start = float(match.group(1))
                end = float(match.group(2))
                text = match.group(3).strip()

                segments.append({
                    "start_seconds": start,
                    "end_seconds": end,
                    "midpoint_seconds": round((start + end) / 2, 2),
                    "speech": text,
                })

    return segments


def _load_ocr(ocr_path: str) -> list[dict]:
    with open(ocr_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_nearest_ocr(midpoint: float, ocr_results: list[dict]) -> dict | None:
    if not ocr_results:
        return None

    return min(
        ocr_results,
        key=lambda item: abs(item["timestamp_seconds"] - midpoint),
    )


def build_timeline(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    transcript_path = job.get("transcript_path")
    ocr_path = job.get("ocr_path")

    if not transcript_path:
        raise ValueError("Transcript not found. Run transcription first.")

    if not ocr_path:
        raise ValueError("OCR not found. Run OCR first.")

    transcript_segments = _parse_transcript(transcript_path)
    ocr_results = _load_ocr(ocr_path)

    timeline = []

    for segment in transcript_segments:
        nearest_ocr = _find_nearest_ocr(
            midpoint=segment["midpoint_seconds"],
            ocr_results=ocr_results,
        )

        screen_text = []
        frame_path = None
        frame_timestamp = None

        if nearest_ocr:
            frame_path = nearest_ocr.get("frame_path")
            frame_timestamp = nearest_ocr.get("timestamp_seconds")
            screen_text = [
                item["text"]
                for item in nearest_ocr.get("text", [])
                if item.get("text")
            ]

        timeline.append({
            "start_seconds": segment["start_seconds"],
            "end_seconds": segment["end_seconds"],
            "midpoint_seconds": segment["midpoint_seconds"],
            "speech": segment["speech"],
            "nearest_frame_timestamp_seconds": frame_timestamp,
            "frame_path": frame_path,
            "screen_text": screen_text,
        })

    timeline_dir = Path(settings.data_dir) / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = timeline_dir / f"{job_id}.json"

    with timeline_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    update_job(job_id, {
        "timeline_path": str(timeline_path),
        "timeline_segments": len(timeline),
        "status": "timeline_built",
    })

    return {
        "job_id": job_id,
        "status": "timeline_built",
        "timeline_path": str(timeline_path),
        "segments": len(timeline),
        "sample": timeline[:3],
    }