import json
import re
from pathlib import Path

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


TIMESTAMP_PATTERN = re.compile(r"\[(\d+(?:\.\d+)?) - (\d+(?:\.\d+)?)\]\s*(.*)")


def _round_time(value: float) -> float:
    return round(float(value), 3)


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

                segments.append(
                    {
                        "start_seconds": start,
                        "end_seconds": end,
                        "midpoint_seconds": round((start + end) / 2, 2),
                        "speech": text,
                    }
                )

    return segments


def _load_ocr(ocr_path: str) -> list[dict]:
    with open(ocr_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        results = payload.get("results", [])

        if isinstance(results, list):
            return results

    return []


def _find_nearest_ocr(midpoint: float, ocr_results: list[dict]) -> dict | None:
    valid_results = [
        item
        for item in ocr_results
        if isinstance(item, dict)
        and item.get("timestamp_seconds") is not None
        and not item.get("error")
    ]

    if not valid_results:
        return None

    return min(
        valid_results,
        key=lambda item: abs(float(item["timestamp_seconds"]) - midpoint),
    )


def _diarization_path_for_job(job_id: str) -> Path:
    return Path(settings.data_dir) / "diarization" / f"{job_id}.json"


def _load_diarization_segments(job_id: str) -> list[dict]:
    """
    Load diarization segments if they exist.

    Diarization is optional. If data/diarization/{job_id}.json does not exist,
    the timeline should still build without speaker labels.
    """

    diarization_path = _diarization_path_for_job(job_id)

    if not diarization_path.exists():
        return []

    with diarization_path.open("r", encoding="utf-8") as f:
        diarization_payload = json.load(f)

    segments = diarization_payload.get("segments", [])

    if not isinstance(segments, list):
        return []

    valid_segments = []

    for segment in segments:
        try:
            start_seconds = float(segment["start_seconds"])
            end_seconds = float(segment["end_seconds"])
            speaker = str(segment["speaker"])
        except (KeyError, TypeError, ValueError):
            continue

        if end_seconds <= start_seconds:
            continue

        valid_segments.append(
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "speaker": speaker,
            }
        )

    return valid_segments


def _calculate_overlap_seconds(
    *,
    segment_start: float,
    segment_end: float,
    speaker_start: float,
    speaker_end: float,
) -> float:
    overlap_start = max(segment_start, speaker_start)
    overlap_end = min(segment_end, speaker_end)

    if overlap_end <= overlap_start:
        return 0.0

    return overlap_end - overlap_start


def _find_best_speaker_for_segment(
    *,
    segment_start: float,
    segment_end: float,
    diarization_segments: list[dict],
) -> dict:
    """
    Find the speaker with the largest overlap with a transcript segment.

    Returns:
        {
            "speaker": str | None,
            "speaker_overlap_seconds": float,
            "speaker_overlap_ratio": float
        }

    If no diarization segment overlaps, speaker remains None.
    """

    segment_duration = max(segment_end - segment_start, 0.0)

    if segment_duration <= 0 or not diarization_segments:
        return {
            "speaker": None,
            "speaker_overlap_seconds": 0.0,
            "speaker_overlap_ratio": 0.0,
        }

    overlap_by_speaker: dict[str, float] = {}

    for speaker_segment in diarization_segments:
        speaker = speaker_segment["speaker"]

        overlap_seconds = _calculate_overlap_seconds(
            segment_start=segment_start,
            segment_end=segment_end,
            speaker_start=speaker_segment["start_seconds"],
            speaker_end=speaker_segment["end_seconds"],
        )

        if overlap_seconds <= 0:
            continue

        overlap_by_speaker[speaker] = (
            overlap_by_speaker.get(speaker, 0.0) + overlap_seconds
        )

    if not overlap_by_speaker:
        return {
            "speaker": None,
            "speaker_overlap_seconds": 0.0,
            "speaker_overlap_ratio": 0.0,
        }

    best_speaker = max(overlap_by_speaker, key=overlap_by_speaker.get)
    best_overlap_seconds = overlap_by_speaker[best_speaker]
    best_overlap_ratio = best_overlap_seconds / segment_duration

    return {
        "speaker": best_speaker,
        "speaker_overlap_seconds": _round_time(best_overlap_seconds),
        "speaker_overlap_ratio": _round_time(best_overlap_ratio),
    }


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
    diarization_segments = _load_diarization_segments(job_id)

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
                if isinstance(item, dict) and item.get("text")
            ]

        speaker_info = _find_best_speaker_for_segment(
            segment_start=segment["start_seconds"],
            segment_end=segment["end_seconds"],
            diarization_segments=diarization_segments,
        )

        timeline.append(
            {
                "start_seconds": segment["start_seconds"],
                "end_seconds": segment["end_seconds"],
                "midpoint_seconds": segment["midpoint_seconds"],
                "speech": segment["speech"],
                "speaker": speaker_info["speaker"],
                "speaker_overlap_seconds": speaker_info["speaker_overlap_seconds"],
                "speaker_overlap_ratio": speaker_info["speaker_overlap_ratio"],
                "nearest_frame_timestamp_seconds": frame_timestamp,
                "frame_path": frame_path,
                "screen_text": screen_text,
            }
        )

    timeline_dir = Path(settings.data_dir) / "timeline"
    timeline_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = timeline_dir / f"{job_id}.json"

    with timeline_path.open("w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)

    speaker_labeled_segments = sum(
        1 for segment in timeline if segment.get("speaker") is not None
    )

    update_job(
        job_id,
        {
            "timeline_path": str(timeline_path),
            "timeline_segments": len(timeline),
            "timeline_speaker_labeled_segments": speaker_labeled_segments,
            "diarization_used": bool(diarization_segments),
            "status": "timeline_built",
        },
    )

    return {
        "job_id": job_id,
        "status": "timeline_built",
        "timeline_path": str(timeline_path),
        "segments": len(timeline),
        "diarization_used": bool(diarization_segments),
        "speaker_labeled_segments": speaker_labeled_segments,
        "sample": timeline[:3],
    }