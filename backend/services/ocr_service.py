import json
from pathlib import Path

from paddleocr import PaddleOCR

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


ocr_engine = PaddleOCR(lang="en")


def normalize_ocr_result(result):
    detected_text = []

    if not result:
        return detected_text

    # Newer PaddleOCR sometimes returns a dict-style result
    if isinstance(result, dict):
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])

        for text, score in zip(texts, scores):
            detected_text.append({
                "text": str(text),
                "confidence": float(score),
            })

        return detected_text

    # Sometimes result is a list of dicts
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                texts = item.get("rec_texts", [])
                scores = item.get("rec_scores", [])

                for text, score in zip(texts, scores):
                    detected_text.append({
                        "text": str(text),
                        "confidence": float(score),
                    })

            elif isinstance(item, list):
                for line in item:
                    try:
                        text = line[1][0]
                        confidence = float(line[1][1])
                        detected_text.append({
                            "text": str(text),
                            "confidence": confidence,
                        })
                    except Exception:
                        continue

    return detected_text


def run_ocr_on_frames(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    frames = job.get("frames")
    if not frames:
        raise ValueError("Frames not found. Run extract-frames first.")

    ocr_results = []

    for frame in frames:
        frame_path = frame["path"]
        result = ocr_engine.ocr(frame_path)

        detected_text = normalize_ocr_result(result)

        ocr_results.append({
            "frame_number": frame["frame_number"],
            "timestamp_seconds": frame["timestamp_seconds"],
            "frame_path": frame_path,
            "text": detected_text,
        })

    ocr_dir = Path(settings.data_dir) / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    ocr_path = ocr_dir / f"{job_id}.json"

    with ocr_path.open("w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2)

    update_job(job_id, {
        "ocr_path": str(ocr_path),
        "ocr_results": ocr_results,
        "status": "ocr_completed",
    })

    return {
        "job_id": job_id,
        "status": "ocr_completed",
        "ocr_path": str(ocr_path),
        "frames_processed": len(ocr_results),
        "sample": ocr_results[:2],
    }