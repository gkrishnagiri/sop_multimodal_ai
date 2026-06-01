from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def _safe_confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _object_to_plain_data(value: Any) -> Any:
    """
    Convert common PaddleOCR result objects into JSON-friendly data.

    PaddleOCR versions differ:
    - Some return dict/list structures.
    - Some return result objects with json/dict-like methods.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _object_to_plain_data(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_object_to_plain_data(item) for item in value]

    if isinstance(value, tuple):
        return [_object_to_plain_data(item) for item in value]

    for method_name in ["json", "to_json", "dict", "to_dict"]:
        method = getattr(value, method_name, None)

        if callable(method):
            try:
                converted = method()
                return _object_to_plain_data(converted)
            except Exception:
                pass

    # Last resort: keep a string representation.
    return str(value)


def normalize_ocr_result(result: Any, min_confidence: float) -> list[dict[str, Any]]:
    """
    Normalize PaddleOCR outputs across multiple versions.

    Supported shapes:
    1. dict with rec_texts / rec_scores
    2. list of dicts with rec_texts / rec_scores
    3. older nested list format where each line has line[1][0], line[1][1]
    4. Paddle result objects converted by _object_to_plain_data
    """

    detected_text: list[dict[str, Any]] = []

    plain_result = _object_to_plain_data(result)

    if not plain_result:
        return detected_text

    def append_text(text: Any, confidence: Any) -> None:
        text_value = _safe_text(text)
        confidence_value = _safe_confidence(confidence)

        if not text_value:
            return

        if confidence_value < min_confidence:
            return

        detected_text.append(
            {
                "text": text_value,
                "confidence": confidence_value,
            }
        )

    def handle_dict(item: dict[str, Any]) -> None:
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])

        if isinstance(texts, list) and isinstance(scores, list):
            for text, score in zip(texts, scores):
                append_text(text, score)

        # Some versions may use different keys.
        text = item.get("text")
        score = item.get("confidence") or item.get("score")

        if text:
            append_text(text, score)

    if isinstance(plain_result, dict):
        handle_dict(plain_result)
        return detected_text

    if isinstance(plain_result, list):
        for item in plain_result:
            if isinstance(item, dict):
                handle_dict(item)

            elif isinstance(item, list):
                # Old PaddleOCR format:
                # [
                #   [
                #     [[x1,y1],...],
                #     ["recognized text", 0.98]
                #   ]
                # ]
                for line in item:
                    try:
                        text = line[1][0]
                        confidence = line[1][1]
                        append_text(text, confidence)
                    except Exception:
                        continue

    return detected_text


def _create_ocr_engine() -> Any:
    """
    Create PaddleOCR engine inside this worker process.

    PaddleOCR API parameters vary by version, so try the most compatible
    initialization first.
    """

    # These environment variables must be set before importing PaddleOCR.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")

    from paddleocr import PaddleOCR

    print("[ocr-worker] Initializing PaddleOCR engine", flush=True)

    init_attempts = [
        lambda: PaddleOCR(lang="en"),
        lambda: PaddleOCR(use_angle_cls=False, lang="en"),
    ]

    last_error: Exception | None = None

    for attempt in init_attempts:
        try:
            engine = attempt()
            print("[ocr-worker] PaddleOCR engine initialized", flush=True)
            return engine
        except TypeError as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to initialize PaddleOCR engine: {last_error}")


def _run_ocr_for_frame(ocr_engine: Any, frame_path: str, min_confidence: float) -> list[dict[str, Any]]:
    if not Path(frame_path).exists():
        raise RuntimeError(f"Frame file not found: {frame_path}")

    if hasattr(ocr_engine, "predict"):
        result = ocr_engine.predict(frame_path)
    else:
        result = ocr_engine.ocr(frame_path)

    return normalize_ocr_result(result, min_confidence=min_confidence)


def _load_input(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected input JSON object: {path}")

    return payload


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_worker(input_path: Path, output_path: Path) -> int:
    started_at = time.perf_counter()

    try:
        payload = _load_input(input_path)

        job_id = str(payload.get("job_id") or "")
        frames = payload.get("frames") or []
        min_confidence = float(payload.get("min_confidence") or 0.0)
        continue_on_frame_error = bool(payload.get("continue_on_frame_error", True))

        if not job_id:
            raise RuntimeError("job_id is missing from OCR worker input")

        if not isinstance(frames, list) or not frames:
            raise RuntimeError("No frames provided to OCR worker")

        print(
            "[ocr-worker] Starting OCR",
            {
                "job_id": job_id,
                "selected_frames": len(frames),
                "min_confidence": min_confidence,
                "continue_on_frame_error": continue_on_frame_error,
            },
            flush=True,
        )

        ocr_engine = _create_ocr_engine()

        results: list[dict[str, Any]] = []
        frame_errors: list[dict[str, Any]] = []

        for index, frame in enumerate(frames, start=1):
            frame_path = frame.get("path")
            frame_number = frame.get("frame_number")
            timestamp_seconds = frame.get("timestamp_seconds")

            print(
                "[ocr-worker] Processing frame",
                {
                    "job_id": job_id,
                    "index": index,
                    "selected_frames": len(frames),
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "frame_path": frame_path,
                },
                flush=True,
            )

            try:
                detected_text = _run_ocr_for_frame(
                    ocr_engine=ocr_engine,
                    frame_path=str(frame_path),
                    min_confidence=min_confidence,
                )

                results.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp_seconds,
                        "frame_path": frame_path,
                        "text": detected_text,
                        "error": None,
                    }
                )

            except Exception as exc:
                error_payload = {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "frame_path": frame_path,
                    "error_type": type(exc).__name__,
                    "error": str(exc) or repr(exc),
                }

                frame_errors.append(error_payload)

                print(
                    "[ocr-worker] Frame OCR failed",
                    {
                        "job_id": job_id,
                        **error_payload,
                    },
                    flush=True,
                )

                results.append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": timestamp_seconds,
                        "frame_path": frame_path,
                        "text": [],
                        "error": error_payload,
                    }
                )

                if not continue_on_frame_error:
                    raise RuntimeError(
                        "OCR failed for frame "
                        f"{frame_number} at {timestamp_seconds}s: "
                        f"{error_payload['error']}"
                    ) from exc

        elapsed_seconds = round(time.perf_counter() - started_at, 3)

        output_payload = {
            "job_id": job_id,
            "status": "worker_completed",
            "frames_processed": len(results),
            "frame_errors": frame_errors,
            "frame_error_count": len(frame_errors),
            "elapsed_seconds": elapsed_seconds,
            "results": results,
        }

        _write_output(output_path, output_payload)

        print(
            "[ocr-worker] Completed OCR",
            {
                "job_id": job_id,
                "frames_processed": len(results),
                "frame_error_count": len(frame_errors),
                "elapsed_seconds": elapsed_seconds,
                "output_path": str(output_path),
            },
            flush=True,
        )

        return 0

    except Exception as exc:
        elapsed_seconds = round(time.perf_counter() - started_at, 3)

        error_payload = {
            "status": "worker_failed",
            "error_type": type(exc).__name__,
            "error": str(exc) or repr(exc),
            "elapsed_seconds": elapsed_seconds,
            "results": [],
            "frame_errors": [],
            "frame_error_count": 0,
        }

        _write_output(output_path, error_payload)

        print(
            "[ocr-worker] Worker failed",
            error_payload,
            flush=True,
        )

        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    return run_worker(
        input_path=Path(args.input),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    sys.exit(main())