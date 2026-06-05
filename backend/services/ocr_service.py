from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


DEFAULT_OCR_FRAME_STRIDE = 1
DEFAULT_OCR_CONTINUE_ON_FRAME_ERROR = True
DEFAULT_OCR_MIN_CONFIDENCE = 0.0

# For MVP reliability, keep this False so transcript-only processing can continue
# even if PaddleOCR fails for all frames.
DEFAULT_OCR_FAIL_PIPELINE_IF_ALL_FRAMES_FAIL = False


class OcrServiceError(RuntimeError):
    """Raised when OCR cannot be completed."""


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_ocr_frame_stride() -> int:
    """
    Read OCR frame stride from backend settings.

    Important:
    Do not read this with os.getenv() because pydantic-settings reads values
    from .env into the settings object. It does not guarantee that .env values
    are exported back into os.environ.
    """

    try:
        value = int(settings.ocr_frame_stride)
    except (TypeError, ValueError):
        return DEFAULT_OCR_FRAME_STRIDE

    return max(value, 1)


def _get_ocr_continue_on_frame_error() -> bool:
    try:
        return bool(settings.ocr_continue_on_frame_error)
    except Exception:
        return DEFAULT_OCR_CONTINUE_ON_FRAME_ERROR


def _get_ocr_min_confidence() -> float:
    try:
        value = float(settings.ocr_min_confidence)
    except (TypeError, ValueError):
        return DEFAULT_OCR_MIN_CONFIDENCE

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value


def _get_ocr_fail_pipeline_if_all_frames_fail() -> bool:
    try:
        return bool(settings.ocr_fail_pipeline_if_all_frames_fail)
    except Exception:
        return DEFAULT_OCR_FAIL_PIPELINE_IF_ALL_FRAMES_FAIL


def _filter_frames(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Optionally reduce OCR workload.

    OCR_FRAME_STRIDE=1 means process every frame.
    OCR_FRAME_STRIDE=2 means process every second frame.
    OCR_FRAME_STRIDE=3 means process every third frame.
    """

    stride = _get_ocr_frame_stride()

    if stride <= 1:
        return frames

    return frames[::stride]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise OcrServiceError(f"Expected JSON object in {path}")

    return payload


def _run_ocr_worker(
    *,
    job_id: str,
    selected_frames: list[dict[str, Any]],
    worker_input_path: Path,
    worker_output_path: Path,
) -> dict[str, Any]:
    """
    Run OCR in a separate Python process and stream logs live.

    This avoids hiding worker logs until the process finishes.
    """

    project_root = _project_root()

    worker_payload = {
        "job_id": job_id,
        "frames": selected_frames,
        "min_confidence": _get_ocr_min_confidence(),
        "continue_on_frame_error": _get_ocr_continue_on_frame_error(),
    }

    _write_json(worker_input_path, worker_payload)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["PYTHONUNBUFFERED"] = "1"

    # Keep native libraries conservative inside WSL/CPU environment.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("FLAGS_allocator_strategy", "auto_growth")

    command = [
        sys.executable,
        "-u",
        "-m",
        "backend.services.ocr_frame_worker",
        "--input",
        str(worker_input_path),
        "--output",
        str(worker_output_path),
    ]

    print(
        "[ocr] Launching isolated OCR worker",
        {
            "job_id": job_id,
            "command": " ".join(command),
            "worker_input_path": str(worker_input_path),
            "worker_output_path": str(worker_output_path),
            "selected_frames": len(selected_frames),
        },
        flush=True,
    )

    process = subprocess.Popen(
        command,
        cwd=str(project_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    assert process.stdout is not None

    last_log_time = time.perf_counter()

    for line in process.stdout:
        line = line.rstrip()

        if line:
            print(line, flush=True)
            last_log_time = time.perf_counter()

    return_code = process.wait()

    silence_seconds = round(time.perf_counter() - last_log_time, 2)

    print(
        "[ocr] OCR worker exited",
        {
            "job_id": job_id,
            "return_code": return_code,
            "seconds_since_last_worker_log": silence_seconds,
            "worker_output_exists": worker_output_path.exists(),
        },
        flush=True,
    )

    if not worker_output_path.exists():
        raise OcrServiceError(
            "OCR worker did not create output JSON. "
            f"Return code: {return_code}. "
            "Check terminal logs above for the worker failure."
        )

    worker_result = _load_json(worker_output_path)

    if return_code != 0:
        error_message = worker_result.get("error") or "Unknown OCR worker error"
        raise OcrServiceError(
            "OCR worker failed. "
            f"Return code: {return_code}. "
            f"Error: {error_message}"
        )

    return worker_result


def run_ocr_on_frames(job_id: str) -> dict[str, Any]:
    job = JOBS.get(job_id)

    if not job:
        raise ValueError("Job not found")

    frames = job.get("frames")
    if not frames:
        raise ValueError("Frames not found. Run extract-frames first.")

    frame_stride = _get_ocr_frame_stride()
    selected_frames = _filter_frames(frames)
    continue_on_frame_error = _get_ocr_continue_on_frame_error()
    fail_pipeline_if_all_frames_fail = _get_ocr_fail_pipeline_if_all_frames_fail()
    min_confidence = _get_ocr_min_confidence()

    if not selected_frames:
        raise OcrServiceError("No frames selected for OCR.")

    started_at = time.perf_counter()

    print(
        "[ocr] Starting OCR with isolated worker",
        {
            "job_id": job_id,
            "total_frames": len(frames),
            "selected_frames": len(selected_frames),
            "frame_stride": frame_stride,
            "continue_on_frame_error": continue_on_frame_error,
            "min_confidence": min_confidence,
            "fail_pipeline_if_all_frames_fail": fail_pipeline_if_all_frames_fail,
            "settings_source": "backend.config.settings",
        },
        flush=True,
    )

    data_dir = Path(settings.data_dir)
    tmp_dir = data_dir / "tmp" / "ocr"
    ocr_dir = data_dir / "ocr"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    worker_input_path = tmp_dir / f"{job_id}_ocr_input.json"
    worker_output_path = tmp_dir / f"{job_id}_ocr_worker_output.json"
    final_ocr_path = ocr_dir / f"{job_id}.json"

    worker_result = _run_ocr_worker(
        job_id=job_id,
        selected_frames=selected_frames,
        worker_input_path=worker_input_path,
        worker_output_path=worker_output_path,
    )

    ocr_results = worker_result.get("results", [])
    frame_errors = worker_result.get("frame_errors", [])

    if not isinstance(ocr_results, list):
        ocr_results = []

    if not isinstance(frame_errors, list):
        frame_errors = []

    elapsed_seconds = round(time.perf_counter() - started_at, 3)

    successful_frame_count = sum(
        1
        for item in ocr_results
        if isinstance(item, dict) and not item.get("error")
    )

    all_selected_frames_failed = (
        len(selected_frames) > 0
        and len(frame_errors) == len(selected_frames)
        and successful_frame_count == 0
    )

    output_status = "ocr_completed"
    warning = ""

    if all_selected_frames_failed:
        warning = (
            "OCR failed for all selected frames. "
            "Pipeline is continuing with transcript-only evidence because "
            "OCR_FAIL_PIPELINE_IF_ALL_FRAMES_FAIL is false."
        )

        print(
            "[ocr] WARNING",
            {
                "job_id": job_id,
                "warning": warning,
                "first_error": frame_errors[0] if frame_errors else {},
            },
            flush=True,
        )

        if fail_pipeline_if_all_frames_fail:
            first_error = frame_errors[0] if frame_errors else {}
            raise OcrServiceError(
                "OCR failed for all selected frames. "
                f"First error: {first_error.get('error', 'Unknown error')}"
            )

        output_status = "ocr_completed_with_warnings"

    output_payload = {
        "job_id": job_id,
        "status": output_status,
        "total_frames": len(frames),
        "selected_frames": len(selected_frames),
        "frame_stride": frame_stride,
        "frames_processed": len(ocr_results),
        "successful_frame_count": successful_frame_count,
        "frame_errors": frame_errors,
        "frame_error_count": len(frame_errors),
        "all_selected_frames_failed": all_selected_frames_failed,
        "warning": warning,
        "elapsed_seconds": elapsed_seconds,
        "worker_output_path": str(worker_output_path),
        "results": ocr_results,
    }

    _write_json(final_ocr_path, output_payload)

    update_job(
        job_id,
        {
            "ocr_path": str(final_ocr_path),
            "ocr_total_frames": len(frames),
            "ocr_selected_frames": len(selected_frames),
            "ocr_frame_stride": frame_stride,
            "ocr_frames_processed": len(ocr_results),
            "ocr_successful_frame_count": successful_frame_count,
            "ocr_frame_error_count": len(frame_errors),
            "ocr_all_selected_frames_failed": all_selected_frames_failed,
            "ocr_warning": warning,
            "ocr_elapsed_seconds": elapsed_seconds,
            "status": output_status,
        },
    )

    print(
        "[ocr] Completed OCR step",
        {
            "job_id": job_id,
            "status": output_status,
            "ocr_path": str(final_ocr_path),
            "total_frames": len(frames),
            "selected_frames": len(selected_frames),
            "frame_stride": frame_stride,
            "frames_processed": len(ocr_results),
            "successful_frame_count": successful_frame_count,
            "frame_error_count": len(frame_errors),
            "all_selected_frames_failed": all_selected_frames_failed,
            "elapsed_seconds": elapsed_seconds,
        },
        flush=True,
    )

    return {
        "job_id": job_id,
        "status": output_status,
        "ocr_path": str(final_ocr_path),
        "total_frames": len(frames),
        "selected_frames": len(selected_frames),
        "frame_stride": frame_stride,
        "frames_processed": len(ocr_results),
        "successful_frame_count": successful_frame_count,
        "frame_error_count": len(frame_errors),
        "all_selected_frames_failed": all_selected_frames_failed,
        "warning": warning,
        "elapsed_seconds": elapsed_seconds,
        "sample": ocr_results[:2],
    }