import shutil
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

from backend.config.settings import settings
from backend.services.job_service import JOBS, create_upload_job, get_job, persist_jobs, update_job
from backend.services.audio_service import extract_audio_from_video
from backend.services.transcription_service import transcribe_audio
from backend.services.frame_service import extract_frames_from_video
from backend.services.ocr_service import run_ocr_on_frames
from backend.services.timeline_service import build_timeline
from backend.services.activity_detection_service import ActivityDetectionService
from backend.services.activity_refinement_service import ActivityRefinementService
from backend.services.sop_generation_service import SopGenerationService
from backend.services.diarization_service import (
    DiarizationServiceError,
    get_diarization,
    run_diarization,
)

router = APIRouter(prefix="/api")

activity_detection_service = ActivityDetectionService()
activity_refinement_service = ActivityRefinementService()
sop_generation_service = SopGenerationService()

class DeleteJobsRequest(BaseModel):
    job_ids: list[str] = Field(default_factory=list)


def _project_root() -> Path:
    return Path.cwd().resolve()


def _data_root() -> Path:
    data_dir = Path(settings.data_dir)

    if data_dir.is_absolute():
        return data_dir.resolve()

    return (_project_root() / data_dir).resolve()


def _resolve_project_path(value: Any) -> Path | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    path = Path(text)

    if not path.is_absolute():
        path = _project_root() / path

    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_safe_to_delete(path: Path) -> bool:
    project_root = _project_root()
    data_root = _data_root()

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()

    allowed_roots = [project_root, data_root]

    for root in allowed_roots:
        if resolved == root:
            return False

        if root in resolved.parents:
            return True

    return False


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    result: list[Path] = []

    for path in paths:
        text = str(path)

        if text in seen:
            continue

        seen.add(text)
        result.append(path)

    # Delete deeper paths first so files/directories inside job folders go before parents.
    result.sort(key=lambda item: len(item.parts), reverse=True)

    return result


def _collect_job_artifact_paths(job_id: str, job: dict[str, Any]) -> list[Path]:
    data_root = _data_root()
    paths: list[Path] = []

    explicit_path_fields = [
        "audio_path",
        "transcript_path",
        "ocr_path",
        "timeline_path",
        "activities_path",
        "refined_activities_path",
        "diarization_path",
        "sop_json_path",
        "sop_markdown_path",
        "sop_docx_path",
        "frames_dir",
    ]

    for field_name in explicit_path_fields:
        resolved = _resolve_project_path(job.get(field_name))

        if resolved:
            paths.append(resolved)

    video_payload = job.get("video")
    if isinstance(video_payload, dict):
        for field_name in ["stored_path", "path"]:
            resolved = _resolve_project_path(video_payload.get(field_name))

            if resolved:
                paths.append(resolved)

    frames = job.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue

            resolved = _resolve_project_path(frame.get("path"))

            if resolved:
                paths.append(resolved)

    standard_paths = [
        data_root / "audio" / f"{job_id}.wav",
        data_root / "transcripts" / f"{job_id}.txt",
        data_root / "ocr" / f"{job_id}.json",
        data_root / "timeline" / f"{job_id}.json",
        data_root / "activities" / f"{job_id}.json",
        data_root / "refined_activities" / f"{job_id}.json",
        data_root / "diarization" / f"{job_id}.json",
        data_root / "outputs" / f"{job_id}_sop.json",
        data_root / "outputs" / f"{job_id}_sop.md",
        data_root / "outputs" / f"{job_id}_sop.docx",
        data_root / "frames" / job_id,
        data_root / "tmp" / f"{job_id}_first_60s.wav",
        data_root / "tmp" / "ocr" / f"{job_id}_ocr_input.json",
        data_root / "tmp" / "ocr" / f"{job_id}_ocr_worker_output.json",
    ]

    paths.extend(standard_paths)

    return _dedupe_paths(paths)


def _delete_path_if_exists(path: Path) -> tuple[bool, str]:
    if not _is_safe_to_delete(path):
        return False, f"Skipped unsafe path: {path}"

    if not path.exists():
        return False, ""

    try:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    except OSError as exc:
        return False, f"Failed to delete {path}: {exc}"

    return True, ""


PIPELINE_STEPS = [
    "extract_audio",
    "transcribe",
    "extract_frames",
    "run_ocr",
    "run_diarization",
    "build_timeline",
    "detect_activities",
    "refine_activities",
    "generate_sop",
]


def _default_pipeline_steps() -> dict[str, dict[str, str]]:
    return {
        step: {
            "status": "not_started",
            "error": "",
        }
        for step in PIPELINE_STEPS
    }


def _get_existing_pipeline_steps(job_id: str) -> dict[str, dict[str, str]]:
    job = JOBS.get(job_id) or {}
    existing_steps = job.get("pipeline_steps")

    if not isinstance(existing_steps, dict):
        return _default_pipeline_steps()

    merged_steps = _default_pipeline_steps()

    for step_name, step_payload in existing_steps.items():
        if step_name not in merged_steps:
            continue

        if isinstance(step_payload, dict):
            merged_steps[step_name] = {
                "status": str(step_payload.get("status") or "not_started"),
                "error": str(step_payload.get("error") or ""),
            }

    return merged_steps


def _derive_pipeline_summary(job: dict[str, Any]) -> str:
    pipeline_status = job.get("pipeline_status")

    if pipeline_status in {"completed", "failed", "in_progress"}:
        return str(pipeline_status)

    if job.get("status") == "sop_generated":
        return "completed"

    if job.get("pipeline_failed_step"):
        return "failed"

    if job.get("status"):
        return "in_progress"

    return "uploaded"


def _mark_step_running(job_id: str, step_name: str) -> None:
    pipeline_steps = _get_existing_pipeline_steps(job_id)

    pipeline_steps[step_name] = {
        "status": "running",
        "error": "",
    }

    update_job(
        job_id,
        {
            "pipeline_status": "in_progress",
            "pipeline_current_step": step_name,
            "pipeline_failed_step": "",
            "pipeline_error": "",
            "pipeline_steps": pipeline_steps,
        },
    )


def _mark_step_completed(job_id: str, step_name: str) -> None:
    pipeline_steps = _get_existing_pipeline_steps(job_id)

    pipeline_steps[step_name] = {
        "status": "completed",
        "error": "",
    }

    payload: dict[str, Any] = {
        "pipeline_status": "in_progress",
        "pipeline_current_step": "",
        "pipeline_failed_step": "",
        "pipeline_error": "",
        "pipeline_steps": pipeline_steps,
    }

    if step_name == "generate_sop":
        payload["pipeline_status"] = "completed"

    update_job(job_id, payload)


def _mark_step_failed(job_id: str, step_name: str, error: str) -> None:
    pipeline_steps = _get_existing_pipeline_steps(job_id)

    pipeline_steps[step_name] = {
        "status": "failed",
        "error": error,
    }

    update_job(
        job_id,
        {
            "pipeline_status": "failed",
            "pipeline_current_step": "",
            "pipeline_failed_step": step_name,
            "pipeline_error": error,
            "pipeline_steps": pipeline_steps,
        },
    )


def _mark_step_skipped(job_id: str, step_name: str, reason: str = "") -> None:
    pipeline_steps = _get_existing_pipeline_steps(job_id)

    pipeline_steps[step_name] = {
        "status": "skipped",
        "error": reason,
    }

    update_job(
        job_id,
        {
            "pipeline_status": "in_progress",
            "pipeline_current_step": "",
            "pipeline_failed_step": "",
            "pipeline_error": "",
            "pipeline_steps": pipeline_steps,
        },
    )


def _run_tracked_step(
    *,
    job_id: str,
    step_name: str,
    operation: Callable[[], Any],
) -> Any:
    _mark_step_running(job_id, step_name)

    try:
        result = operation()
    except Exception as exc:
        _mark_step_failed(job_id, step_name, str(exc))
        raise

    _mark_step_completed(job_id, step_name)
    return result


def _job_with_summary(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    payload = dict(job)
    payload["job_id"] = job_id
    payload["pipeline_steps"] = _get_existing_pipeline_steps(job_id)
    payload["pipeline_status"] = _derive_pipeline_summary(payload)
    return payload


@router.post("/jobs/upload")
def upload_video(
    file: UploadFile = File(...),
    output_filename: str = Form(...),
    enable_diarization: bool = Form(False),
    extract_screenshots: bool = Form(True),
):
    if not output_filename.strip():
        raise HTTPException(status_code=400, detail="output_filename is required")

    result = create_upload_job(
        file=file,
        output_filename=output_filename.strip(),
        enable_diarization=enable_diarization,
        extract_screenshots=extract_screenshots,
    )

    job_id = result.get("job_id") or result.get("id")

    if job_id:
        update_job(
            job_id,
            {
                "pipeline_status": "uploaded",
                "pipeline_current_step": "",
                "pipeline_failed_step": "",
                "pipeline_error": "",
                "pipeline_steps": _default_pipeline_steps(),
            },
        )

    return result


@router.get("/jobs")
def list_jobs():
    jobs = []

    for job_id, job in JOBS.items():
        jobs.append(_job_with_summary(job_id, job))

    jobs.sort(
        key=lambda item: str(
            item.get("created_at")
            or item.get("updated_at")
            or item.get("job_id")
            or ""
        ),
        reverse=True,
    )

    return {
        "count": len(jobs),
        "jobs": jobs,
    }


@router.delete("/jobs")
def delete_jobs(request: DeleteJobsRequest):
    job_ids = []
    seen_job_ids = set()

    for raw_job_id in request.job_ids:
        job_id = str(raw_job_id).strip()

        if not job_id or job_id in seen_job_ids:
            continue

        seen_job_ids.add(job_id)
        job_ids.append(job_id)

    if not job_ids:
        raise HTTPException(status_code=400, detail="Provide at least one job_id to delete.")

    deleted_jobs = []
    missing_job_ids = []
    deleted_paths = []
    errors = []

    for job_id in job_ids:
        job = JOBS.get(job_id)

        if not job:
            missing_job_ids.append(job_id)
            continue

        artifact_paths = _collect_job_artifact_paths(job_id, job)

        for path in artifact_paths:
            deleted, error = _delete_path_if_exists(path)

            if deleted:
                deleted_paths.append(str(path))
            elif error:
                errors.append(error)

        JOBS.pop(job_id, None)
        deleted_jobs.append(job_id)

    persist_jobs()

    return {
        "status": "deleted",
        "requested_count": len(job_ids),
        "deleted_count": len(deleted_jobs),
        "missing_count": len(missing_job_ids),
        "deleted_job_ids": deleted_jobs,
        "missing_job_ids": missing_job_ids,
        "deleted_path_count": len(deleted_paths),
        "deleted_paths": deleted_paths,
        "errors": errors,
    }


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_with_summary(job_id, job)


@router.post("/jobs/{job_id}/extract-audio")
def extract_audio(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="extract_audio",
            operation=lambda: extract_audio_from_video(job_id),
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/transcribe")
def transcribe(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="transcribe",
            operation=lambda: transcribe_audio(job_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/extract-frames")
def extract_frames(job_id: str, interval_seconds: int = 5):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="extract_frames",
            operation=lambda: extract_frames_from_video(
                job_id=job_id,
                interval_seconds=interval_seconds,
            ),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/run-ocr")
def run_ocr(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="run_ocr",
            operation=lambda: run_ocr_on_frames(job_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/run-diarization")
def run_job_diarization(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not bool(job.get("enable_diarization", False)):
        reason = "Diarization disabled for this job."
        _mark_step_skipped(job_id, "run_diarization", reason)

        return {
            "job_id": job_id,
            "status": "skipped",
            "step": "run_diarization",
            "reason": reason,
        }

    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="run_diarization",
            operation=lambda: run_diarization(job_id),
        )
    except DiarizationServiceError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Diarization failed: {str(e)}",
        )


@router.get("/jobs/{job_id}/diarization")
def get_job_diarization(job_id: str):
    try:
        return get_diarization(job_id)
    except DiarizationServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load diarization: {str(e)}",
        )


@router.post("/jobs/{job_id}/build-timeline")
def build_job_timeline(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="build_timeline",
            operation=lambda: build_timeline(job_id),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/detect-activities")
def detect_job_activities(job_id: str):
    try:
        def operation():
            result = activity_detection_service.detect_activities_for_job(job_id)

            return {
                "job_id": job_id,
                "status": "activities_detected",
                "activity_count": result.get("activity_count", 0),
                "activities_path": f"data/activities/{job_id}.json",
                "result": result,
            }

        return _run_tracked_step(
            job_id=job_id,
            step_name="detect_activities",
            operation=operation,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Activity detection failed: {str(e)}",
        )


@router.get("/jobs/{job_id}/activities")
def get_job_activities(job_id: str):
    try:
        return activity_detection_service.get_activities_for_job(job_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load activities: {str(e)}",
        )


@router.post("/jobs/{job_id}/refine-activities")
def refine_job_activities(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="refine_activities",
            operation=lambda: activity_refinement_service.refine_activities_for_job(
                job_id
            ),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Activity refinement failed: {str(e)}",
        )


@router.get("/jobs/{job_id}/refined-activities")
def get_job_refined_activities(job_id: str):
    try:
        return activity_refinement_service.get_refined_activities_for_job(job_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load refined activities: {str(e)}",
        )


@router.post("/jobs/{job_id}/generate-sop")
def generate_job_sop(job_id: str):
    try:
        return _run_tracked_step(
            job_id=job_id,
            step_name="generate_sop",
            operation=lambda: sop_generation_service.generate_sop_for_job(job_id),
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SOP generation failed: {str(e)}",
        )


@router.get("/jobs/{job_id}/sop")
def get_job_sop(job_id: str):
    try:
        return sop_generation_service.get_sop_for_job(job_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load SOP: {str(e)}",
        )


@router.get("/jobs/{job_id}/sop/markdown")
def get_job_sop_markdown_file(job_id: str, download: bool = False):
    sop_markdown_path = Path(settings.data_dir) / "outputs" / f"{job_id}_sop.md"

    if not sop_markdown_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"SOP markdown file not found for job_id={job_id}. "
                "Run Generate SOP first."
            ),
        )

    disposition_type = "attachment" if download else "inline"

    return FileResponse(
        path=sop_markdown_path,
        media_type="text/markdown",
        filename=f"{job_id}_sop.md",
        content_disposition_type=disposition_type,
    )