from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.services.job_service import create_upload_job, get_job
from backend.services.audio_service import extract_audio_from_video
from backend.services.transcription_service import transcribe_audio
from backend.services.frame_service import extract_frames_from_video
from backend.services.ocr_service import run_ocr_on_frames
from backend.services.timeline_service import build_timeline
from backend.services.activity_detection_service import ActivityDetectionService
from backend.services.sop_generation_service import SopGenerationService

router = APIRouter(prefix="/api")

activity_detection_service = ActivityDetectionService()
sop_generation_service = SopGenerationService()

@router.post("/jobs/upload")
def upload_video(
    file: UploadFile = File(...),
    output_filename: str = Form(...),
    enable_diarization: bool = Form(True),
    extract_screenshots: bool = Form(True),
):
    if not output_filename.strip():
        raise HTTPException(status_code=400, detail="output_filename is required")

    return create_upload_job(
        file=file,
        output_filename=output_filename.strip(),
        enable_diarization=enable_diarization,
        extract_screenshots=extract_screenshots,
    )


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.post("/jobs/{job_id}/extract-audio")
def extract_audio(job_id: str):
    try:
        return extract_audio_from_video(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Job not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/transcribe")
def transcribe(job_id: str):
    try:
        return transcribe_audio(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/extract-frames")
def extract_frames(job_id: str, interval_seconds: int = 5):
    try:
        return extract_frames_from_video(
            job_id=job_id,
            interval_seconds=interval_seconds,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/run-ocr")
def run_ocr(job_id: str):
    try:
        return run_ocr_on_frames(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/build-timeline")
def build_job_timeline(job_id: str):
    try:
        return build_timeline(job_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/detect-activities")
def detect_job_activities(job_id: str):
    """
    MVP 7: Detect generic workflow activities from timeline JSON.

    Input:
        data/timeline/{job_id}.json

    Output:
        data/activities/{job_id}.json
    """
    try:
        result = activity_detection_service.detect_activities_for_job(job_id)

        return {
            "job_id": job_id,
            "status": "activities_detected",
            "activity_count": result.get("activity_count", 0),
            "activities_path": f"data/activities/{job_id}.json",
            "result": result,
        }

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
    """
    Return previously generated activity detection output.
    """
    try:
        return activity_detection_service.get_activities_for_job(job_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load activities: {str(e)}",
        )

@router.post("/jobs/{job_id}/generate-sop")
def generate_job_sop(job_id: str):
    """
    MVP 8: Generate a generic SOP from activities JSON.

    Input:
        data/activities/{job_id}.json

    Output:
        data/outputs/{job_id}_sop.json
        data/outputs/{job_id}_sop.md
    """
    try:
        return sop_generation_service.generate_sop_for_job(job_id)

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
    """
    Return previously generated SOP output.
    """
    try:
        return sop_generation_service.get_sop_for_job(job_id)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load SOP: {str(e)}",
        )