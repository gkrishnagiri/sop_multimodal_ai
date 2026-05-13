from backend.storage.file_manager import save_uploaded_video
from backend.storage.job_store import load_jobs, save_jobs


JOBS = load_jobs()


def persist_jobs():
    save_jobs(JOBS)


def create_upload_job(file, output_filename: str, enable_diarization: bool, extract_screenshots: bool):
    saved_file = save_uploaded_video(file)

    job = {
        "job_id": saved_file["job_id"],
        "status": "uploaded",
        "output_filename": output_filename,
        "enable_diarization": enable_diarization,
        "extract_screenshots": extract_screenshots,
        "video": saved_file,
    }

    JOBS[job["job_id"]] = job
    persist_jobs()

    return job


def get_job(job_id: str):
    return JOBS.get(job_id)


def update_job(job_id: str, updates: dict):
    job = JOBS.get(job_id)

    if not job:
        return None

    job.update(updates)
    persist_jobs()

    return job