from __future__ import annotations

import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
DIARIZATION_DIR = DATA_DIR / "diarization"
TEMP_DIR = DATA_DIR / "tmp"

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_MAX_DURATION_SECONDS = 60


class DiarizationServiceError(RuntimeError):
    """Raised when speaker diarization cannot be completed."""


def _round_time(value: float) -> float:
    return round(float(value), 3)


def _load_env() -> None:
    """
    Load backend/.env explicitly.

    This keeps behavior stable whether the app is started from:
    - project root
    - backend folder
    - uv --project backend
    """

    load_dotenv(BACKEND_DIR / ".env")


@lru_cache(maxsize=1)
def _get_pipeline() -> Any:
    """
    Lazy-load the pyannote pipeline.

    Important:
    - Do not import pyannote.audio at module import time.
    - Do not load the model at FastAPI startup.
    - Load only when /run-diarization is called.
    """

    _load_env()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise DiarizationServiceError(
            "HF_TOKEN is missing. Add HF_TOKEN=your_token_here to backend/.env"
        )

    model_name = os.getenv("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)

    try:
        from pyannote.audio import Pipeline
    except Exception as exc:
        raise DiarizationServiceError(
            "pyannote.audio is not installed or cannot be imported. "
            "Run: uv add pyannote.audio"
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token)
    except Exception as exc:
        raise DiarizationServiceError(
            f"Failed to load diarization model: {model_name}. "
            "Confirm that you accepted the model terms on Hugging Face "
            "and that HF_TOKEN is valid."
        ) from exc

    return pipeline


def _audio_path_for_job(job_id: str) -> Path:
    return AUDIO_DIR / f"{job_id}.wav"


def _output_path_for_job(job_id: str) -> Path:
    return DIARIZATION_DIR / f"{job_id}.json"


def _temp_audio_path_for_job(job_id: str, max_duration_seconds: int) -> Path:
    return TEMP_DIR / f"{job_id}_first_{max_duration_seconds}s.wav"


def _create_limited_audio_file(
    *,
    job_id: str,
    source_audio_path: Path,
    max_duration_seconds: int,
) -> Path:
    """
    Create a temporary clipped WAV for diarization.

    Output format:
    - mono
    - 16 kHz
    - WAV

    This keeps MVP 9 validation fast and avoids waiting on full recordings.
    """

    if max_duration_seconds <= 0:
        raise DiarizationServiceError("max_duration_seconds must be greater than 0")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    limited_audio_path = _temp_audio_path_for_job(
        job_id=job_id,
        max_duration_seconds=max_duration_seconds,
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_audio_path),
        "-t",
        str(max_duration_seconds),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(limited_audio_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise DiarizationServiceError(
            "ffmpeg was not found. Install ffmpeg and ensure it is available on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise DiarizationServiceError(
            "Failed to create temporary clipped audio for diarization. "
            f"ffmpeg stderr: {exc.stderr}"
        ) from exc

    return limited_audio_path


def _unwrap_diarization_annotation(diarization_result: Any) -> Any:
    """
    Normalize pyannote output across versions.

    Older pyannote versions returned an Annotation-like object directly:
        diarization_result.itertracks(yield_label=True)

    Newer pyannote versions may return a DiarizeOutput wrapper object.
    In that case, the actual speaker annotation is usually available as:
        diarization_result.speaker_diarization

    This function returns the object that supports itertracks(...).
    """

    if hasattr(diarization_result, "itertracks"):
        return diarization_result

    candidate_attribute_names = [
        "speaker_diarization",
        "diarization",
        "annotation",
        "output",
    ]

    for attribute_name in candidate_attribute_names:
        candidate = getattr(diarization_result, attribute_name, None)

        if candidate is not None and hasattr(candidate, "itertracks"):
            return candidate

    available_attributes = sorted(
        attribute_name
        for attribute_name in dir(diarization_result)
        if not attribute_name.startswith("_")
    )

    raise DiarizationServiceError(
        "Could not find speaker diarization annotation in pyannote output. "
        f"Output type: {type(diarization_result).__name__}. "
        f"Available attributes: {available_attributes}"
    )


def _serialize_diarization(
    *,
    job_id: str,
    source_audio_path: Path,
    diarization_audio_path: Path,
    model_name: str,
    diarization_result: Any,
    elapsed_seconds: float,
    max_duration_seconds: int,
) -> dict[str, Any]:
    annotation = _unwrap_diarization_annotation(diarization_result)

    segments: list[dict[str, Any]] = []

    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start_seconds = _round_time(turn.start)
        end_seconds = _round_time(turn.end)
        duration_seconds = _round_time(end_seconds - start_seconds)

        if duration_seconds <= 0:
            continue

        segments.append(
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": duration_seconds,
                "speaker": str(speaker),
            }
        )

    segments.sort(key=lambda item: (item["start_seconds"], item["end_seconds"]))

    speaker_labels = sorted({segment["speaker"] for segment in segments})

    return {
        "job_id": job_id,
        "source_audio_path": str(source_audio_path),
        "diarization_audio_path": str(diarization_audio_path),
        "model": model_name,
        "max_duration_seconds": max_duration_seconds,
        "elapsed_seconds": _round_time(elapsed_seconds),
        "speaker_count": len(speaker_labels),
        "speakers": speaker_labels,
        "segment_count": len(segments),
        "segments": segments,
    }


def run_diarization(job_id: str) -> dict[str, Any]:
    """
    Run speaker diarization for a job's extracted WAV audio.

    Current MVP 9 behavior:
        Diarize only the first DEFAULT_MAX_DURATION_SECONDS seconds.

    Input:
        data/audio/{job_id}.wav

    Temporary audio:
        data/tmp/{job_id}_first_60s.wav

    Output:
        data/diarization/{job_id}.json
    """

    source_audio_path = _audio_path_for_job(job_id)

    if not source_audio_path.exists():
        raise DiarizationServiceError(
            f"Audio file not found for job_id={job_id}: {source_audio_path}. "
            "Run audio extraction before diarization."
        )

    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)

    _load_env()
    model_name = os.getenv("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)

    max_duration_seconds = DEFAULT_MAX_DURATION_SECONDS

    diarization_audio_path = _create_limited_audio_file(
        job_id=job_id,
        source_audio_path=source_audio_path,
        max_duration_seconds=max_duration_seconds,
    )

    print(
        "[diarization] Starting diarization",
        {
            "job_id": job_id,
            "source_audio_path": str(source_audio_path),
            "diarization_audio_path": str(diarization_audio_path),
            "max_duration_seconds": max_duration_seconds,
            "model": model_name,
        },
        flush=True,
    )

    pipeline = _get_pipeline()

    started_at = time.perf_counter()

    try:
        diarization_result = pipeline(str(diarization_audio_path))
    except Exception as exc:
        raise DiarizationServiceError(
            f"Failed to run diarization for job_id={job_id}."
        ) from exc

    elapsed_seconds = time.perf_counter() - started_at

    print(
        "[diarization] Raw output type",
        {
            "job_id": job_id,
            "output_type": type(diarization_result).__name__,
        },
        flush=True,
    )

    payload = _serialize_diarization(
        job_id=job_id,
        source_audio_path=source_audio_path,
        diarization_audio_path=diarization_audio_path,
        model_name=model_name,
        diarization_result=diarization_result,
        elapsed_seconds=elapsed_seconds,
        max_duration_seconds=max_duration_seconds,
    )

    output_path = _output_path_for_job(job_id)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        "[diarization] Completed diarization",
        {
            "job_id": job_id,
            "output_path": str(output_path),
            "speaker_count": payload["speaker_count"],
            "segment_count": payload["segment_count"],
            "elapsed_seconds": payload["elapsed_seconds"],
        },
        flush=True,
    )

    return {
        "job_id": job_id,
        "status": "completed",
        "diarization_path": str(output_path),
        "speaker_count": payload["speaker_count"],
        "segment_count": payload["segment_count"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "model": payload["model"],
        "max_duration_seconds": max_duration_seconds,
    }


def get_diarization(job_id: str) -> dict[str, Any]:
    output_path = _output_path_for_job(job_id)

    if not output_path.exists():
        raise DiarizationServiceError(
            f"Diarization output not found for job_id={job_id}: {output_path}"
        )

    return json.loads(output_path.read_text(encoding="utf-8"))