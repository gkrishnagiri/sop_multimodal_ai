from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


class ActivityRefinementService:
    """
    MVP 8A: LLM-based generic activity refinement.

    Input:
        data/activities/{job_id}.json

    Output:
        data/refined_activities/{job_id}.json

    Design principles:
    - Generic across applications, domains, and scenarios
    - No hard-coded application-specific rules
    - Uses rule-based activities as an evidence-grounded baseline
    - Produces cleaner activity names and steps for SOP generation
    - Does not invent unobserved workflow completion
    """

    def __init__(self) -> None:
        self.data_dir = Path(settings.data_dir)
        self.activities_dir = self.data_dir / "activities"
        self.refined_activities_dir = self.data_dir / "refined_activities"
        self.refined_activities_dir.mkdir(parents=True, exist_ok=True)

    def refine_activities_for_job(self, job_id: str) -> dict[str, Any]:
        job = JOBS.get(job_id)

        if not job:
            raise ValueError("Job not found")

        activities_path = self.activities_dir / f"{job_id}.json"

        if not activities_path.exists():
            raise FileNotFoundError(
                f"Activities file not found: {activities_path}. Run detect-activities first."
            )

        activities_data = self._load_json(activities_path)
        compact_input = self._build_compact_llm_input(activities_data)

        refined_activities = self._refine_activities_with_llm(
            job_id=job_id,
            activities_data=compact_input,
        )

        output_path = self.refined_activities_dir / f"{job_id}.json"
        self._write_json(output_path, refined_activities)

        update_job(
            job_id,
            {
                "refined_activities_path": str(output_path),
                "refined_activity_count": refined_activities.get("activity_count", 0),
                "status": "activities_refined",
            },
        )

        return {
            "job_id": job_id,
            "status": "activities_refined",
            "refined_activities_path": str(output_path),
            "activity_count": refined_activities.get("activity_count", 0),
            "result": refined_activities,
        }

    def get_refined_activities_for_job(self, job_id: str) -> dict[str, Any]:
        refined_activities_path = self.refined_activities_dir / f"{job_id}.json"

        if not refined_activities_path.exists():
            raise FileNotFoundError(
                f"Refined activities file not found: {refined_activities_path}. "
                "Run refine-activities first."
            )

        return self._load_json(refined_activities_path)

    def _refine_activities_with_llm(
        self,
        job_id: str,
        activities_data: dict[str, Any],
    ) -> dict[str, Any]:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

        client = OpenAI(api_key=settings.openai_api_key)

        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "activity_refinement_result",
                    "strict": True,
                    "schema": self._refined_activities_schema(),
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._user_prompt(
                        job_id=job_id,
                        activities_data=activities_data,
                    ),
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("LLM returned empty activity refinement response.")

        try:
            refined = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        refined["job_id"] = job_id
        refined["activity_count"] = len(refined.get("activities", []))
        refined["metadata"] = {
            "llm_model": settings.llm_model,
            "generator": "generic_llm_activity_refiner_v1",
            "source": "activities_json",
        }

        return refined

    def _system_prompt(self) -> str:
        return """
You are an expert workflow analyst.

Your task is to refine machine-generated workflow activities into cleaner, business-friendly activities.

Important rules:
1. Be generic. Do not hard-code any application-specific, website-specific, company-specific, or scenario-specific logic.
2. Use only the provided activity data as evidence.
3. Do not invent steps, actions, screens, outcomes, or final completion states that are not supported by evidence.
4. Improve activity names so they are clear and business-friendly.
5. Merge repetitive activities only when they represent the same continuous workflow phase.
6. Preserve the overall chronological order.
7. Preserve source activity IDs so every refined activity is traceable.
8. Preserve timestamps using the earliest start and latest end from the source activities.
9. Preserve evidence summaries and frame paths.
10. Remove noisy OCR phrases when they do not help describe the workflow.
11. Avoid raw personal data. Use generic terms such as "user account", "selected record", "saved profile", "registered contact details", "selected item", or "configured option".
12. If the evidence shows only a review or preparation step, do not convert it into a completed submission, payment, approval, booking, save, close, or confirmation.
13. If speech uses words like "can", "could", "would", "will", "if I", or "then I can", treat that as optional or explanatory future behavior, not an observed completed action.
14. Return only valid JSON matching the provided schema.
""".strip()

    def _user_prompt(
        self,
        job_id: str,
        activities_data: dict[str, Any],
    ) -> str:
        return f"""
Refine the following generic rule-based activity detection output.

Job ID:
{job_id}

Activity data:
{json.dumps(activities_data, ensure_ascii=False, indent=2)}

Your output should:
- Produce cleaner activity names.
- Produce concise activity descriptions.
- Preserve timestamps.
- Preserve traceability through source_activity_ids.
- Produce cleaner step instructions.
- Keep evidence references.
- Avoid application-specific hard coding.
- Avoid unsupported completion claims.
- Keep the output generic enough to work for any video recording, any application, and any scenario.

Return JSON only.
""".strip()

    def _build_compact_llm_input(self, activities_data: dict[str, Any]) -> dict[str, Any]:
        compact_activities = []

        for activity in activities_data.get("activities", []):
            compact_steps = []

            for step in activity.get("steps", []):
                compact_steps.append(
                    {
                        "step_number": step.get("step_number"),
                        "start_seconds": step.get("start_seconds"),
                        "end_seconds": step.get("end_seconds"),
                        "intent": step.get("intent"),
                        "instruction": self._clean_text(step.get("instruction", "")),
                        "speech": self._clean_text(step.get("speech", "")),
                        "screen_text_sample": self._dedupe_keep_order(
                            step.get("screen_text_sample", [])
                        )[:8],
                        "frame_path": step.get("frame_path"),
                    }
                )

            compact_activities.append(
                {
                    "activity_id": activity.get("activity_id"),
                    "name": self._clean_text(activity.get("name", "")),
                    "description": self._clean_text(activity.get("description", "")),
                    "start_seconds": activity.get("start_seconds"),
                    "end_seconds": activity.get("end_seconds"),
                    "duration_seconds": activity.get("duration_seconds"),
                    "dominant_intent": activity.get("dominant_intent"),
                    "evidence": {
                        "speech_samples": [
                            self._clean_text(item)
                            for item in activity.get("evidence", {}).get("speech_samples", [])[:6]
                        ],
                        "screen_text_samples": self._dedupe_keep_order(
                            activity.get("evidence", {}).get("screen_text_samples", [])
                        )[:10],
                        "frame_paths": activity.get("evidence", {}).get("frame_paths", [])[:4],
                    },
                    "steps": compact_steps,
                }
            )

        return {
            "job_id": activities_data.get("job_id"),
            "activity_count": len(compact_activities),
            "activities": compact_activities,
            "metadata": activities_data.get("metadata", {}),
        }

    def _refined_activities_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "job_id": {"type": "string"},
                "activity_count": {"type": "integer"},
                "activities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "activity_id": {"type": "string"},
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "start_seconds": {"type": "number"},
                            "end_seconds": {"type": "number"},
                            "duration_seconds": {"type": "number"},
                            "source_activity_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {"type": "number"},
                            "evidence_summary": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "step_number": {"type": "integer"},
                                        "instruction": {"type": "string"},
                                        "ui_action": {"type": "string"},
                                        "expected_result": {"type": "string"},
                                        "start_seconds": {"type": "number"},
                                        "end_seconds": {"type": "number"},
                                        "source_step_refs": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "evidence": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "speech_summary": {"type": "string"},
                                                "screen_text": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "frame_path": {"type": "string"},
                                            },
                                            "required": [
                                                "speech_summary",
                                                "screen_text",
                                                "frame_path",
                                            ],
                                        },
                                    },
                                    "required": [
                                        "step_number",
                                        "instruction",
                                        "ui_action",
                                        "expected_result",
                                        "start_seconds",
                                        "end_seconds",
                                        "source_step_refs",
                                        "evidence",
                                    ],
                                },
                            },
                        },
                        "required": [
                            "activity_id",
                            "name",
                            "description",
                            "start_seconds",
                            "end_seconds",
                            "duration_seconds",
                            "source_activity_ids",
                            "confidence",
                            "evidence_summary",
                            "steps",
                        ],
                    },
                },
                "metadata": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "llm_model": {"type": "string"},
                        "generator": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": [
                        "llm_model",
                        "generator",
                        "source",
                    ],
                },
            },
            "required": [
                "job_id",
                "activity_count",
                "activities",
                "metadata",
            ],
        }

    def _load_json(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in {path}")

        return data

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clean_text(self, value: Any) -> str:
        if value is None:
            return ""

        text = str(value)
        return " ".join(text.split())

    def _dedupe_keep_order(self, items: list[Any]) -> list[str]:
        seen = set()
        result = []

        for item in items:
            text = self._clean_text(item)

            if not text:
                continue

            normalized = text.lower()

            if normalized in seen:
                continue

            seen.add(normalized)
            result.append(text)

        return result