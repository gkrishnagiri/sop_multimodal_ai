from __future__ import annotations

import json
import re
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
        refined_activities = self._postprocess_refined_activities(refined_activities)

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
5. Merge repetitive activities when they represent the same continuous workflow phase, same page state, or repeated review/check/filter/scroll behavior with no new UI action.
6. Preserve the overall chronological order.
7. Preserve source activity IDs so every refined activity is traceable.
8. Preserve timestamps using the earliest start and latest end from the source activities.
9. Preserve evidence summaries and frame paths.
10. Remove noisy OCR phrases when they do not help describe the workflow.
11. Do not create one step for every timestamp, frame, or scroll if the user is doing the same business action. Summarize repeated behavior into one reusable step.
12. Convert repeated review/check patterns into loop-style steps such as "Review each visible record", "Check each available option", or "Repeat until all visible options are reviewed".
13. Keep 3 to 12 meaningful steps per refined activity when possible.
14. Avoid raw personal data. Use generic terms such as "user account", "selected record", "saved profile", "registered contact details", "selected item", or "configured option".
15. If the evidence shows only a review or preparation step, do not convert it into a completed submission, payment, approval, booking, save, close, or confirmation.
16. If speech uses words like "can", "could", "would", "will", "if I", or "then I can", treat that as optional or explanatory future behavior, not an observed completed action.
17. Return only valid JSON matching the provided schema.
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
- Aggressively merge repeated adjacent steps when the instruction, intent, page, or target is essentially the same.
- Avoid repeating the same step with the same expected result just because many frames/timeline segments were observed.
- Use loop-style wording for repeated item review, such as "For each visible option..." or "Repeat until all relevant results are reviewed."
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

    def _postprocess_refined_activities(self, refined: dict[str, Any]) -> dict[str, Any]:
        """Deterministic cleanup after LLM refinement.

        This protects downstream SOP quality when the LLM still mirrors repeated
        OCR/timeline segments as repeated steps.
        """
        refined = dict(refined)
        cleaned_activities = []

        for activity in refined.get("activities", []) or []:
            if not isinstance(activity, dict):
                continue
            cleaned_activity = dict(activity)
            cleaned_activity["steps"] = self._compress_steps(cleaned_activity.get("steps", []) or [])
            cleaned_activities.append(cleaned_activity)

        # Merge adjacent activities if they still have effectively the same name and intent.
        merged_activities: list[dict[str, Any]] = []
        for activity in cleaned_activities:
            if merged_activities and self._activity_merge_key(merged_activities[-1]) == self._activity_merge_key(activity):
                merged_activities[-1] = self._merge_activity_pair(merged_activities[-1], activity)
            else:
                merged_activities.append(activity)

        for activity_index, activity in enumerate(merged_activities, start=1):
            activity["activity_id"] = f"refined_activity_{activity_index:03d}"
            activity["steps"] = self._compress_steps(activity.get("steps", []) or [])

        refined["activities"] = merged_activities
        refined["activity_count"] = len(merged_activities)
        metadata = refined.get("metadata") if isinstance(refined.get("metadata"), dict) else {}
        metadata["postprocessor"] = "mvp11_2_activity_step_dedupe_and_loop_compression"
        refined["metadata"] = metadata
        return refined

    def _compress_steps(self, steps: list[Any]) -> list[dict[str, Any]]:
        compressed: list[dict[str, Any]] = []
        seen_low_value_keys: set[str] = set()

        for raw_step in steps:
            if not isinstance(raw_step, dict):
                continue
            step = dict(raw_step)
            key = self._step_merge_key(step)

            if compressed and self._step_merge_key(compressed[-1]) == key:
                compressed[-1] = self._merge_step_pair(compressed[-1], step)
                seen_low_value_keys.add(key)
                continue

            if key in seen_low_value_keys and self._is_low_value_repeated_step(step):
                compressed[-1] = self._add_repeat_note(compressed[-1])
                continue

            compressed.append(step)
            if self._is_low_value_repeated_step(step):
                seen_low_value_keys.add(key)

        for index, step in enumerate(compressed, start=1):
            step["step_number"] = index

        return compressed

    def _merge_step_pair(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged = dict(left)
        merged["end_seconds"] = right.get("end_seconds", merged.get("end_seconds", 0))

        source_refs = []
        source_refs.extend(left.get("source_step_refs", []) or [])
        source_refs.extend(right.get("source_step_refs", []) or [])
        merged["source_step_refs"] = self._dedupe_keep_order(source_refs)

        left_evidence = left.get("evidence") if isinstance(left.get("evidence"), dict) else {}
        right_evidence = right.get("evidence") if isinstance(right.get("evidence"), dict) else {}
        merged_evidence = dict(left_evidence)

        speech_parts = [
            self._clean_text(left_evidence.get("speech_summary", "")),
            self._clean_text(right_evidence.get("speech_summary", "")),
        ]
        merged_evidence["speech_summary"] = self._truncate(
            " ".join(part for part in speech_parts if part),
            320,
        )

        screen_text = []
        screen_text.extend(left_evidence.get("screen_text", []) or [])
        screen_text.extend(right_evidence.get("screen_text", []) or [])
        merged_evidence["screen_text"] = self._dedupe_keep_order(screen_text)[:12]

        if not merged_evidence.get("frame_path"):
            merged_evidence["frame_path"] = right_evidence.get("frame_path", "")

        merged["evidence"] = merged_evidence
        merged = self._add_repeat_note(merged)
        return merged

    def _add_repeat_note(self, step: dict[str, Any]) -> dict[str, Any]:
        step = dict(step)
        note = "Repeated similar observations were consolidated. Repeat this action for all visible matching items/options when applicable."
        expected = self._clean_text(step.get("expected_result", ""))
        if note.lower() not in expected.lower():
            step["expected_result"] = f"{expected} {note}".strip()
        return step

    def _step_merge_key(self, step: dict[str, Any]) -> str:
        """Return a stable key for repeated-step compression.

        Do not include raw OCR screen text in the key. OCR can change slightly
        from frame to frame even when the user is performing the same business
        action. The key should represent the reusable action, not the frame.
        """
        parts = [
            step.get("instruction", ""),
            step.get("ui_action", ""),
            step.get("expected_result", ""),
        ]
        return self._normalize_for_dedupe(" ".join(str(part) for part in parts))

    def _activity_merge_key(self, activity: dict[str, Any]) -> str:
        return self._normalize_for_dedupe(
            f"{activity.get('name', '')} {activity.get('description', '')}"
        )

    def _merge_activity_pair(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        merged = dict(left)
        merged["end_seconds"] = right.get("end_seconds", merged.get("end_seconds", 0))
        merged["duration_seconds"] = round(
            max(0.0, float(merged.get("end_seconds", 0) or 0) - float(merged.get("start_seconds", 0) or 0)),
            2,
        )

        source_ids = []
        source_ids.extend(left.get("source_activity_ids", []) or [])
        source_ids.extend(right.get("source_activity_ids", []) or [])
        if right.get("activity_id"):
            source_ids.append(str(right.get("activity_id")))
        merged["source_activity_ids"] = self._dedupe_keep_order(source_ids)

        merged["steps"] = self._compress_steps((left.get("steps", []) or []) + (right.get("steps", []) or []))

        left_summary = self._clean_text(left.get("evidence_summary", ""))
        right_summary = self._clean_text(right.get("evidence_summary", ""))
        merged["evidence_summary"] = self._truncate(
            " ".join(part for part in [left_summary, right_summary] if part),
            500,
        )
        return merged

    def _is_low_value_repeated_step(self, step: dict[str, Any]) -> bool:
        text = " ".join(
            str(step.get(key, ""))
            for key in ["instruction", "ui_action", "expected_result"]
        ).lower()
        repeated_terms = [
            "review", "check", "visible", "displayed", "results", "options", "availability",
            "filter", "scroll", "configure", "preference", "payment options",
            "list", "table", "record", "row", "status", "details", "additional"
        ]
        return any(term in text for term in repeated_terms)

    def _normalize_for_dedupe(self, text: str) -> str:
        text = self._clean_text(text).lower()
        text = re.sub(r"\bstep\s+\d+\b", "step", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _truncate(self, text: str, limit: int) -> str:
        text = self._clean_text(text)
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

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