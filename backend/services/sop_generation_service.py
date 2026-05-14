from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from backend.config.settings import settings
from backend.services.job_service import JOBS, update_job


class SopGenerationService:
    """
    MVP 8: Generic SOP generation from activities JSON.

    Input:
        data/activities/{job_id}.json

    Output:
        data/outputs/{job_id}_sop.json
        data/outputs/{job_id}_sop.md

    Design principles:
    - Generic across applications, domains, and scenarios
    - No hard-coded business process rules
    - Uses activities/steps/evidence as the source of truth
    - Does not claim completion unless evidence clearly shows completion
    - Produces structured SOP suitable for later editing and frontend display
    """

    def __init__(self) -> None:
        self.data_dir = Path(settings.data_dir)
        self.activities_dir = self.data_dir / "activities"
        self.outputs_dir = self.data_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def generate_sop_for_job(self, job_id: str) -> dict[str, Any]:
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

        sop_json = self._generate_sop_with_llm(
            job_id=job_id,
            activities_data=compact_input,
        )

        sop_json_path = self.outputs_dir / f"{job_id}_sop.json"
        sop_md_path = self.outputs_dir / f"{job_id}_sop.md"

        self._write_json(sop_json_path, sop_json)

        markdown = self._sop_json_to_markdown(sop_json)
        sop_md_path.write_text(markdown, encoding="utf-8")

        update_job(
            job_id,
            {
                "sop_json_path": str(sop_json_path),
                "sop_markdown_path": str(sop_md_path),
                "status": "sop_generated",
            },
        )

        return {
            "job_id": job_id,
            "status": "sop_generated",
            "sop_json_path": str(sop_json_path),
            "sop_markdown_path": str(sop_md_path),
            "sop": sop_json,
        }

    def get_sop_for_job(self, job_id: str) -> dict[str, Any]:
        sop_json_path = self.outputs_dir / f"{job_id}_sop.json"
        sop_md_path = self.outputs_dir / f"{job_id}_sop.md"

        if not sop_json_path.exists():
            raise FileNotFoundError(
                f"SOP file not found: {sop_json_path}. Run generate-sop first."
            )

        sop_json = self._load_json(sop_json_path)
        markdown = sop_md_path.read_text(encoding="utf-8") if sop_md_path.exists() else ""

        return {
            "job_id": job_id,
            "sop_json_path": str(sop_json_path),
            "sop_markdown_path": str(sop_md_path),
            "sop": sop_json,
            "markdown": markdown,
        }

    def _generate_sop_with_llm(
        self,
        job_id: str,
        activities_data: dict[str, Any],
    ) -> dict[str, Any]:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

        client = OpenAI(api_key=settings.openai_api_key)

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(job_id=job_id, activities_data=activities_data)

        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sop_generation_result",
                    "strict": True,
                    "schema": self._sop_json_schema(),
                },
            },
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("LLM returned empty SOP response.")

        try:
            sop = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM returned invalid JSON: {exc}") from exc

        sop["job_id"] = job_id
        sop["metadata"] = {
            "llm_model": settings.llm_model,
            "generator": "generic_llm_sop_generator_v1",
            "source": "activities_json",
        }

        return sop

    def _system_prompt(self) -> str:
        return """
You are an expert SOP analyst.

Your task is to convert multimodal workflow activity data into a clean, generic Standard Operating Procedure.

Important rules:
1. Do not assume the application, business domain, user intent, final outcome, or process completion beyond the evidence provided.
2. Do not invent steps that are not supported by the activities, speech, screen text, timestamps, or frame references.
3. Do not hard-code any application-specific behavior.
4. Keep the SOP generic enough to apply to the observed workflow while still using observed terminology when it is clearly supported by evidence.
5. Use clear business language.
6. Convert noisy transcript text into clean instructions.
7. Preserve traceability by including evidence timestamps and frame paths where useful.
8. Do not include sensitive raw personal data unless it is necessary to explain the workflow. Prefer generic terms such as "selected record", "saved profile", "registered contact details", "user account", "selected item", or "configured option".
9. Distinguish clearly between observed actions and inferred purpose.
10. Only state that the workflow, transaction, submission, booking, request, case, form, payment, approval, or process was completed if the evidence explicitly shows a final completion, success, confirmation, submitted, paid, saved, closed, approved, or equivalent final state.
11. If the recording stops before the final completion step, describe the SOP as covering the workflow up to the last observed step.
12. If payment, submission, approval, confirmation, or finalization is mentioned but not completed in the evidence, phrase it as "review payment options", "prepare to submit", "review before final submission", "ready to proceed", or similar non-final wording.
13. Completion criteria must reflect only what is actually observed or what is safely required before handoff to the next step.
14. Exceptions, notes, quality checks, and completion criteria must not claim outcomes that are not evidenced.
15. Do not include finalization words such as "finalize", "complete", "submit", "approve", "pay", "save", "close", or "confirm" as completed actions unless the evidence explicitly shows that completed state.
16. If the user says "can", "could", "would", "will", "if I", "then I can", or similar conditional/future language, treat it as explanatory or optional future behavior, not as an observed completed action.
17. Expected results must describe observed screen states or immediate outcomes, not downstream business outcomes that are not shown.
18. The final activity and final step must be consistent with the observed_end_state.
19. Return only valid JSON matching the provided schema.
""".strip()

    def _user_prompt(
        self,
        job_id: str,
        activities_data: dict[str, Any],
    ) -> str:
        return f"""
Generate a structured SOP from the following generic activity detection output.

Job ID:
{job_id}

Activity data:
{json.dumps(activities_data, ensure_ascii=False, indent=2)}

The SOP should include:
- title
- overview
- scope
- observed_end_state
- prerequisites
- business_process_description
- activities
- steps
- atomic UI-level actions
- evidence references
- exceptions_or_notes
- quality_checks
- completion_criteria

Generic evidence rules:
- Use only the provided activity data as evidence.
- Do not invent missing steps.
- Do not assume that the overall workflow was completed unless a final completion, success, confirmation, submitted, saved, approved, paid, or equivalent final state is explicitly present in the evidence.
- If the final visible step is a review, payment selection, confirmation prompt, draft, preview, continue button, or pre-submit screen, the SOP must end at that point and must not include final completion, submission, payment, approval, save, close, or confirmation as a completed or executable step.
- The final activity and final steps must be consistent with observed_end_state.
- Do not include steps such as "finalize", "complete", "submit", "approve", "pay", "save", "close", or "confirm" unless that action is explicitly shown as completed in the evidence.
- If the speech says the user "can", "could", "would", "will", "if I", or "then I can", treat that as explanatory or optional future behavior, not as an observed completed action.
- If the recording stops before completion, the final step should be phrased as "review options", "select option", "prepare to proceed", or "stop before final submission/payment/completion".
- Expected results must describe the observed screen state or immediate visible result, not the downstream business outcome.
- Use "observed workflow", "demonstrated process", or "workflow shown in the recording" when the final business outcome is not confirmed.
- Completion criteria should describe the end state reached in the recording, not an assumed downstream result.
- Use generic wording that works across any application or scenario.
- Avoid raw personal data. Use generic descriptions for saved users, selected records, contact details, IDs, emails, phone numbers, names, addresses, or account-specific information.

Return JSON only.
""".strip()

    def _build_compact_llm_input(self, activities_data: dict[str, Any]) -> dict[str, Any]:
        """
        Reduce raw activities JSON into a compact input for the LLM.

        This keeps the LLM focused on:
        - activity name
        - timing
        - speech
        - representative OCR
        - frame evidence

        It avoids passing excessive repeated OCR text.
        """
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
                        "speech": self._clean_text(step.get("speech", "")),
                        "screen_text_sample": self._dedupe_keep_order(
                            step.get("screen_text_sample", [])
                        )[:10],
                        "frame_path": step.get("frame_path"),
                    }
                )

            compact_activities.append(
                {
                    "activity_id": activity.get("activity_id"),
                    "name": activity.get("name"),
                    "description": self._clean_text(activity.get("description", "")),
                    "start_seconds": activity.get("start_seconds"),
                    "end_seconds": activity.get("end_seconds"),
                    "duration_seconds": activity.get("duration_seconds"),
                    "dominant_intent": activity.get("dominant_intent"),
                    "evidence": {
                        "speech_samples": [
                            self._clean_text(item)
                            for item in activity.get("evidence", {}).get("speech_samples", [])[:5]
                        ],
                        "screen_text_samples": self._dedupe_keep_order(
                            activity.get("evidence", {}).get("screen_text_samples", [])
                        )[:10],
                        "frame_paths": activity.get("evidence", {}).get("frame_paths", [])[:3],
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

    def _sop_json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "job_id": {"type": "string"},
                "title": {"type": "string"},
                "overview": {"type": "string"},
                "scope": {"type": "string"},
                "observed_end_state": {"type": "string"},
                "prerequisites": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "business_process_description": {"type": "string"},
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
                                        "evidence": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "properties": {
                                                "start_seconds": {"type": "number"},
                                                "end_seconds": {"type": "number"},
                                                "speech_summary": {"type": "string"},
                                                "screen_text": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                },
                                                "frame_path": {"type": "string"},
                                            },
                                            "required": [
                                                "start_seconds",
                                                "end_seconds",
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
                            "steps",
                        ],
                    },
                },
                "exceptions_or_notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "quality_checks": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "completion_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
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
                "title",
                "overview",
                "scope",
                "observed_end_state",
                "prerequisites",
                "business_process_description",
                "activities",
                "exceptions_or_notes",
                "quality_checks",
                "completion_criteria",
                "metadata",
            ],
        }

    def _sop_json_to_markdown(self, sop: dict[str, Any]) -> str:
        lines: list[str] = []

        lines.append(f"# {sop.get('title', 'Standard Operating Procedure')}")
        lines.append("")

        lines.append("## Overview")
        lines.append(sop.get("overview", ""))
        lines.append("")

        lines.append("## Scope")
        lines.append(sop.get("scope", ""))
        lines.append("")

        lines.append("## Observed End State")
        lines.append(sop.get("observed_end_state", ""))
        lines.append("")

        lines.append("## Prerequisites")
        prerequisites = sop.get("prerequisites", [])
        if prerequisites:
            for item in prerequisites:
                lines.append(f"- {item}")
        else:
            lines.append("- None specified.")
        lines.append("")

        lines.append("## Business Process Description")
        lines.append(sop.get("business_process_description", ""))
        lines.append("")

        lines.append("## Activities and Steps")
        for activity in sop.get("activities", []):
            lines.append("")
            lines.append(f"### {activity.get('name', 'Activity')}")
            lines.append("")
            lines.append(activity.get("description", ""))
            lines.append("")

            for step in activity.get("steps", []):
                step_number = step.get("step_number")
                instruction = step.get("instruction", "")
                ui_action = step.get("ui_action", "")
                expected_result = step.get("expected_result", "")
                evidence = step.get("evidence", {})

                lines.append(f"{step_number}. **{instruction}**")
                lines.append(f"   - UI action: {ui_action}")
                lines.append(f"   - Expected result: {expected_result}")

                start_seconds = evidence.get("start_seconds")
                end_seconds = evidence.get("end_seconds")
                frame_path = evidence.get("frame_path")

                if start_seconds is not None and end_seconds is not None:
                    lines.append(f"   - Evidence time: {start_seconds}s - {end_seconds}s")

                if frame_path:
                    lines.append(f"   - Frame: `{frame_path}`")

                lines.append("")

        lines.append("## Exceptions or Notes")
        notes = sop.get("exceptions_or_notes", [])
        if notes:
            for item in notes:
                lines.append(f"- {item}")
        else:
            lines.append("- None specified.")
        lines.append("")

        lines.append("## Quality Checks")
        quality_checks = sop.get("quality_checks", [])
        if quality_checks:
            for item in quality_checks:
                lines.append(f"- {item}")
        else:
            lines.append("- None specified.")
        lines.append("")

        lines.append("## Completion Criteria")
        completion_criteria = sop.get("completion_criteria", [])
        if completion_criteria:
            for item in completion_criteria:
                lines.append(f"- {item}")
        else:
            lines.append("- None specified.")
        lines.append("")

        return "\n".join(lines).strip() + "\n"

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