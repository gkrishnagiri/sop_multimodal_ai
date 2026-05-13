from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ActivityDetectionConfig:
    timeline_dir: Path = Path("data/timeline")
    activities_dir: Path = Path("data/activities")

    # Boundary tuning
    max_gap_seconds_for_same_activity: float = 12.0
    strong_screen_change_threshold: float = 0.35
    weak_screen_change_threshold: float = 0.55
    min_activity_duration_seconds: float = 8.0
    min_segments_per_activity: int = 1

    # Text limits
    max_evidence_items: int = 12
    max_screen_text_per_step: int = 12


class ActivityDetectionService:
    """
    Generic MVP 7 activity detector.

    This service converts timeline JSON into activity JSON.

    It is intentionally domain-agnostic:
    - no IRCTC-specific logic
    - no app-specific workflow rules
    - no hard-coded activity names for a particular business process

    It uses generic signals:
    - speech intent
    - screen text changes
    - URL/page changes
    - transition phrases
    - time gaps
    - repeated UI states
    """

    def __init__(self, config: ActivityDetectionConfig | None = None) -> None:
        self.config = config or ActivityDetectionConfig()
        self.config.activities_dir.mkdir(parents=True, exist_ok=True)

    def detect_activities_for_job(self, job_id: str) -> dict[str, Any]:
        timeline_path = self.config.timeline_dir / f"{job_id}.json"
        if not timeline_path.exists():
            raise FileNotFoundError(f"Timeline not found: {timeline_path}")

        timeline = self._load_timeline(timeline_path)
        activities = self.detect_activities(job_id=job_id, timeline=timeline)

        output_path = self.config.activities_dir / f"{job_id}.json"
        self._write_json(output_path, activities)

        return activities

    def get_activities_for_job(self, job_id: str) -> dict[str, Any]:
        activities_path = self.config.activities_dir / f"{job_id}.json"
        if not activities_path.exists():
            raise FileNotFoundError(f"Activities not found: {activities_path}")

        with activities_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def detect_activities(
        self,
        job_id: str,
        timeline: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not timeline:
            return {
                "job_id": job_id,
                "activity_count": 0,
                "activities": [],
                "metadata": {
                    "detector": "generic_rule_based_v1",
                    "note": "No timeline segments found.",
                },
            }

        enriched_segments = [self._enrich_segment(i, segment) for i, segment in enumerate(timeline)]
        groups = self._group_segments_into_activities(enriched_segments)
        groups = self._merge_tiny_groups(groups)

        activities = [
            self._build_activity(activity_index=i, segments=group)
            for i, group in enumerate(groups, start=1)
        ]

        return {
            "job_id": job_id,
            "activity_count": len(activities),
            "activities": activities,
            "metadata": {
                "detector": "generic_rule_based_v1",
                "input_segments": len(timeline),
                "principle": "Generic activity detection using speech, OCR, time, and screen-state signals. No application-specific hard coding.",
            },
        }

    def _load_timeline(self, timeline_path: Path) -> list[dict[str, Any]]:
        with timeline_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Timeline JSON must be a list of timeline segments.")

        return data

    def _write_json(self, output_path: Path, data: dict[str, Any]) -> None:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _enrich_segment(self, index: int, segment: dict[str, Any]) -> dict[str, Any]:
        speech = str(segment.get("speech") or "")
        screen_text = segment.get("screen_text") or []

        if not isinstance(screen_text, list):
            screen_text = []

        clean_screen_text = [self._clean_text(str(item)) for item in screen_text]
        clean_screen_text = [item for item in clean_screen_text if item]

        speech_clean = self._clean_text(speech)
        speech_tokens = self._tokens(speech_clean)
        screen_tokens = self._tokens(" ".join(clean_screen_text))

        urls = self._extract_urls(clean_screen_text)
        page_state = self._detect_page_state(clean_screen_text, urls)
        intent = self._detect_intent(speech_clean, clean_screen_text)
        transition_score = self._transition_score(speech_clean)

        return {
            **segment,
            "_index": index,
            "_speech_clean": speech_clean,
            "_speech_tokens": speech_tokens,
            "_screen_text_clean": clean_screen_text,
            "_screen_tokens": screen_tokens,
            "_screen_signature": self._screen_signature(clean_screen_text),
            "_urls": urls,
            "_page_state": page_state,
            "_intent": intent,
            "_transition_score": transition_score,
        }

    def _group_segments_into_activities(
        self,
        segments: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        groups: list[list[dict[str, Any]]] = []
        current_group: list[dict[str, Any]] = []

        for i, segment in enumerate(segments):
            if i == 0:
                current_group.append(segment)
                continue

            previous = segments[i - 1]
            boundary_score, reasons = self._boundary_score(previous, segment)

            if boundary_score >= 1.0:
                current_group[-1]["_boundary_after_reasons"] = reasons
                groups.append(current_group)
                current_group = [segment]
            else:
                current_group.append(segment)

        if current_group:
            groups.append(current_group)

        return groups

    def _boundary_score(
        self,
        previous: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[float, list[str]]:
        score = 0.0
        reasons: list[str] = []

        previous_end = self._safe_float(previous.get("end_seconds"))
        current_start = self._safe_float(current.get("start_seconds"))
        gap = max(0.0, current_start - previous_end)

        if gap >= self.config.max_gap_seconds_for_same_activity:
            score += 0.55
            reasons.append(f"time_gap:{gap:.2f}")

        previous_page = previous.get("_page_state")
        current_page = current.get("_page_state")

        if previous_page and current_page and previous_page != current_page:
            score += 0.65
            reasons.append("page_state_changed")

        screen_similarity = self._jaccard_similarity(
            set(previous.get("_screen_signature", [])),
            set(current.get("_screen_signature", [])),
        )
        screen_change = 1.0 - screen_similarity

        if screen_change >= self.config.strong_screen_change_threshold:
            score += 0.45
            reasons.append(f"strong_screen_change:{screen_change:.2f}")
        elif screen_change >= self.config.weak_screen_change_threshold:
            score += 0.25
            reasons.append(f"weak_screen_change:{screen_change:.2f}")

        previous_intent = previous.get("_intent", {}).get("primary")
        current_intent = current.get("_intent", {}).get("primary")

        if previous_intent and current_intent and previous_intent != current_intent:
            score += 0.35
            reasons.append(f"intent_changed:{previous_intent}->{current_intent}")

        transition_score = current.get("_transition_score", 0.0)
        if transition_score >= 0.4:
            score += transition_score
            reasons.append(f"transition_cue:{transition_score:.2f}")

        # Avoid over-splitting when screen is stable and intent is similar.
        if screen_change < 0.15 and previous_intent == current_intent:
            score -= 0.4
            reasons.append("stable_screen_and_intent")

        return max(0.0, score), reasons

    def _merge_tiny_groups(
        self,
        groups: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        if not groups:
            return groups

        merged: list[list[dict[str, Any]]] = []

        for group in groups:
            duration = self._group_duration(group)
            is_tiny = (
                duration < self.config.min_activity_duration_seconds
                and len(group) <= self.config.min_segments_per_activity
            )

            if is_tiny and merged:
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    def _build_activity(
        self,
        activity_index: int,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        start_seconds = self._safe_float(segments[0].get("start_seconds"))
        end_seconds = self._safe_float(segments[-1].get("end_seconds"))

        intents = [segment.get("_intent", {}).get("primary") for segment in segments]
        intents = [intent for intent in intents if intent]
        dominant_intent = self._most_common(intents) or "workflow_step"

        activity_name = self._activity_name_from_intent(dominant_intent, segments)
        description = self._activity_description(activity_name, segments)

        steps = [
            self._build_step(step_index=i, segment=segment)
            for i, segment in enumerate(segments, start=1)
        ]

        evidence = self._activity_evidence(segments)
        confidence = self._activity_confidence(segments)

        return {
            "activity_id": f"activity_{activity_index:03d}",
            "name": activity_name,
            "description": description,
            "start_seconds": start_seconds,
            "end_seconds": end_seconds,
            "duration_seconds": round(max(0.0, end_seconds - start_seconds), 2),
            "confidence": confidence,
            "dominant_intent": dominant_intent,
            "evidence": evidence,
            "steps": steps,
        }

    def _build_step(
        self,
        step_index: int,
        segment: dict[str, Any],
    ) -> dict[str, Any]:
        intent = segment.get("_intent", {}).get("primary") or "observe"
        instruction = self._instruction_from_segment(intent, segment)

        screen_text = segment.get("_screen_text_clean", [])
        screen_text_sample = self._select_representative_screen_text(
            screen_text,
            limit=self.config.max_screen_text_per_step,
        )

        return {
            "step_number": step_index,
            "start_seconds": self._safe_float(segment.get("start_seconds")),
            "end_seconds": self._safe_float(segment.get("end_seconds")),
            "instruction": instruction,
            "intent": intent,
            "speech": segment.get("speech", ""),
            "screen_text_sample": screen_text_sample,
            "frame_path": segment.get("frame_path"),
            "nearest_frame_timestamp_seconds": segment.get("nearest_frame_timestamp_seconds"),
        }

    def _detect_intent(
        self,
        speech: str,
        screen_text: list[str],
    ) -> dict[str, Any]:
        combined = f"{speech} {' '.join(screen_text[:40])}".lower()

        intent_patterns: list[tuple[str, list[str]]] = [
            (
                "introduce_or_explain",
                [
                    "this is",
                    "recording",
                    "overview",
                    "as you can see",
                    "there are",
                    "i can see",
                    "it shows",
                    "explain",
                    "demo",
                    "demonstrate",
                ],
            ),
            (
                "open_or_navigate",
                [
                    "open",
                    "go to",
                    "navigate",
                    "access",
                    "homepage",
                    "login",
                    "log in",
                    "menu",
                    "option",
                    "page",
                    "screen",
                ],
            ),
            (
                "enter_information",
                [
                    "enter",
                    "type",
                    "fill",
                    "input",
                    "put",
                    "add",
                    "name",
                    "age",
                    "number",
                    "details",
                    "select my name",
                ],
            ),
            (
                "select_option",
                [
                    "select",
                    "choose",
                    "pick",
                    "check",
                    "checkbox",
                    "dropdown",
                    "category",
                    "preference",
                    "option",
                ],
            ),
            (
                "search_or_submit",
                [
                    "search",
                    "submit",
                    "hit",
                    "continue",
                    "proceed",
                    "go ahead",
                    "save",
                    "apply",
                    "confirm",
                ],
            ),
            (
                "review_results",
                [
                    "results",
                    "comes up",
                    "come up",
                    "available",
                    "availability",
                    "status",
                    "list",
                    "shows",
                    "displayed",
                ],
            ),
            (
                "compare_or_filter",
                [
                    "filter",
                    "sort",
                    "refine",
                    "from",
                    "to",
                    "class",
                    "type",
                    "time",
                    "arrival",
                    "departure",
                    "multiple",
                ],
            ),
            (
                "handle_confirmation",
                [
                    "confirmation",
                    "do you want",
                    "yes",
                    "no",
                    "ok",
                    "cancel",
                    "proceed",
                    "continue with",
                ],
            ),
            (
                "configure_preferences",
                [
                    "preference",
                    "upgradation",
                    "insurance",
                    "food",
                    "meal",
                    "coach",
                    "berth",
                    "birth",
                    "lower",
                    "upper",
                    "auto",
                ],
            ),
            (
                "payment_or_completion",
                [
                    "payment",
                    "pay",
                    "credit card",
                    "debit card",
                    "upi",
                    "net banking",
                    "wallet",
                    "fare",
                    "fee",
                    "complete",
                    "finish",
                    "stop here",
                ],
            ),
        ]

        scores: Counter[str] = Counter()

        for intent, patterns in intent_patterns:
            for pattern in patterns:
                if pattern in combined:
                    scores[intent] += 1

        if not scores:
            return {
                "primary": "observe",
                "scores": {},
            }

        primary, score = scores.most_common(1)[0]

        return {
            "primary": primary,
            "scores": dict(scores),
            "score": score,
        }

    def _activity_name_from_intent(
        self,
        intent: str,
        segments: list[dict[str, Any]],
    ) -> str:
        generic_names = {
            "introduce_or_explain": "Introduce or explain the workflow",
            "open_or_navigate": "Open or navigate within the application",
            "enter_information": "Enter required information",
            "select_option": "Select options",
            "search_or_submit": "Search, submit, or continue",
            "review_results": "Review displayed results or status",
            "compare_or_filter": "Filter, compare, or refine options",
            "handle_confirmation": "Handle confirmation prompt",
            "configure_preferences": "Configure preferences",
            "payment_or_completion": "Review payment or completion options",
            "observe": "Review current screen",
            "workflow_step": "Perform workflow step",
        }

        base_name = generic_names.get(intent, "Perform workflow step")

        # Add a generic page context when available, without using app-specific names.
        page_states = [segment.get("_page_state") for segment in segments if segment.get("_page_state")]
        page_state = self._most_common(page_states)

        if page_state and page_state not in {"unknown", "same_page"}:
            return f"{base_name} on {page_state}"

        return base_name

    def _activity_description(
        self,
        activity_name: str,
        segments: list[dict[str, Any]],
    ) -> str:
        speech_samples = [
            segment.get("_speech_clean", "")
            for segment in segments
            if segment.get("_speech_clean")
        ]
        speech_samples = [sample for sample in speech_samples if len(sample) > 8]

        if not speech_samples:
            return f"The user performs the activity: {activity_name}."

        summary_source = " ".join(speech_samples[:2])
        summary_source = self._truncate(summary_source, 220)

        return f"The user performs this part of the workflow. Speech context: {summary_source}"

    def _instruction_from_segment(
        self,
        intent: str,
        segment: dict[str, Any],
    ) -> str:
        speech = segment.get("_speech_clean", "")

        if intent == "introduce_or_explain":
            return "Introduce or explain the current part of the workflow."
        if intent == "open_or_navigate":
            return "Open the relevant page or navigate to the required section."
        if intent == "enter_information":
            return "Enter the required information into the visible fields."
        if intent == "select_option":
            return "Select the appropriate visible option."
        if intent == "search_or_submit":
            return "Search, submit, continue, or proceed using the visible control."
        if intent == "review_results":
            return "Review the displayed results, status, or available records."
        if intent == "compare_or_filter":
            return "Use filters or visible criteria to refine the displayed information."
        if intent == "handle_confirmation":
            return "Review the confirmation message and choose the appropriate response."
        if intent == "configure_preferences":
            return "Configure the visible preferences or optional settings."
        if intent == "payment_or_completion":
            return "Review payment, fee, or completion options before proceeding."

        if speech:
            return f"Review the screen while following the spoken instruction: {self._truncate(speech, 120)}"

        return "Review the current screen."

    def _activity_evidence(
        self,
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        speech_evidence = []
        screen_counter: Counter[str] = Counter()
        frame_paths = []

        for segment in segments:
            speech = segment.get("_speech_clean")
            if speech:
                speech_evidence.append(self._truncate(speech, 180))

            for text in segment.get("_screen_text_clean", []):
                if self._is_representative_screen_text(text):
                    screen_counter[text] += 1

            frame_path = segment.get("frame_path")
            if frame_path:
                frame_paths.append(frame_path)

        screen_evidence = [
            text for text, _count in screen_counter.most_common(self.config.max_evidence_items)
        ]

        return {
            "speech_samples": speech_evidence[: self.config.max_evidence_items],
            "screen_text_samples": screen_evidence,
            "frame_paths": frame_paths[:5],
        }

    def _activity_confidence(
        self,
        segments: list[dict[str, Any]],
    ) -> float:
        if not segments:
            return 0.0

        has_speech = sum(1 for segment in segments if segment.get("_speech_clean"))
        has_screen = sum(1 for segment in segments if segment.get("_screen_text_clean"))
        has_intent = sum(
            1
            for segment in segments
            if segment.get("_intent", {}).get("primary") not in {None, "observe"}
        )

        speech_score = has_speech / len(segments)
        screen_score = has_screen / len(segments)
        intent_score = has_intent / len(segments)

        confidence = 0.35 * speech_score + 0.35 * screen_score + 0.30 * intent_score

        return round(min(0.98, max(0.35, confidence)), 2)

    def _transition_score(self, speech: str) -> float:
        speech_lower = speech.lower()

        strong_cues = [
            "now",
            "next",
            "after that",
            "then finally",
            "finally",
            "once we",
            "so now",
            "go ahead",
            "continue",
            "proceed",
            "stop here",
        ]

        weak_cues = [
            "then",
            "so",
            "and then",
            "likewise",
            "for now",
        ]

        score = 0.0

        for cue in strong_cues:
            if cue in speech_lower:
                score += 0.35

        for cue in weak_cues:
            if cue in speech_lower:
                score += 0.12

        return min(score, 0.7)

    def _detect_page_state(
        self,
        screen_text: list[str],
        urls: list[str],
    ) -> str:
        if urls:
            normalized_url = self._normalize_url_for_state(urls[0])
            if normalized_url:
                return normalized_url

        # Fallback: infer a generic page state from repeated visible headings.
        candidate_headings = []
        for text in screen_text[:30]:
            if self._looks_like_heading(text):
                candidate_headings.append(text.lower())

        if candidate_headings:
            return self._slugify(" ".join(candidate_headings[:2]))

        return "unknown"

    def _normalize_url_for_state(self, url: str) -> str:
        url = url.lower().strip()
        url = re.sub(r"^https?://", "", url)
        url = re.sub(r"\?.*$", "", url)
        url = re.sub(r"#.*$", "", url)

        parts = [part for part in url.split("/") if part]
        if len(parts) >= 2:
            return self._slugify("/".join(parts[:3]))

        return self._slugify(url)

    def _extract_urls(self, screen_text: list[str]) -> list[str]:
        urls = []

        for text in screen_text:
            if "." in text and "/" in text:
                urls.append(text.strip())
            elif re.search(r"\b[a-zA-Z0-9.-]+\.(com|in|org|net|io)\b", text):
                urls.append(text.strip())

        return urls

    def _screen_signature(self, screen_text: list[str]) -> list[str]:
        tokens = []

        for text in screen_text:
            clean = self._clean_text(text).lower()
            if not self._is_representative_screen_text(clean):
                continue

            for token in self._tokens(clean):
                if len(token) >= 3:
                    tokens.append(token)

        counts = Counter(tokens)

        return [
            token
            for token, _count in counts.most_common(80)
        ]

    def _select_representative_screen_text(
        self,
        screen_text: list[str],
        limit: int,
    ) -> list[str]:
        selected = []

        for text in screen_text:
            if self._is_representative_screen_text(text):
                selected.append(text)

            if len(selected) >= limit:
                break

        return selected

    def _is_representative_screen_text(self, text: str) -> bool:
        text = self._clean_text(text)

        if not text:
            return False

        if len(text) <= 1:
            return False

        if text in {"-", "_", "|", "/", "\\", "^", "*", "©", "c", "o", "x"}:
            return False

        if re.fullmatch(r"[^\w]+", text):
            return False

        if re.fullmatch(r"\d+", text) and len(text) <= 2:
            return False

        return True

    def _looks_like_heading(self, text: str) -> bool:
        clean = self._clean_text(text)

        if len(clean) < 3:
            return False

        if len(clean) > 45:
            return False

        uppercase_ratio = sum(1 for char in clean if char.isupper()) / max(1, len(clean))
        word_count = len(clean.split())

        return uppercase_ratio > 0.35 or word_count <= 4

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def _clean_text(self, text: str) -> str:
        text = text.replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _safe_float(self, value: Any) -> float:
        try:
            number = float(value)
            if math.isnan(number) or math.isinf(number):
                return 0.0
            return round(number, 2)
        except (TypeError, ValueError):
            return 0.0

    def _jaccard_similarity(self, left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0

        if not left or not right:
            return 0.0

        return len(left & right) / len(left | right)

    def _most_common(self, values: list[str]) -> str | None:
        if not values:
            return None

        return Counter(values).most_common(1)[0][0]

    def _group_duration(self, group: list[dict[str, Any]]) -> float:
        if not group:
            return 0.0

        start = self._safe_float(group[0].get("start_seconds"))
        end = self._safe_float(group[-1].get("end_seconds"))

        return max(0.0, end - start)

    def _truncate(self, text: str, limit: int) -> str:
        text = self._clean_text(text)

        if len(text) <= limit:
            return text

        return text[: limit - 3].rstrip() + "..."

    def _slugify(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_") or "unknown"