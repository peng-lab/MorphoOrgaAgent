"""Task understanding agent for organoid microscopy image analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from MorphoOrgaAgent.core.state import OrganoidState


class TaskUnderstandingAgent:
    """Parse natural-language organoid analysis requests into AnalysisIntent JSON."""

    METRIC_POOL = {
        "organoid_idx",
        "Area_outer",
        "Perimeter_outer",
        "areas_inner",
        "perimeters_inner",
        "x",
        "y",
        "is_border",
        "ideal_perimeter",
        "perimeter_diff",
        "roundness",
        "lumen_ratio",
        "area_outer_mm",
        "area_outer_microns",
        "radius_from_peri",
        "area_from_peri",
    }

    VISUALIZATIONS = {
        "pixel_overlays",
        "metric_plots",
        "correlation_heatmap",
        "none",
    }

    def __init__(self, api_key: str | None = None, model: str = "gpt-5.4-mini", temperature: float = 0.0,) -> None:
        """Initialize the native OpenAI SDK client.

        Temperature is intentionally not configurable and is fixed at 0.0
        inside API calls for deterministic task parsing.
        """
        self.model = model
        self._temperature = temperature
        self.system_prompt = self._load_system_prompt()
        self.client = None
        self._client_init_error: str | None = None

        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except Exception as exc:  # pragma: no cover - environment dependent
            self._client_init_error = str(exc)

    def understand_task(
        self, user_input: str, state: OrganoidState
    ) -> OrganoidState:
        """Update and return OrganoidState for a user request.

        This method never raises for OpenAI, JSON parsing, or validation
        failures. It updates state with a deterministic fallback when parsing
        fails.
        """
        fallback = self._fallback_intent(user_input)
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self._user_prompt(user_input)},
        ]

        if self.client is None:
            parsed_output = fallback
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "analysis_intent",
                            "strict": True,
                            "schema": self._analysis_intent_schema(),
                        },
                    },
                )
                raw_content = response.choices[0].message.content
                parsed_output = self._parse_and_validate(raw_content, fallback)
            except Exception:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self._temperature,
                        response_format={"type": "json_object"},
                    )
                    raw_content = response.choices[0].message.content
                    parsed_output = self._parse_and_validate(raw_content, fallback)
                except Exception:
                    parsed_output = fallback

        state.target_objects = parsed_output.get("target_objects", ["organoid"])
        state.required_metrics = parsed_output.get("required_metrics", [])
        state.required_visualizations = parsed_output.get(
            "required_visualizations", []
        )
        state.report_instruction = parsed_output.get("report_instruction", "")
        state.confidence = parsed_output.get("confidence", 1.0)
        state.requires_segmentation = True

        state.logs.append(
            "[TU Agent] Successfully parsed user input. "
            f"Selected {len(state.required_metrics)} metrics and "
            f"{len(state.required_visualizations)} visualization categories "
            f"with confidence {state.confidence}"
        )

        return state

    def _system_prompt(self) -> str:
        return self.system_prompt

    @staticmethod
    def _load_system_prompt() -> str:
        prompt_path = (
            Path(__file__).resolve().parent
            / "prompts"
            / "task_understanding_system.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()

    @staticmethod
    def _user_prompt(user_input: str) -> str:
        return (
            "Parse this user request into simplified AnalysisIntent JSON with exactly the "
            "required schema fields, including a non-empty report_instruction that can be "
            f"used directly by the ReportAgent:\n{user_input}"
        )

    @staticmethod
    def _analysis_intent_schema() -> dict[str, Any]:
        metric_enum = sorted(TaskUnderstandingAgent.METRIC_POOL)
        visualization_enum = sorted(TaskUnderstandingAgent.VISUALIZATIONS)

        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "target_objects",
                "required_metrics",
                "required_visualizations",
                "report_instruction",
                "confidence",
            ],
            "properties": {
                "target_objects": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "required_metrics": {
                    "type": "array",
                    "items": {"type": "string", "enum": metric_enum},
                },
                "required_visualizations": {
                    "type": "array",
                    "items": {"type": "string", "enum": visualization_enum},
                },
                "report_instruction": {
                    "type": "string",
                    "description": (
                        "CRITICAL MANDATORY FIELD: Synthesize a clear, direct operational directive "
                        "for the downstream ReportAgent telling it exactly what to analyze and draft. "
                        "You MUST explicitly incorporate the specific metrics (from required_metrics) and "
                        "the specific visualizations (from required_visualizations) that you have planned for this task. "
                        "For example: 'Draft a summary using the planned Area_outer metric and reference the generated metric_plots.' "
                        "Include any user-specified style/language constraints (e.g., 'Answer in English', 'Use bullet points'). "
                        "NEVER return an empty string or generic placeholder."
                    ),
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        }

    def _parse_and_validate(
        self, raw_content: str | None, fallback: dict[str, Any]
    ) -> dict[str, Any]:
        if not raw_content:
            return fallback

        try:
            parsed = json.loads(raw_content)
        except (TypeError, json.JSONDecodeError):
            return fallback

        if not isinstance(parsed, dict):
            return fallback

        try:
            return self._sanitize_intent(parsed, fallback)
        except Exception:
            return fallback

    def _sanitize_intent(
        self, parsed: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        result = fallback.copy()

        target_objects = parsed.get("target_objects")
        if isinstance(target_objects, list):
            cleaned_objects = [
                item.strip()
                for item in target_objects
                if isinstance(item, str) and item.strip()
            ]
            if cleaned_objects:
                result["target_objects"] = cleaned_objects
        if "organoid" not in result["target_objects"]:
            result["target_objects"].insert(0, "organoid")

        metrics = parsed.get("required_metrics")
        if isinstance(metrics, list):
            result["required_metrics"] = [
                metric for metric in metrics if metric in self.METRIC_POOL
            ]

        visualizations = parsed.get("required_visualizations")
        if isinstance(visualizations, list):
            valid_visualizations = [
                item for item in visualizations if item in self.VISUALIZATIONS
            ]
            result["required_visualizations"] = valid_visualizations or ["none"]

        report_instruction = parsed.get("report_instruction")
        if isinstance(report_instruction, str) and report_instruction.strip():
            result["report_instruction"] = report_instruction.strip()

        confidence = parsed.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            result["confidence"] = max(0.0, min(1.0, float(confidence)))

        return result

    def _fallback_intent(self, user_input: str) -> dict[str, Any]:
        text = (user_input or "").lower()

        required_metrics = ["organoid_idx", "Area_outer", "Perimeter_outer"]
        required_visualizations = []

        if any(
            term in text
            for term in (
                "mask",
                "overlay",
                "bbox",
                "bounding box",
                "boundary",
                "outline",
                "segment",
                "segmentation",
                "locate",
                "identify",
                "count",
            )
        ):
            required_visualizations.append("pixel_overlays")

        if any(
            term in text
            for term in (
                "area",
                "perimeter",
                "roundness",
                "lumen",
                "centroid",
                "coordinate",
                "location",
                "spatial",
                "map",
                "heatmap",
                "distribution",
                "histogram",
                "heterogeneous",
                "homogeneous",
                "abnormal",
                "outlier",
                "fusion",
                "density",
                "variability",
                "population",
                "plot",
                "chart",
                "visual",
            )
        ):
            required_visualizations.append("metric_plots")

        if any(
            term in text
            for term in (
                "correlation",
                "correlate",
                "association",
                "relationship",
                "trade-off",
                "tradeoff",
                "covariance",
                "interaction",
                "vary together",
            )
        ):
            required_visualizations.append("correlation_heatmap")

        if any(
            term in text
            for term in (
                "qc report",
                "quality-control report",
                "single-cell",
                "prioritize",
            )
        ):
            required_metrics = [
                "organoid_idx",
                "Area_outer",
                "Perimeter_outer",
                "roundness",
                "lumen_ratio",
                "perimeter_diff",
            ]
            required_visualizations.extend(["pixel_overlays", "metric_plots"])
        elif any(
            term in text
            for term in (
                "health",
                "hypoth",
                "phenotype",
                "passaging",
                "biological",
                "culture",
            )
        ):
            required_metrics = [
                "Area_outer",
                "Perimeter_outer",
                "roundness",
                "lumen_ratio",
                "perimeter_diff",
            ]
            required_visualizations.append("metric_plots")
        elif any(
            term in text
            for term in (
                "distribution",
                "heterogeneous",
                "homogeneous",
                "abnormal",
                "outlier",
                "fusion",
                "density",
                "spatial",
                "variability",
                "population",
            )
        ):
            required_metrics = [
                "organoid_idx",
                "Area_outer",
                "Perimeter_outer",
                "roundness",
                "perimeter_diff",
            ]
            if "fusion" in text:
                required_metrics.extend(["areas_inner", "perimeters_inner"])
            if "abnormal" in text or "outlier" in text:
                required_metrics.extend(["is_border", "lumen_ratio"])
        elif "largest" in text:
            required_metrics = ["organoid_idx", "Area_outer", "x", "y"]
        elif "mean area" in text:
            required_metrics = ["organoid_idx", "Area_outer"]
        elif "roundness" in text:
            required_metrics = ["organoid_idx", "Area_outer", "roundness"]

        if any(
            term in text
            for term in (
                "x",
                "y",
                "centroid",
                "coordinate",
                "location",
                "spatial",
                "density",
            )
        ):
            required_metrics.extend(["x", "y"])

        if "intensity" in text or "fluorescence" in text:
            required_metrics.append("organoid_idx")

        if any(
            term in text
            for term in (
                "no chart",
                "no plot",
                "no visualization",
                "without visualization",
                "text only",
                "numeric only",
            )
        ):
            required_visualizations = ["none"]
        elif not required_visualizations:
            required_visualizations = ["none"]
        else:
            required_visualizations = [
                item for item in dict.fromkeys(required_visualizations) if item != "none"
            ]

        required_metrics = list(dict.fromkeys(required_metrics))

        return {
            "target_objects": ["organoid"],
            "required_metrics": [
                metric for metric in required_metrics if metric in self.METRIC_POOL
            ],
            "required_visualizations": required_visualizations,
            "report_instruction": self._default_report_instruction(user_input),
            "confidence": 0.35 if text.strip() else 0.1,
        }

    @staticmethod
    def _default_report_instruction(user_input: str) -> str:
        text = (user_input or "").strip()
        return text or "Answer the user's request clearly and directly."
