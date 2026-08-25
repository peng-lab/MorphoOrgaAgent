"""Final report agent for organoid microscopy image analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from MorphoOrgaAgent.core.state import OrganoidState


class ReportAgent:
    """Generate an evidence-grounded markdown report from OrganoidState."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.4",
        temperature: float = 0.0,
    ) -> None:
        """Initialize the native OpenAI SDK client.

        The API key is intentionally optional. If no usable key/client is
        available, report generation falls back to a conservative local
        markdown summary built only from OrganoidState.
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

    def generate_report(
        self,
        organoid_state: OrganoidState,
        report_instruction: str,
    ) -> OrganoidState:
        """Generate a markdown scientific report and write it to state.

        This method does not re-plan, re-run segmentation, modify masks, or
        modify any metrics. It only reads OrganoidState, asks the model to
        synthesize a report, and stores the markdown in ``state.final_report``.
        """
        context = self._build_report_context(organoid_state)
        instruction = report_instruction or organoid_state.report_instruction
        fallback = self._fallback_report(context, instruction)

        if self.client is None:
            report = fallback
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._system_prompt()},
                        {
                            "role": "user",
                            "content": self._user_prompt(context, instruction),
                        },
                    ],
                    temperature=self._temperature,
                )
                report = response.choices[0].message.content or fallback
            except Exception:
                report = fallback

        organoid_state.final_report = report.strip()
        organoid_state.append_log(
            "ReportAgent",
            "Generated final markdown report from OrganoidState evidence.",
        )
        return organoid_state

    def run(
        self,
        organoid_state: OrganoidState,
        report_instruction: str,
    ) -> OrganoidState:
        """Alias for pipeline-style agent execution."""
        return self.generate_report(organoid_state, report_instruction)

    def _build_report_context(self, state: OrganoidState) -> dict[str, Any]:
        """Build a compact, JSON-safe evidence package from OrganoidState."""
        metric_values = self._json_safe(state.metric_values)
        object_count = len(metric_values) if isinstance(metric_values, dict) else None

        return {
            "original_user_request": self._first_available(
                state,
                "user_request",
                "original_user_request",
                "user_input",
                "query",
            ),
            "task_understanding": {
                "target_objects": self._json_safe(state.target_objects),
                "required_metrics": self._json_safe(state.required_metrics),
                "required_visualizations": self._json_safe(
                    state.required_visualizations
                ),
                "confidence": state.confidence,
                "requires_segmentation": state.requires_segmentation,
                "intent": self._first_available(state, "intent", "task_intent"),
                "task_summary": self._first_available(
                    state,
                    "task_summary",
                    "task_understanding",
                    "analysis_intent",
                ),
            },
            "analysis_plan": self._first_available(
                state,
                "analysis_plan",
                "execution_plan",
                "plan",
            ),
            "raw_inputs": {
                "image_path": state.image_path,
                "raw_image": self._array_summary(state.raw_image),
                "metadata": self._json_safe(state.metadata),
            },
            "segmentation_summary": {
                "mask_path": state.mask_path or "unavailable",
                "segmentation_masks": self._array_summary(state.segmentation_masks),
                "object_count_from_metric_values": object_count,
            },
            "measurement_summary": {
                "csv_path": state.csv_path or "unavailable",
                "metric_names": self._metric_names(metric_values),
                "object_count": object_count,
            },
            "per_object_metrics": metric_values,
            "visualization_paths": self._json_safe(state.generated_plots),
            "warnings": self._collect_by_keywords(state.logs, ("warning", "warn")),
            "limitations": self._collect_limitations(state),
            "logs": self._json_safe(state.logs[-20:]),
        }

    @staticmethod
    def _load_system_prompt() -> str:
        prompt_path = (
            Path(__file__).resolve().parent
            / "prompts"
            / "report_agent_system.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()

    def _system_prompt(self) -> str:
        return self.system_prompt

    @staticmethod
    def _user_prompt(context: dict[str, Any], report_instruction: str) -> str:
        context_json = json.dumps(context, ensure_ascii=False, indent=2)
        return (
            "Report instruction to answer directly:\n"
            f"{report_instruction or 'No explicit report_instruction was provided.'}\n\n"
            "Compact OrganoidState report context:\n"
            f"{context_json}\n\n"
            "Write the final markdown scientific report now."
        )

    @staticmethod
    def _fallback_report(
        context: dict[str, Any],
        report_instruction: str,
    ) -> str:
        task = context.get("task_understanding", {})
        raw_inputs = context.get("raw_inputs", {})
        segmentation = context.get("segmentation_summary", {})
        measurements = context.get("measurement_summary", {})
        visualizations = context.get("visualization_paths", {})
        limitations = context.get("limitations", [])

        return "\n".join(
            [
                "# Organoid Analysis Report",
                "",
                "## Report Instruction",
                report_instruction or "No explicit report instruction was provided.",
                "",
                "## Evidence Summary",
                f"- Original image path: {raw_inputs.get('image_path') or 'unavailable'}",
                f"- Target objects: {task.get('target_objects') or 'unavailable'}",
                f"- Required metrics: {task.get('required_metrics') or 'unavailable'}",
                f"- Required visualizations: {task.get('required_visualizations') or 'unavailable'}",
                f"- Mask path: {segmentation.get('mask_path') or 'unavailable'}",
                f"- Object count: {measurements.get('object_count') if measurements.get('object_count') is not None else 'unavailable'}",
                f"- Metric names: {measurements.get('metric_names') or 'unavailable'}",
                f"- CSV path: {measurements.get('csv_path') or 'unavailable'}",
                f"- Visualization paths: {visualizations or 'unavailable'}",
                "",
                "## Limitations and Uncertainty",
                self_join(limitations) if limitations else "- No explicit limitations were recorded in OrganoidState.",
                "",
                "## Note",
                "The OpenAI model was unavailable, so this conservative fallback report only summarizes evidence already present in OrganoidState.",
            ]
        )

    @staticmethod
    def _array_summary(value: Any) -> dict[str, Any] | str:
        if value is None:
            return "unavailable"
        if isinstance(value, np.ndarray):
            return {
                "type": "numpy.ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": ReportAgent._safe_array_stat(value, "min"),
                "max": ReportAgent._safe_array_stat(value, "max"),
            }
        return ReportAgent._json_safe(value)

    @staticmethod
    def _safe_array_stat(value: np.ndarray, stat_name: str) -> Any:
        try:
            if value.size == 0:
                return "unavailable"
            if stat_name == "min":
                result = np.nanmin(value)
            elif stat_name == "max":
                result = np.nanmax(value)
            else:
                return "unavailable"
            return result.item() if hasattr(result, "item") else result
        except Exception:
            return "unavailable"

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if is_dataclass(value):
            return ReportAgent._json_safe(asdict(value))
        if isinstance(value, np.ndarray):
            return ReportAgent._array_summary(value)
        if isinstance(value, dict):
            return {
                str(key): ReportAgent._json_safe(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [ReportAgent._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _metric_names(metric_values: Any) -> list[str]:
        if not isinstance(metric_values, dict):
            return []

        names: set[str] = set()
        for object_metrics in metric_values.values():
            if isinstance(object_metrics, dict):
                names.update(str(name) for name in object_metrics.keys())
        return sorted(names)

    @staticmethod
    def _first_available(state: OrganoidState, *names: str) -> Any:
        for name in names:
            if hasattr(state, name):
                value = getattr(state, name)
                if value not in (None, "", [], {}):
                    return ReportAgent._json_safe(value)
        return "unavailable"

    @staticmethod
    def _collect_by_keywords(logs: list[str], keywords: tuple[str, ...]) -> list[str]:
        return [
            log
            for log in logs
            if any(keyword in log.lower() for keyword in keywords)
        ]

    @staticmethod
    def _collect_limitations(state: OrganoidState) -> list[str]:
        explicit = ReportAgent._first_available(state, "limitations", "uncertainties")
        if explicit != "unavailable":
            if isinstance(explicit, list):
                return explicit
            return [str(explicit)]

        limitation_logs = ReportAgent._collect_by_keywords(
            state.logs,
            ("limitation", "uncertain", "failed", "fallback", "unavailable"),
        )
        return limitation_logs


def self_join(items: list[Any]) -> str:
    """Render fallback bullet lines without changing evidence content."""
    return "\n".join(f"- {item}" for item in items)
