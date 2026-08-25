"""Reusable orchestration for the complete organoid-analysis pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from MorphoOrgaAgent.agents.report_agent import ReportAgent
from MorphoOrgaAgent.agents.task_understanding_agent import TaskUnderstandingAgent
from MorphoOrgaAgent.core.state import OrganoidState
from MorphoOrgaAgent.tools import morphology_tools as morph
from MorphoOrgaAgent.tools import segmentation_tools as segmentation
from MorphoOrgaAgent.tools import visualization_tools as viz


TEXT_PROMPT = segmentation.DEFAULT_TEXT_PROMPT


def announce(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def make_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the standard artifact directories."""
    dirs = {
        "root": output_dir,
        "cellpose": output_dir / "cellpose",
        "sam3": output_dir / "sam3",
        "metrics": output_dir / "metrics",
        "visualizations": output_dir / "visualizations",
        "debug": output_dir / "debug",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_task_understanding(
    state: OrganoidState,
    query: str,
    api_key: str | None,
) -> OrganoidState:
    agent = TaskUnderstandingAgent(api_key=api_key)
    return agent.understand_task(user_input=query, state=state)


def run_fixed_segmentation(
    state: OrganoidState,
    dirs: dict[str, Path],
    confidence_threshold: float,
    min_cellpose_area: int,
    device: str | None,
) -> OrganoidState:
    """Run the fixed Cellpose-to-SAM3 segmentation cascade."""
    import torch

    image_path = Path(state.image_path)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    announce(f"Calling Cellpose for coarse segmentation on {image_path}...")
    cellpose_mask, cellpose_path = segmentation.run_cellpose_single_image(
        image_path,
        dirs["cellpose"],
    )
    cellpose_count = int(cellpose_mask.max(initial=0))
    announce(f"Cellpose found {cellpose_count} coarse objects; mask saved to {cellpose_path}")
    state.append_log("Segmentation", f"Saved Cellpose coarse mask to {cellpose_path}")

    announce(
        f"Calling SAM3 refinement on {selected_device} using text prompt '{TEXT_PROMPT}'..."
    )
    refined_mask, refined_path = segmentation.run_sam3_refinement_single_image(
        image_path=image_path,
        cellpose_mask=cellpose_mask,
        output_dir=dirs["sam3"],
        confidence_threshold=confidence_threshold,
        min_cellpose_area=min_cellpose_area,
        device=selected_device,
        text_prompt=TEXT_PROMPT,
    )
    refined_count = int(refined_mask.max(initial=0))
    announce(f"SAM3 refinement found {refined_count} organoids; mask saved to {refined_path}")
    state.segmentation_masks = refined_mask.astype(np.int32, copy=False)
    state.mask_path = str(refined_path)
    state.append_log(
        "Segmentation",
        f"Saved SAM3 refined mask to {refined_path} using text prompt '{TEXT_PROMPT}'",
    )
    return state


def compute_requested_metrics(
    state: OrganoidState,
    metrics_dir: Path,
) -> pd.DataFrame:
    """Compute requested morphology metrics and save JSON/CSV artifacts."""
    if state.segmentation_masks is None:
        raise ValueError("OrganoidState.segmentation_masks is required before metrics.")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    mask = state.segmentation_masks.astype(np.int32, copy=False)
    requested = [name for name in state.required_metrics if name]
    if not requested:
        requested = ["organoid_idx"]
    if "organoid_idx" not in requested:
        requested = ["organoid_idx", *requested]

    cache: dict[str, list[Any]] = {}
    skipped: list[str] = []

    def get_metric(name: str) -> list[Any] | None:
        if name in cache:
            return cache[name]

        if name == "organoid_idx":
            cache[name] = morph.get_organoid_ids(mask)
        elif name == "Area_outer":
            cache[name] = morph.calculate_area_outer(mask)
        elif name == "Perimeter_outer":
            cache[name] = morph.calculate_perimeter_outer(mask)
        elif name in {"areas_inner", "perimeters_inner"}:
            areas_inner, perimeters_inner = morph.calculate_inner_hole_metrics(mask)
            cache["areas_inner"] = areas_inner
            cache["perimeters_inner"] = perimeters_inner
        elif name in {"x", "y"}:
            x_coords, y_coords = morph.calculate_centroids(mask)
            cache["x"] = x_coords
            cache["y"] = y_coords
        elif name == "is_border":
            cache[name] = morph.check_is_border(mask)
        elif name == "ideal_perimeter":
            cache[name] = morph.calculate_ideal_perimeter(
                mask,
                area_outer=get_metric("Area_outer"),
            )
        elif name == "perimeter_diff":
            cache[name] = morph.calculate_perimeter_diff(
                mask,
                perimeter_outer=get_metric("Perimeter_outer"),
                ideal_perimeter=get_metric("ideal_perimeter"),
            )
        elif name == "roundness":
            cache[name] = morph.calculate_roundness(
                mask,
                area_outer=get_metric("Area_outer"),
                perimeter_outer=get_metric("Perimeter_outer"),
            )
        elif name == "lumen_ratio":
            cache[name] = morph.calculate_lumen_ratio(
                mask,
                area_outer=get_metric("Area_outer"),
                area_inner=get_metric("areas_inner"),
            )
        elif name == "area_outer_mm":
            cache[name] = morph.calculate_area_outer_mm(
                mask,
                area_outer=get_metric("Area_outer"),
            )
        elif name in {"area_outer_microns", "area_outer_mikrons"}:
            area_mm = morph.calculate_area_outer_mm(
                mask,
                area_outer=get_metric("Area_outer"),
            )
            cache[name] = [float(value * 1_000_000) for value in area_mm]
        elif name in {"radius_from_peri", "area_from_peri"}:
            radii, areas = morph.calculate_metrics_from_perimeter(
                mask,
                perimeter_outer=get_metric("Perimeter_outer"),
            )
            cache["radius_from_peri"] = radii
            cache["area_from_peri"] = areas
        else:
            skipped.append(name)
            return None

        return cache[name]

    data: dict[str, list[Any]] = {}
    for metric_name in requested:
        values = get_metric(metric_name)
        if values is not None:
            data[metric_name] = values

    dataframe = pd.DataFrame(data)
    state.metric_values = dataframe_to_state_metrics(dataframe)

    json_path = metrics_dir / "metric_values.json"
    csv_path = metrics_dir / "metric_values.csv"
    json_path.write_text(json.dumps(state.metric_values, indent=2), encoding="utf-8")
    dataframe.to_csv(csv_path, index=False)
    state.csv_path = str(csv_path)

    if skipped:
        state.append_log(
            "Metrics",
            f"Skipped unsupported metrics: {', '.join(sorted(set(skipped)))}",
        )
    state.append_log("Metrics", f"Saved metric values to {json_path} and {csv_path}")
    announce(f"Metrics saved to {json_path} and {csv_path}")
    return dataframe


def dataframe_to_state_metrics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    metric_values: dict[str, dict[str, Any]] = {}
    if dataframe.empty:
        return metric_values

    for row_idx, row in dataframe.iterrows():
        object_id = row.get("organoid_idx", row_idx + 1)
        object_key = f"organoid_{int(object_id)}"
        metric_values[object_key] = {
            str(column): json_safe(value) for column, value in row.items()
        }
    return metric_values


def generate_requested_visualizations(
    state: OrganoidState,
    metrics_dataframe: pd.DataFrame,
    visualizations_dir: Path,
) -> OrganoidState:
    """Generate only the visualization categories requested by the agent."""
    import cv2

    if state.segmentation_masks is None:
        raise ValueError(
            "OrganoidState.segmentation_masks is required before visualization."
        )

    requested = set(state.required_visualizations)
    requested.discard("none")
    if not requested:
        state.append_log("Visualization", "No visualizations requested.")
        announce("No visualizations requested.")
        return state

    visualizations_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(state.image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image for visualization: {state.image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    generated = dict(state.generated_plots)
    mask = state.segmentation_masks

    if "pixel_overlays" in requested:
        overlay_path = visualizations_dir / "mask_overlay.png"
        bbox_path = visualizations_dir / "bbox_overlay.png"
        viz.plot_mask_overlay(image_rgb, mask, str(overlay_path))
        viz.plot_bbox_overlay(image_rgb, mask, str(bbox_path))
        generated["mask_overlay"] = str(overlay_path)
        generated["bbox_overlay"] = str(bbox_path)

        for metric_name in preferred_heatmap_metrics(metrics_dataframe):
            metric_path = visualizations_dir / f"metric_heatmap_{metric_name}.png"
            viz.plot_metric_heatmap_overlay(
                image_rgb,
                mask,
                metrics_dataframe[metric_name].tolist(),
                metric_name,
                str(metric_path),
            )
            generated[f"metric_heatmap_{metric_name}"] = str(metric_path)

    if "metric_plots" in requested:
        for metric_name in numeric_metric_columns(metrics_dataframe):
            metric_path = visualizations_dir / f"metric_distribution_{metric_name}.png"
            viz.plot_metric_distribution(
                metric_name,
                metrics_dataframe[metric_name].dropna().tolist(),
                str(metric_path),
            )
            generated[f"metric_distribution_{metric_name}"] = str(metric_path)

    if "correlation_heatmap" in requested:
        correlation_path = visualizations_dir / "correlation_heatmap.png"
        viz.plot_correlation_heatmap(metrics_dataframe, str(correlation_path))
        generated["correlation_heatmap"] = str(correlation_path)

    state.generated_plots = generated
    state.append_log(
        "Visualization",
        f"Saved {len(generated)} visualization artifact paths.",
    )
    announce(f"Saved {len(generated)} visualization artifact paths.")
    return state


def numeric_metric_columns(dataframe: pd.DataFrame) -> list[str]:
    excluded = {"organoid_idx", "x", "y"}
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    return [column for column in numeric_columns if column not in excluded]


def preferred_heatmap_metrics(dataframe: pd.DataFrame) -> list[str]:
    candidates = ["Area_outer", "roundness", "lumen_ratio", "perimeter_diff"]
    return [name for name in candidates if name in dataframe.columns]


def run_report_agent(
    state: OrganoidState,
    api_key: str | None,
    output_dir: Path,
) -> OrganoidState:
    announce("Calling report agent...")
    report_agent = ReportAgent(api_key=api_key)
    state = report_agent.generate_report(
        organoid_state=state,
        report_instruction=state.report_instruction,
    )
    report_path = output_dir / "final_report.md"
    report_path.write_text(state.final_report, encoding="utf-8")
    state.generated_plots.setdefault("final_report", str(report_path))
    state.append_log("ReportAgent", f"Saved final report to {report_path}")
    announce(f"Final report saved to {report_path}")
    return state


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        summary: dict[str, Any] = {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size:
            summary["min"] = json_safe(np.nanmin(value))
            summary["max"] = json_safe(np.nanmax(value))
        else:
            summary["min"] = "unavailable"
            summary["max"] = "unavailable"
        return summary
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_state_json(state: OrganoidState, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(json_safe(state), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target_path


def run_pipeline(
    image_path: str | Path,
    query: str,
    output_dir: str | Path,
    *,
    api_key: str | None = None,
    sam3_confidence_threshold: float = 0.5,
    min_cellpose_area: int = 1,
    device: str | None = None,
) -> OrganoidState:
    """Run all pipeline stages and return the completed shared state."""
    resolved_image_path = Path(image_path).expanduser().resolve()
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {resolved_image_path}")

    announce("Starting organoid analysis pipeline.")
    announce(f"Input image: {resolved_image_path}")
    announce(f"Output directory: {resolved_output_dir}")
    dirs = make_output_dirs(resolved_output_dir)
    state = OrganoidState(
        image_path=str(resolved_image_path),
        output_dir=str(resolved_output_dir),
    )

    state.append_log("Pipeline", "Initialized OrganoidState.")
    announce("Organoid state initialized successfully.")
    announce("Understanding task request...")
    state = run_task_understanding(state, query, api_key)
    announce(
        "Task understood: "
        f"metrics={state.required_metrics}, "
        f"visualizations={state.required_visualizations}"
    )
    announce("Starting segmentation...")
    state = run_fixed_segmentation(
        state,
        dirs,
        confidence_threshold=sam3_confidence_threshold,
        min_cellpose_area=min_cellpose_area,
        device=device,
    )
    announce("Computing requested metrics...")
    metrics_dataframe = compute_requested_metrics(state, dirs["metrics"])
    announce(
        f"Computed {len(metrics_dataframe.columns)} metrics for "
        f"{len(metrics_dataframe.index)} organoids."
    )
    announce("Generating requested visualizations...")
    state = generate_requested_visualizations(
        state,
        metrics_dataframe,
        dirs["visualizations"],
    )
    state = run_report_agent(state, api_key, resolved_output_dir)

    final_state_path = save_state_json(
        state,
        dirs["debug"] / "organoid_state_final.json",
    )
    announce(f"Pipeline complete. Final debug state saved to {final_state_path}")
    return state
