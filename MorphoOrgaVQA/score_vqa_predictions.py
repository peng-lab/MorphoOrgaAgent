#!/usr/bin/env python3
"""Score MorphoOrga-VQA predictions against deterministic ground truth.

Prediction files are matched to ground-truth masks by filename stem, so an
image key such as ``sample.jpg`` matches ``sample.npy``. Missing images, slots,
modes, or numeric fields receive zero credit and remain visible in the detailed
CSV rather than being silently excluded.

Example:
    python score_vqa_predictions.py \
        --predictions vqa_agent_predictions.json \
        --groundtruth vqa_groundtruth_answers.json \
        --output-dir scoring_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


MODES = ("clear", "open")
SLOT_TYPES = ("max", "tertile_mean")
GT_NEAR_ZERO_EPSILON = 1e-6
ABS_ERROR_FALLBACK_THRESHOLD = 1e-3
SCORING_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "max": (15.0, 30.0),
    "tertile_mean": (15.0, 30.0),
}

RECORD_FIELDS = [
    "image",
    "category",
    "slot_type",
    "mode",
    "field",
    "gt_value",
    "pred_value",
    "pct_error",
    "abs_error",
    "credit",
    "status",
    "unparsable",
]


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON object and reject non-object top-level payloads."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at the top level: {path}")
    return payload


def stem(filename: str) -> str:
    """Strip the extension so image and mask filenames can be matched."""
    return Path(filename).stem


def index_by_stem(payload: Mapping[str, Any], label: str) -> Dict[str, Any]:
    """Index a filename-keyed mapping and reject ambiguous duplicate stems."""
    indexed: Dict[str, Any] = {}
    source_names: Dict[str, str] = {}
    for filename, value in payload.items():
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"{label} contains a non-string or empty filename key")
        key = stem(filename)
        if key in indexed:
            raise ValueError(
                f"{label} contains duplicate filename stem {key!r}: "
                f"{source_names[key]!r} and {filename!r}"
            )
        indexed[key] = value
        source_names[key] = filename
    return indexed


def slot_type_of(slot_key: str) -> str:
    slot_type = slot_key.rsplit("__", 1)[-1]
    if slot_type not in SLOT_TYPES:
        raise ValueError(
            f"Unsupported slot type {slot_type!r} in {slot_key!r}; "
            f"expected one of {SLOT_TYPES}"
        )
    return slot_type


def category_of(slot_key: str) -> str:
    return slot_key.rsplit("__", 1)[0]


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def extract_gt_fields(gt_value: Any) -> Dict[str, float]:
    """Normalize one ground-truth slot to ``{field_name: finite_float}``."""
    if isinstance(gt_value, dict):
        fields: Dict[str, float] = {}
        for name, value in gt_value.items():
            if isinstance(name, str):
                numeric = _finite_float(value)
                if numeric is not None:
                    fields[name] = numeric
        return fields
    numeric = _finite_float(gt_value)
    return {"value": numeric} if numeric is not None else {}


def extract_pred_fields(final_answer: Any) -> Optional[Dict[str, float]]:
    """Normalize a prediction's ``final_answer`` object."""
    if not isinstance(final_answer, dict):
        return None
    fields: Dict[str, float] = {}
    for name, value in final_answer.items():
        if isinstance(name, str):
            numeric = _finite_float(value)
            if numeric is not None:
                fields[name] = numeric
    return fields or None


def safe_pct_error(
    prediction: float,
    groundtruth: float,
    epsilon: float = GT_NEAR_ZERO_EPSILON,
) -> Tuple[float, bool]:
    """Return error and whether it is a percentage rather than absolute."""
    if abs(groundtruth) < epsilon:
        return abs(prediction - groundtruth), False
    return (prediction - groundtruth) / groundtruth * 100.0, True


def score_from_error(
    pct_error_abs: Optional[float],
    abs_error: Optional[float],
    slot_type: str,
    thresholds: Mapping[str, Tuple[float, float]] = SCORING_THRESHOLDS,
    absolute_error_threshold: float = ABS_ERROR_FALLBACK_THRESHOLD,
) -> float:
    """Assign 1.0, 0.5, or 0.0 credit to one numeric prediction field."""
    full_pct, half_pct = thresholds[slot_type]
    if pct_error_abs is not None:
        if pct_error_abs <= full_pct:
            return 1.0
        if pct_error_abs <= half_pct:
            return 0.5
        return 0.0
    if abs_error is None:
        return 0.0
    if abs_error <= absolute_error_threshold:
        return 1.0
    if abs_error <= absolute_error_threshold * 3:
        return 0.5
    return 0.0


def _zero_credit_records(
    image: str,
    slot_key: str,
    mode: str,
    gt_fields: Mapping[str, float],
    status: str,
) -> List[Dict[str, Any]]:
    return [
        {
            "image": image,
            "category": category_of(slot_key),
            "slot_type": slot_type_of(slot_key),
            "mode": mode,
            "field": field_name,
            "gt_value": gt_value,
            "pred_value": None,
            "pct_error": None,
            "abs_error": None,
            "credit": 0.0,
            "status": status,
            "unparsable": True,
        }
        for field_name, gt_value in gt_fields.items()
    ]


def evaluate_predictions(
    predictions: Mapping[str, Any],
    groundtruth: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Tuple[float, float]] = SCORING_THRESHOLDS,
    near_zero_epsilon: float = GT_NEAR_ZERO_EPSILON,
    absolute_error_threshold: float = ABS_ERROR_FALLBACK_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Return detailed score records and evaluation coverage diagnostics."""
    pred_by_stem = index_by_stem(predictions, "Predictions")
    gt_by_stem = index_by_stem(groundtruth, "Ground truth")
    pred_stems = set(pred_by_stem)
    gt_stems = set(gt_by_stem)

    diagnostics = {
        "n_images_in_predictions": len(pred_by_stem),
        "n_images_in_groundtruth": len(gt_by_stem),
        "n_images_matched_to_gt": len(pred_stems & gt_stems),
        "n_prediction_images_without_gt": len(pred_stems - gt_stems),
        "n_gt_images_without_prediction": len(gt_stems - pred_stems),
        "n_errored_prediction_images": 0,
        "n_missing_slots": 0,
        "n_missing_modes": 0,
        "n_unknown_prediction_slots": 0,
        "n_invalid_groundtruth_slots": 0,
        "n_unparsable_predictions": 0,
        "n_missing_prediction_fields": 0,
    }
    records: List[Dict[str, Any]] = []

    for image_stem in sorted(gt_stems):
        gt_slots = gt_by_stem[image_stem]
        if not isinstance(gt_slots, dict):
            raise ValueError(f"Ground truth for {image_stem!r} must be an object")

        image_slots = pred_by_stem.get(image_stem)
        if image_slots is None:
            for slot_key, gt_value in gt_slots.items():
                try:
                    gt_fields = extract_gt_fields(gt_value)
                    slot_type_of(slot_key)
                except (TypeError, ValueError):
                    diagnostics["n_invalid_groundtruth_slots"] += 1
                    continue
                for mode in MODES:
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "missing_image",
                        )
                    )
            continue

        if not isinstance(image_slots, dict) or "error" in image_slots:
            diagnostics["n_errored_prediction_images"] += 1
            for slot_key, gt_value in gt_slots.items():
                try:
                    gt_fields = extract_gt_fields(gt_value)
                    slot_type_of(slot_key)
                except (TypeError, ValueError):
                    diagnostics["n_invalid_groundtruth_slots"] += 1
                    continue
                for mode in MODES:
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "image_error",
                        )
                    )
            continue

        for predicted_slot in image_slots:
            if predicted_slot == "error":
                continue
            if predicted_slot not in gt_slots:
                diagnostics["n_unknown_prediction_slots"] += 1

        for slot_key, gt_value in gt_slots.items():
            if not isinstance(slot_key, str):
                diagnostics["n_invalid_groundtruth_slots"] += 1
                continue
            try:
                slot_type = slot_type_of(slot_key)
            except ValueError:
                diagnostics["n_invalid_groundtruth_slots"] += 1
                continue
            gt_fields = extract_gt_fields(gt_value)
            if not gt_fields:
                diagnostics["n_invalid_groundtruth_slots"] += 1
                continue

            modes = image_slots.get(slot_key)
            if not isinstance(modes, dict):
                diagnostics["n_missing_slots"] += 1
                for mode in MODES:
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "missing_slot",
                        )
                    )
                continue

            for mode in MODES:
                mode_entry = modes.get(mode)
                if not isinstance(mode_entry, dict):
                    diagnostics["n_missing_modes"] += 1
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "missing_mode",
                        )
                    )
                    continue
                if "error" in mode_entry:
                    diagnostics["n_unparsable_predictions"] += 1
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "prediction_error",
                        )
                    )
                    continue

                pred_fields = extract_pred_fields(mode_entry.get("final_answer"))
                if pred_fields is None:
                    diagnostics["n_unparsable_predictions"] += 1
                    records.extend(
                        _zero_credit_records(
                            image_stem,
                            slot_key,
                            mode,
                            gt_fields,
                            "unparsable_answer",
                        )
                    )
                    continue

                for field_name, gt_value_float in gt_fields.items():
                    if field_name not in pred_fields:
                        diagnostics["n_missing_prediction_fields"] += 1
                        records.extend(
                            _zero_credit_records(
                                image_stem,
                                slot_key,
                                mode,
                                {field_name: gt_value_float},
                                "missing_field",
                            )
                        )
                        continue

                    pred_value = pred_fields[field_name]
                    error, is_percentage = safe_pct_error(
                        pred_value,
                        gt_value_float,
                        epsilon=near_zero_epsilon,
                    )
                    pct_error = error if is_percentage else None
                    abs_error = abs(pred_value - gt_value_float)
                    records.append(
                        {
                            "image": image_stem,
                            "category": category_of(slot_key),
                            "slot_type": slot_type,
                            "mode": mode,
                            "field": field_name,
                            "gt_value": gt_value_float,
                            "pred_value": pred_value,
                            "pct_error": pct_error,
                            "abs_error": abs_error,
                            "credit": score_from_error(
                                abs(pct_error) if pct_error is not None else None,
                                abs_error,
                                slot_type,
                                thresholds=thresholds,
                                absolute_error_threshold=absolute_error_threshold,
                            ),
                            "status": "ok",
                            "unparsable": False,
                        }
                    )

    return records, diagnostics


def percentile(sorted_values: List[float], percent: float) -> Optional[float]:
    if not sorted_values:
        return None
    index = min(
        len(sorted_values) - 1,
        int(round(percent / 100 * (len(sorted_values) - 1))),
    )
    return sorted_values[index]


def aggregate(
    records: List[Dict[str, Any]],
    key_fn: Callable[[Dict[str, Any]], str],
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key_fn(record)].append(record)

    summary: Dict[str, Dict[str, Any]] = {}
    for key, group in groups.items():
        pct_errors = sorted(
            abs(record["pct_error"])
            for record in group
            if record["pct_error"] is not None
        )
        abs_errors = [
            record["abs_error"]
            for record in group
            if record["abs_error"] is not None
        ]
        credits = [record["credit"] for record in group]
        summary[key] = {
            "n_records": len(group),
            "n_unparsable": sum(record["unparsable"] for record in group),
            "n_with_pct_error": len(pct_errors),
            "mean_abs_pct_error": (
                sum(pct_errors) / len(pct_errors) if pct_errors else None
            ),
            "median_abs_pct_error": percentile(pct_errors, 50),
            "p90_abs_pct_error": percentile(pct_errors, 90),
            "max_abs_pct_error": pct_errors[-1] if pct_errors else None,
            "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
            "mean_credit": sum(credits) / len(credits) if credits else None,
            "pct_full_credit": (
                sum(credit == 1.0 for credit in credits) / len(credits) * 100
                if credits
                else None
            ),
            "pct_zero_credit": (
                sum(credit == 0.0 for credit in credits) / len(credits) * 100
                if credits
                else None
            ),
        }
    return summary


def write_summary_csv(
    summary: Mapping[str, Dict[str, Any]],
    path: Path,
    key_column: str,
) -> None:
    fieldnames = [
        key_column,
        "n_records",
        "n_unparsable",
        "n_with_pct_error",
        "mean_abs_pct_error",
        "median_abs_pct_error",
        "p90_abs_pct_error",
        "max_abs_pct_error",
        "mean_abs_error",
        "mean_credit",
        "pct_full_credit",
        "pct_zero_credit",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, row in sorted(summary.items()):
            writer.writerow({key_column: key, **row})


def write_outputs(
    records: List[Dict[str, Any]],
    diagnostics: Dict[str, int],
    output_dir: Path,
    thresholds: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "per_record_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    summaries = {
        "slot": aggregate(records, lambda row: f"{row['category']}__{row['slot_type']}"),
        "slot_type": aggregate(records, lambda row: row["slot_type"]),
        "category": aggregate(records, lambda row: row["category"]),
        "mode": aggregate(records, lambda row: row["mode"]),
    }
    write_summary_csv(
        summaries["slot"],
        output_dir / "summary_by_slot.csv",
        "category__slot_type",
    )
    write_summary_csv(
        summaries["slot_type"],
        output_dir / "summary_by_slot_type.csv",
        "slot_type",
    )
    write_summary_csv(
        summaries["category"],
        output_dir / "summary_by_category.csv",
        "category",
    )
    write_summary_csv(
        summaries["mode"],
        output_dir / "summary_by_mode.csv",
        "mode",
    )

    pct_errors = [
        abs(record["pct_error"])
        for record in records
        if record["pct_error"] is not None
    ]
    credits = [record["credit"] for record in records]
    overall: Dict[str, Any] = {
        **diagnostics,
        "n_score_records": len(records),
        "n_valid_numeric_records": sum(record["status"] == "ok" for record in records),
        "overall_mean_abs_pct_error": (
            sum(pct_errors) / len(pct_errors) if pct_errors else None
        ),
        "overall_mean_credit": sum(credits) / len(credits) if credits else None,
        "scoring_thresholds_used": dict(thresholds),
    }
    with (output_dir / "summary_overall.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)
        handle.write("\n")
    return overall


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score MorphoOrga-VQA predictions against ground truth."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Filename-keyed prediction JSON produced by an evaluated agent.",
    )
    parser.add_argument(
        "--groundtruth",
        type=Path,
        required=True,
        help="Ground-truth JSON produced by generate_vqa_groundtruth_answers.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scoring_outputs"),
        help="Directory for detailed and aggregate scores (default: scoring_outputs).",
    )
    parser.add_argument(
        "--full-credit-pct",
        type=float,
        default=15.0,
        help="Maximum absolute percentage error for full credit (default: 15).",
    )
    parser.add_argument(
        "--half-credit-pct",
        type=float,
        default=30.0,
        help="Maximum absolute percentage error for half credit (default: 30).",
    )
    parser.add_argument(
        "--near-zero-epsilon",
        type=float,
        default=GT_NEAR_ZERO_EPSILON,
        help="GT magnitude below which absolute error is used.",
    )
    parser.add_argument(
        "--absolute-error-threshold",
        type=float,
        default=ABS_ERROR_FALLBACK_THRESHOLD,
        help="Full-credit absolute-error threshold for near-zero GT values.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if args.full_credit_pct < 0 or args.half_credit_pct < args.full_credit_pct:
        raise ValueError(
            "Require 0 <= --full-credit-pct <= --half-credit-pct"
        )
    if args.near_zero_epsilon < 0 or args.absolute_error_threshold < 0:
        raise ValueError("Error thresholds must be non-negative")

    thresholds = {
        slot_type: (args.full_credit_pct, args.half_credit_pct)
        for slot_type in SLOT_TYPES
    }
    predictions = load_json(args.predictions)
    groundtruth = load_json(args.groundtruth)
    records, diagnostics = evaluate_predictions(
        predictions,
        groundtruth,
        thresholds=thresholds,
        near_zero_epsilon=args.near_zero_epsilon,
        absolute_error_threshold=args.absolute_error_threshold,
    )
    overall = write_outputs(records, diagnostics, args.output_dir, thresholds)

    print("MorphoOrga-VQA scoring summary")
    print(f"  Prediction images: {overall['n_images_in_predictions']}")
    print(f"  Ground-truth images: {overall['n_images_in_groundtruth']}")
    print(f"  Matched images: {overall['n_images_matched_to_gt']}")
    print(f"  Valid numeric records: {overall['n_valid_numeric_records']}")
    print(f"  Total score records: {overall['n_score_records']}")
    mean_credit = overall["overall_mean_credit"]
    print(f"  Overall mean credit: {mean_credit:.4f}" if mean_credit is not None else "  Overall mean credit: n/a")
    print(f"  Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
