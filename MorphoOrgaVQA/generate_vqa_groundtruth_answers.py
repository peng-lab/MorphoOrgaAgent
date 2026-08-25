"""Generate MorphoOrga-VQA ground-truth answers from instance-label masks.

Users may provide one or more ``--mask`` arguments or a ``--mask-dir``. Each
input must be a 2D NumPy ``.npy`` array with background label 0 and positive
integer instance IDs.

Output JSON structure:
{
  "mask_file.npy": {
    "Category__slot": answer_value
  }
}

If some masks fail and --fail-fast is not set, failures are written to a
separate JSON file so the main answer file keeps this filename-keyed shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = DEFAULT_BASE_DIR.parent
PROJECT_PARENT = REPOSITORY_DIR.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MorphoOrgaAgent.tools import morphology_tools as morph


DEFAULT_BENCHMARK_JSON = DEFAULT_BASE_DIR / "benchmark_questions.json"
DEFAULT_OUTPUT = DEFAULT_BASE_DIR / "vqa_groundtruth_answers.json"

EXPECTED_CATEGORIES = {
    "Area": "Area_outer",
    "Perimeter": "Perimeter_outer",
    "Roundness": "roundness",
    "Roughness": "perimeter_diff",
}
EXPECTED_MODES = ("clear", "open")
EXPECTED_SLOTS = ("max", "tertile_mean")


def to_json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas scalar containers into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def collect_mask_paths(
    mask_dir: Path,
    limit: int | None = None,
    recursive: bool = False,
) -> List[Path]:
    """Collect unique ``.npy`` mask files from a directory."""
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")

    pattern = "**/*.npy" if recursive else "*.npy"
    mask_paths = sorted(path for path in mask_dir.glob(pattern) if path.is_file())
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        mask_paths = mask_paths[:limit]

    if not mask_paths:
        raise FileNotFoundError(f"No .npy masks found in: {mask_dir}")

    return mask_paths


def validate_explicit_masks(mask_paths: List[Path]) -> List[Path]:
    """Validate explicit mask paths and reject ambiguous output keys."""
    resolved_paths: List[Path] = []
    for path in mask_paths:
        resolved = path.expanduser().resolve()
        if resolved.suffix.lower() != ".npy":
            raise ValueError(f"Mask must be a .npy file: {path}")
        if not resolved.is_file():
            raise FileNotFoundError(f"Mask file does not exist: {path}")
        resolved_paths.append(resolved)

    return ensure_unique_mask_names(resolved_paths)


def ensure_unique_mask_names(mask_paths: List[Path]) -> List[Path]:
    """Ensure filename-keyed output cannot overwrite results silently."""
    names = [path.name for path in mask_paths]
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(
            "Mask filenames must be unique because filenames are used as JSON keys: "
            f"{duplicate_names}"
        )
    return sorted(mask_paths, key=lambda path: str(path).lower())


def validate_instance_mask(mask: np.ndarray, source: Path | None = None) -> np.ndarray:
    """Validate and normalize a 2D non-negative integer instance-label mask."""
    label = f" {source}" if source else ""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Mask{label} is not a NumPy array")
    if mask.ndim != 2:
        raise ValueError(f"Mask{label} must be 2D; found shape {mask.shape}")
    if mask.size == 0:
        raise ValueError(f"Mask{label} is empty")
    if not np.issubdtype(mask.dtype, np.number) or np.issubdtype(mask.dtype, np.bool_):
        raise TypeError(f"Mask{label} must have a numeric integer dtype; found {mask.dtype}")
    if not np.all(np.isfinite(mask)):
        raise ValueError(f"Mask{label} contains NaN or infinite values")
    if np.any(mask < 0):
        raise ValueError(f"Mask{label} contains negative labels")
    if not np.all(mask == np.floor(mask)):
        raise ValueError(f"Mask{label} contains non-integer labels")
    if not np.any(mask > 0):
        raise ValueError(f"Mask{label} contains no positive instance IDs")
    if float(np.max(mask)) > np.iinfo(np.int32).max:
        raise ValueError(f"Mask{label} contains labels larger than int32 supports")
    return mask.astype(np.int32, copy=False)


def build_organoid_table(mask: np.ndarray) -> pd.DataFrame:
    """Build the metric table required by the benchmark answer templates."""
    organoid_ids = morph.get_organoid_ids(mask)
    area_outer = morph.calculate_area_outer(mask)
    perimeter_outer = morph.calculate_perimeter_outer(mask)
    x_coords, y_coords = morph.calculate_centroids(mask)
    ideal_perimeter = morph.calculate_ideal_perimeter(mask, area_outer=area_outer)
    perimeter_diff = morph.calculate_perimeter_diff(
        mask,
        perimeter_outer=perimeter_outer,
        ideal_perimeter=ideal_perimeter,
    )
    roundness = morph.calculate_roundness(
        mask,
        area_outer=area_outer,
        perimeter_outer=perimeter_outer,
    )
    return pd.DataFrame(
        {
            "organoid_idx": organoid_ids,
            "Area_outer": area_outer,
            "Perimeter_outer": perimeter_outer,
            "x": x_coords,
            "y": y_coords,
            "roundness": roundness,
            "perimeter_diff": perimeter_diff,
        }
    )


def load_benchmark_questions(path: Path) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
    """Load and validate the compact benchmark question schema."""
    with path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    if set(benchmark) != set(EXPECTED_CATEGORIES):
        raise ValueError(
            "Expected benchmark categories "
            f"{sorted(EXPECTED_CATEGORIES)}, found {sorted(benchmark)}"
        )

    normalized: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
    for category, metric in EXPECTED_CATEGORIES.items():
        modes = benchmark[category]
        if set(modes) != set(EXPECTED_MODES):
            raise ValueError(
                f"Category {category!r} must contain modes {EXPECTED_MODES}; "
                f"found {sorted(modes)}"
            )

        normalized[category] = {}
        for mode in EXPECTED_MODES:
            questions = list(modes[mode].items())
            if len(questions) != len(EXPECTED_SLOTS):
                raise ValueError(
                    f"Category {category!r}, mode {mode!r} must contain exactly "
                    f"{len(EXPECTED_SLOTS)} questions; found {len(questions)}"
                )

            mode_questions: Dict[str, List[str]] = {}
            for slot, (_, required_metrics) in zip(EXPECTED_SLOTS, questions):
                if isinstance(required_metrics, str):
                    required_metrics = [required_metrics]
                if not isinstance(required_metrics, list) or not all(
                    isinstance(item, str) for item in required_metrics
                ):
                    raise ValueError(
                        f"Required metrics for {category!r}/{mode!r}/{slot!r} "
                        "must be a list of strings"
                    )
                if metric not in required_metrics:
                    raise ValueError(
                        f"Question {category!r}/{mode!r}/{slot!r} must require "
                        f"metric {metric!r}; found {required_metrics}"
                    )
                mode_questions[slot] = required_metrics
            normalized[category][mode] = mode_questions

    return normalized


def _all_tertile_means(df: Any, sort_metric: str, mean_metric: str) -> Dict[str, Any]:
    """Return equal-count low/middle/high tier means for one metric."""
    if df.empty:
        return {"low": None, "mid": None, "high": None}
    if len(df) < 3:
        mean = float(df[mean_metric].mean())
        return {"low": mean, "mid": mean, "high": mean}

    # Rank first so duplicate values still produce three equal-count groups.
    ranks = df[sort_metric].rank(method="first")
    tiers = pd.qcut(ranks, 3, labels=["low", "mid", "high"])
    return {
        tier: (None if df[tiers == tier].empty else float(df.loc[tiers == tier, mean_metric].mean()))
        for tier in ("low", "mid", "high")
    }


def generate_answers_for_benchmark(mask: np.ndarray, benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Generate answers matching the current benchmark's four categories."""
    answers: Dict[str, Any] = {}
    df = build_organoid_table(mask)

    for category, metric in EXPECTED_CATEGORIES.items():
        if df.empty:
            answers[f"{category}__max"] = None
        else:
            row = df.loc[df[metric].idxmax()]
            if category == "Area":
                answers[f"{category}__max"] = {
                    "Area_outer": float(row["Area_outer"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            else:
                answers[f"{category}__max"] = {metric: float(row[metric])}

        answers[f"{category}__tertile_mean"] = _all_tertile_means(
            df, sort_metric=metric, mean_metric=metric
        )

    return answers


def generate_answers_for_masks(
    mask_paths: List[Path],
    benchmark: Dict[str, Any],
    fail_fast: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    answers_by_file: Dict[str, Any] = {}
    failures: Dict[str, str] = {}

    for mask_path in mask_paths:
        try:
            mask = validate_instance_mask(
                np.load(mask_path, allow_pickle=False),
                source=mask_path,
            )
            answers_by_file[mask_path.name] = to_json_safe(
                generate_answers_for_benchmark(mask, benchmark)
            )
        except Exception as exc:
            if fail_fast:
                raise
            failures[mask_path.name] = f"{type(exc).__name__}: {exc}"

    return answers_by_file, failures


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MorphoOrga-VQA ground-truth answers for .npy masks."
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=DEFAULT_BENCHMARK_JSON,
        help=f"Benchmark question JSON. Default: {DEFAULT_BENCHMARK_JSON}",
    )
    mask_group = parser.add_mutually_exclusive_group(required=True)
    mask_group.add_argument(
        "--mask",
        dest="masks",
        type=Path,
        action="append",
        help="One .npy instance mask. Repeat the option to process multiple masks.",
    )
    mask_group.add_argument(
        "--mask-dir",
        type=Path,
        help="Directory containing .npy instance masks.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search --mask-dir recursively for .npy files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for failed masks. Default: output path with "
            "'.failures.json' suffix when failures occur."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of masks to process for a manual smoke test.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use 0 for compact single-line JSON.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any mask fails instead of recording failures.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    if args.masks and args.recursive:
        raise ValueError("--recursive can only be used with --mask-dir")

    benchmark = load_benchmark_questions(args.benchmark_json)
    if args.masks:
        mask_paths = validate_explicit_masks(args.masks)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit must be a positive integer")
            mask_paths = mask_paths[: args.limit]
    else:
        mask_paths = ensure_unique_mask_names(
            collect_mask_paths(
                args.mask_dir,
                limit=args.limit,
                recursive=args.recursive,
            )
        )
    answers_by_file, failures = generate_answers_for_masks(
        mask_paths,
        benchmark=benchmark,
        fail_fast=args.fail_fast,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent == 0 else args.indent
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(answers_by_file, f, indent=indent, sort_keys=True)
        f.write("\n")

    print(f"Input masks: {len(mask_paths)}")
    print(f"Successful: {len(answers_by_file)}")
    print(f"Failed: {len(failures)}")
    print(f"Ground truth: {args.output.resolve()}")
    if failures:
        failures_output = args.failures_output
        if failures_output is None:
            failures_output = args.output.with_suffix(".failures.json")
        failures_output.parent.mkdir(parents=True, exist_ok=True)
        with failures_output.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=indent, sort_keys=True)
            f.write("\n")
        print(f"Wrote {len(failures)} failed masks to {failures_output}")


if __name__ == "__main__":
    main()
