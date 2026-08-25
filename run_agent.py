#!/usr/bin/env python3
"""Command-line entry point for the complete MorphoOrgaAgent pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_DIR = SCRIPT_PATH.parent
PROJECT_PARENT = PACKAGE_DIR.parent
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MorphoOrgaAgent.core.pipeline import run_pipeline
from MorphoOrgaAgent.core.state import OrganoidState


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete MorphoOrgaAgent microscopy-analysis pipeline."
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input microscopy image.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural-language analysis request.",
    )
    parser.add_argument(
        "--save_folder",
        "--output_dir",
        dest="save_folder",
        default=None,
        help=(
            "Folder for pipeline artifacts. When omitted, a folder under "
            "MorphoOrgaAgent/outputs is derived from the image path."
        ),
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="OpenAI API key. Defaults to the OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--sam3-confidence-threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold for refined segmentation (default: 0.5).",
    )
    parser.add_argument(
        "--min-cellpose-area",
        type=int,
        default=1,
        help="Ignore Cellpose prompt objects below this pixel area (default: 1).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="SAM3 device. By default, CUDA is selected when available.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_api_key(cli_api_key: str | None) -> str | None:
    """Resolve an API key without logging or otherwise exposing its value."""
    if cli_api_key and cli_api_key.strip():
        return cli_api_key.strip()

    environment_api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    return environment_api_key or None


def default_save_folder(
    image_path: str | Path,
    *,
    working_dir: str | Path | None = None,
    repo_dir: str | Path = PACKAGE_DIR,
) -> Path:
    """Derive a safe output folder from an image's path relative to the CWD."""
    cwd = Path(working_dir or Path.cwd()).expanduser().resolve()
    image = Path(image_path).expanduser()
    resolved_image = image.resolve() if image.is_absolute() else (cwd / image).resolve()

    try:
        relative_image = resolved_image.relative_to(cwd)
    except ValueError:
        relative_image = Path(resolved_image.name)

    relative_without_suffix = relative_image.with_suffix("")
    return Path(repo_dir).expanduser().resolve() / "outputs" / relative_without_suffix


def main(argv: Sequence[str] | None = None) -> OrganoidState:
    args = parse_args(argv)
    api_key = resolve_api_key(args.api_key)
    save_folder = (
        Path(args.save_folder).expanduser()
        if args.save_folder
        else default_save_folder(args.image_path)
    )
    return run_pipeline(
        image_path=args.image_path,
        query=args.query,
        output_dir=save_folder,
        api_key=api_key,
        sam3_confidence_threshold=args.sam3_confidence_threshold,
        min_cellpose_area=args.min_cellpose_area,
        device=args.device,
    )


if __name__ == "__main__":
    main()
