"""Lightweight tests for the public command-line runner."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from MorphoOrgaAgent.core.state import OrganoidState
from MorphoOrgaAgent.run_agent import (
    default_save_folder,
    main,
    parse_args,
    resolve_api_key,
)


class RunAgentTests(unittest.TestCase):
    def test_save_folder_is_optional(self) -> None:
        args = parse_args(["--image_path", "images/sample.png", "--query", "count"])
        self.assertIsNone(args.save_folder)

    def test_api_key_comes_from_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}, clear=True):
            self.assertEqual(resolve_api_key(None), "environment-key")

    def test_cli_api_key_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "environment-key"}, clear=True):
            self.assertEqual(resolve_api_key("cli-key"), "cli-key")

    def test_missing_api_key_keeps_fallback_behavior(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_api_key(None))

    def test_api_key_resolution_does_not_print_the_key(self) -> None:
        captured_stdout = StringIO()
        captured_stderr = StringIO()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret-test-key"}, clear=True):
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                self.assertEqual(resolve_api_key(None), "secret-test-key")

        self.assertNotIn("secret-test-key", captured_stdout.getvalue())
        self.assertNotIn("secret-test-key", captured_stderr.getvalue())

    def test_default_save_folder_preserves_relative_image_path(self) -> None:
        result = default_save_folder(
            "images/experiment_1/sample.png",
            working_dir="/work/project",
            repo_dir="/work/project/MorphoOrgaAgent",
        )
        self.assertEqual(
            result,
            Path("/work/project/MorphoOrgaAgent/outputs/images/experiment_1/sample"),
        )

    def test_absolute_external_image_falls_back_to_stem(self) -> None:
        result = default_save_folder(
            "/data/microscopy/sample.png",
            working_dir="/work/project",
            repo_dir="/work/project/MorphoOrgaAgent",
        )
        self.assertEqual(
            result,
            Path("/work/project/MorphoOrgaAgent/outputs/sample"),
        )

    def test_main_wires_cli_arguments_to_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            output_path = Path(temp_dir) / "results"
            expected_state = OrganoidState(
                image_path=str(image_path),
                output_dir=str(output_path),
            )
            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "MorphoOrgaAgent.run_agent.run_pipeline",
                    return_value=expected_state,
                ) as mocked_pipeline:
                    result = main(
                        [
                            "--image_path",
                            str(image_path),
                            "--query",
                            "measure roundness",
                            "--save_folder",
                            str(output_path),
                            "--device",
                            "cuda",
                        ]
                    )

            self.assertIs(result, expected_state)
            mocked_pipeline.assert_called_once_with(
                image_path=str(image_path),
                query="measure roundness",
                output_dir=output_path,
                api_key=None,
                sam3_confidence_threshold=0.5,
                min_cellpose_area=1,
                device="cuda",
            )

    def test_main_forwards_environment_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            output_path = Path(temp_dir) / "results"
            expected_state = OrganoidState(
                image_path=str(image_path),
                output_dir=str(output_path),
            )
            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "environment-key"},
                clear=True,
            ):
                with patch(
                    "MorphoOrgaAgent.run_agent.run_pipeline",
                    return_value=expected_state,
                ) as mocked_pipeline:
                    main(
                        [
                            "--image_path",
                            str(image_path),
                            "--query",
                            "count organoids",
                            "--save_folder",
                            str(output_path),
                        ]
                    )

            self.assertEqual(
                mocked_pipeline.call_args.kwargs["api_key"],
                "environment-key",
            )


if __name__ == "__main__":
    unittest.main()
