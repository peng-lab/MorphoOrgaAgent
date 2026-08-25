"""Synthetic, model-free tests for the MorphoOrga-VQA utilities."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from MorphoOrgaAgent.MorphoOrgaVQA.generate_vqa_groundtruth_answers import (
    collect_mask_paths,
    generate_answers_for_benchmark,
    load_benchmark_questions,
    main as generate_main,
    validate_instance_mask,
)
from MorphoOrgaAgent.MorphoOrgaVQA.score_vqa_predictions import (
    evaluate_predictions,
    score_from_error,
    write_outputs,
)


BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "MorphoOrgaVQA"
    / "benchmark_questions.json"
)


def synthetic_mask() -> np.ndarray:
    mask = np.zeros((12, 24), dtype=np.int32)
    mask[1:5, 1:5] = 1
    mask[2:9, 12:21] = 2
    return mask


class GroundTruthGenerationTests(unittest.TestCase):
    def test_generates_all_benchmark_answer_slots(self) -> None:
        benchmark = load_benchmark_questions(BENCHMARK_PATH)
        answers = generate_answers_for_benchmark(synthetic_mask(), benchmark)
        self.assertEqual(len(answers), 8)
        self.assertEqual(
            set(answers),
            {
                "Area__max",
                "Area__tertile_mean",
                "Perimeter__max",
                "Perimeter__tertile_mean",
                "Roundness__max",
                "Roundness__tertile_mean",
                "Roughness__max",
                "Roughness__tertile_mean",
            },
        )
        self.assertEqual(set(answers["Area__max"]), {"Area_outer", "x", "y"})

    def test_single_mask_cli_writes_groundtruth(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mask_path = temp_path / "sample.npy"
            output_path = temp_path / "answers.json"
            np.save(mask_path, synthetic_mask())

            with redirect_stdout(StringIO()):
                generate_main(
                    [
                        "--mask",
                        str(mask_path),
                        "--output",
                        str(output_path),
                        "--fail-fast",
                    ]
                )

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"sample.npy"})
            self.assertIn("Roughness__tertile_mean", payload["sample.npy"])

    def test_batch_collection_supports_recursive_search(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            np.save(root / "one.npy", synthetic_mask())
            np.save(nested / "two.npy", synthetic_mask())

            self.assertEqual(len(collect_mask_paths(root)), 1)
            self.assertEqual(len(collect_mask_paths(root, recursive=True)), 2)

    def test_rejects_non_integer_mask_values(self) -> None:
        invalid = synthetic_mask().astype(np.float32)
        invalid[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "non-integer"):
            validate_instance_mask(invalid)

    def test_rejects_mask_without_instances(self) -> None:
        with self.assertRaisesRegex(ValueError, "no positive instance IDs"):
            validate_instance_mask(np.zeros((8, 8), dtype=np.int32))


class PredictionScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.groundtruth = {
            "sample.npy": {
                "Area__max": {"Area_outer": 10.0, "x": 2.0, "y": 3.0}
            }
        }

    def test_missing_prediction_fields_receive_zero_credit(self) -> None:
        predictions = {
            "sample.jpg": {
                "Area__max": {
                    "clear": {"final_answer": {"Area_outer": 10.0}},
                    "open": {"final_answer": {"Area_outer": 10.0}},
                }
            }
        }
        records, diagnostics = evaluate_predictions(predictions, self.groundtruth)

        missing = [record for record in records if record["status"] == "missing_field"]
        self.assertEqual(len(missing), 4)
        self.assertTrue(all(record["credit"] == 0.0 for record in missing))
        self.assertEqual(diagnostics["n_missing_prediction_fields"], 4)

    def test_missing_image_is_reported_and_scores_zero(self) -> None:
        records, diagnostics = evaluate_predictions({}, self.groundtruth)
        self.assertEqual(diagnostics["n_gt_images_without_prediction"], 1)
        self.assertTrue(records)
        self.assertTrue(all(record["status"] == "missing_image" for record in records))
        self.assertTrue(all(record["credit"] == 0.0 for record in records))

    def test_credit_thresholds_are_configurable(self) -> None:
        thresholds = {"max": (5.0, 10.0), "tertile_mean": (5.0, 10.0)}
        self.assertEqual(score_from_error(4.0, None, "max", thresholds), 1.0)
        self.assertEqual(score_from_error(8.0, None, "max", thresholds), 0.5)
        self.assertEqual(score_from_error(12.0, None, "max", thresholds), 0.0)

    def test_scoring_outputs_include_coverage_summaries(self) -> None:
        predictions = {
            "sample.jpg": {
                "Area__max": {
                    "clear": {
                        "final_answer": {"Area_outer": 10.0, "x": 2.0, "y": 3.0}
                    },
                    "open": {
                        "final_answer": {"Area_outer": 10.0, "x": 2.0, "y": 3.0}
                    },
                }
            }
        }
        records, diagnostics = evaluate_predictions(predictions, self.groundtruth)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            overall = write_outputs(
                records,
                diagnostics,
                output_dir,
                {"max": (15.0, 30.0), "tertile_mean": (15.0, 30.0)},
            )
            self.assertEqual(overall["n_images_matched_to_gt"], 1)
            self.assertTrue((output_dir / "per_record_scores.csv").is_file())
            self.assertTrue((output_dir / "summary_by_mode.csv").is_file())
            self.assertTrue((output_dir / "summary_overall.json").is_file())


if __name__ == "__main__":
    unittest.main()
