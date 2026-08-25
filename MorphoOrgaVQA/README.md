# MorphoOrga-VQA benchmark utilities

This directory contains the question specification and deterministic evaluation
utilities used by the MorphoOrga-VQA benchmark. The tools are intentionally
separate from the end-to-end agent: ground truth is calculated directly from
instance-label masks, while predictions are read from a JSON export produced by
the system being evaluated.

## Files

| File | Purpose |
| --- | --- |
| `benchmark_questions.json` | Defines clear and open-language question variants and their required metrics. |
| `generate_vqa_groundtruth_answers.py` | Computes deterministic answers from one or more instance masks. |
| `score_vqa_predictions.py` | Matches predictions to ground truth and produces detailed and aggregate scores. |

The benchmark currently contains four categories—area, perimeter, roundness,
and roughness—with `max` and `tertile_mean` tasks. Each task has semantically
equivalent `clear` and `open` question modes. Because the two modes target the
same measurement, one deterministic ground-truth answer is shared by both.

## Mask requirements

Each input mask must be a two-dimensional NumPy `.npy` array:

- background pixels are labeled `0`;
- every organoid has a positive integer instance ID;
- values must be finite and non-negative;
- semantic or probability masks are not accepted.

For example:

```python
import numpy as np

mask = np.zeros((512, 512), dtype=np.int32)
mask[50:120, 80:150] = 1
mask[200:300, 240:360] = 2
np.save("my_organoid_mask.npy", mask)
```

## Generate ground truth

Run these commands from the `MorphoOrgaAgent/` repository directory.

### One mask

```bash
python MorphoOrgaVQA/generate_vqa_groundtruth_answers.py \
  --mask /path/to/my_organoid_mask.npy \
  --output outputs/my_groundtruth.json
```

Repeat `--mask` to process several explicitly selected files:

```bash
python MorphoOrgaVQA/generate_vqa_groundtruth_answers.py \
  --mask /path/to/sample_1.npy \
  --mask /path/to/sample_2.npy \
  --output outputs/my_groundtruth.json
```

### Directory of masks

```bash
python MorphoOrgaVQA/generate_vqa_groundtruth_answers.py \
  --mask-dir /path/to/masks \
  --output outputs/vqa_groundtruth_answers.json
```

Add `--recursive` to search nested directories. By default, a malformed mask is
recorded in a separate `*.failures.json` file and processing continues; use
`--fail-fast` when any invalid input should stop the run.

The output is keyed by mask filename:

```json
{
  "sample.npy": {
    "Area__max": {
      "Area_outer": 1200.0,
      "x": 150.5,
      "y": 98.0
    },
    "Area__tertile_mean": {
      "low": 250.0,
      "mid": 600.0,
      "high": 1100.0
    }
  }
}
```

The actual file also contains the perimeter, roundness, and roughness slots.
Tertiles are equal-count groups obtained by ranking values before applying
three-way quantile binning; samples with fewer than three instances use the
overall mean for all three tiers.

## Prediction format

Predictions are keyed by image filename. Image and mask extensions may differ:
the scorer matches `sample.jpg` to `sample.npy` using the filename stem.

```json
{
  "sample.jpg": {
    "Area__max": {
      "clear": {
        "final_answer": {
          "Area_outer": 1185.0,
          "x": 149.0,
          "y": 99.0
        }
      },
      "open": {
        "final_answer": {
          "Area_outer": 1210.0,
          "x": 151.0,
          "y": 97.5
        }
      }
    }
  }
}
```

Each `final_answer` must be a JSON object whose numeric field names match the
corresponding ground-truth answer. Missing images, slots, modes, fields, errors,
and non-finite values are visible in the coverage summary and receive zero
credit.

## Score predictions

```bash
python MorphoOrgaVQA/score_vqa_predictions.py \
  --predictions /path/to/vqa_agent_predictions.json \
  --groundtruth outputs/vqa_groundtruth_answers.json \
  --output-dir outputs/scoring_outputs
```

The default field-level credit thresholds are:

| Absolute percentage error | Credit |
| ---: | ---: |
| ≤ 15% | 1.0 |
| > 15% and ≤ 30% | 0.5 |
| > 30% | 0.0 |

The thresholds can be changed for an evaluation protocol:

```bash
python MorphoOrgaVQA/score_vqa_predictions.py \
  --predictions predictions.json \
  --groundtruth groundtruth.json \
  --full-credit-pct 10 \
  --half-credit-pct 25
```

When the absolute ground-truth value is below `1e-6`, percentage error is
unstable, so scoring falls back to absolute error. The default full-credit
absolute-error threshold is `1e-3`, with half credit up to three times that
value. Both settings are configurable from the CLI.

The scorer writes:

```text
scoring_outputs/
├── per_record_scores.csv
├── summary_by_slot.csv
├── summary_by_slot_type.csv
├── summary_by_category.csv
├── summary_by_mode.csv
└── summary_overall.json
```

For reporting, use the coverage fields in `summary_overall.json` together with
median and 90th-percentile absolute percentage error, mean credit, and the
detailed per-record table. The default credit thresholds are evaluation
settings, not universal biological tolerances, and should be reported whenever
benchmark scores are published.
