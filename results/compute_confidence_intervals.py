"""
Compute 95% bootstrap confidence intervals for saved evaluation runs.

The script processes the default dataset/model/phrase/mode/n grid defined below
and writes confidence_{timestamp}.json beside each run's latest reasoning file.
Existing confidence files are skipped only when their metadata still matches the
source reasoning file and fixed bootstrap settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import helpers


N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 0
N_JOBS = -1

DATASETS = ["d3", "df40", "genimage"]
MODELS = ["llava-onevision-7b", "qwen25-vl-7b", "qwen3-vl-8b"]
PHRASES = ["baseline", "cot", "s2"]
MODES = ["prefill", "prompt", "prefill-pseudo-system"]
N_VALUES = [1]

PredictionMetric = Callable[[List[str], List[str]], float]


def is_base_reasoning_file(path: Path) -> bool:
    """Return True for original reasoning files, excluding derived variants."""
    return (
        path.name.startswith("reasoning_")
        and path.suffix == ".json"
        and "_with_" not in path.stem
    )


def find_latest_reasoning_file(output_dir: Path) -> Optional[Path]:
    """Find the latest base reasoning_*.json file in an output directory."""
    reasoning_files = [
        path for path in output_dir.glob("reasoning_*.json")
        if is_base_reasoning_file(path)
    ]
    if not reasoning_files:
        return None
    return max(reasoning_files, key=lambda path: path.name)


def confidence_file_for_reasoning(reasoning_file: Path) -> Path:
    """Build confidence_{timestamp}.json for a reasoning_{timestamp}.json file."""
    prefix = "reasoning_"
    if not reasoning_file.name.startswith(prefix):
        raise ValueError(f"Unexpected reasoning filename: {reasoning_file.name}")

    timestamp = reasoning_file.stem[len(prefix):]
    return reasoning_file.with_name(f"confidence_{timestamp}.json")


def load_reasoning_file(path: Path) -> List[Dict[str, Any]]:
    """Load a reasoning JSON file and validate the fields needed for scoring."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, found {type(data).__name__}")
    if not data:
        raise ValueError(f"Reasoning file is empty: {path}")

    required_keys = {"aggregated_prediction", "ground_truth"}
    for index, result in enumerate(data):
        missing = required_keys.difference(result)
        if missing:
            raise ValueError(f"{path} row {index} is missing keys: {sorted(missing)}")

    return data


def optional_float(value: Any) -> Optional[float]:
    """Convert a JSON scalar to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def confidence_is_current(
    confidence_file: Path,
    reasoning_file: Path,
    point_macro_f1: float,
) -> bool:
    """Check whether an existing confidence file matches the current inputs."""
    if not confidence_file.exists():
        return False

    try:
        with confidence_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False

    metadata = data.get("metadata", {})
    macro_f1 = data.get("macro_f1", {})
    source = str(reasoning_file.relative_to(config.PROJECT_ROOT))

    stored_point = optional_float(macro_f1.get("point"))
    stored_confidence_level = optional_float(macro_f1.get("confidence_level"))

    return (
        metadata.get("reasoning_file") == source
        and metadata.get("random_seed") == RANDOM_SEED
        and macro_f1.get("n_bootstrap") == N_BOOTSTRAP
        and stored_point is not None
        and abs(stored_point - point_macro_f1) <= 1e-12
        and stored_confidence_level is not None
        and abs(stored_confidence_level - CONFIDENCE_LEVEL) <= 1e-12
    )


def _bootstrap_iteration(
    predictions: List[str],
    ground_truth: List[str],
    metric_fn: PredictionMetric,
    seed: int,
) -> float:
    """Run one bootstrap iteration: sample rows with replacement and score."""
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(predictions), size=len(predictions), replace=True)
    sample_preds = [predictions[i] for i in indices]
    sample_truth = [ground_truth[i] for i in indices]
    return metric_fn(sample_preds, sample_truth)


def bootstrap_confidence_interval(
    predictions: List[str],
    ground_truth: List[str],
    metric_fn: PredictionMetric,
) -> Dict[str, Any]:
    """Compute a fixed 95% percentile bootstrap confidence interval."""
    point_estimate = metric_fn(predictions, ground_truth)
    print(f"  Running {N_BOOTSTRAP} bootstrap iterations using {mp.cpu_count()} CPUs...")

    bootstrap_scores = Parallel(n_jobs=N_JOBS)(
        delayed(_bootstrap_iteration)(
            predictions,
            ground_truth,
            metric_fn,
            RANDOM_SEED + iteration,
        )
        for iteration in tqdm(range(N_BOOTSTRAP), desc="  Bootstrap", ncols=80)
    )

    alpha = 1 - CONFIDENCE_LEVEL
    ci_lower = np.percentile(bootstrap_scores, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha / 2) * 100)

    return {
        "point": float(point_estimate),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "n_bootstrap": N_BOOTSTRAP,
        "confidence_level": CONFIDENCE_LEVEL,
    }


def compute_and_save_confidence_intervals(
    dataset: str,
    model: str,
    phrase: str,
    mode: str = "prefill",
    n: int = 1,
    override: bool = False,
) -> str:
    """Compute and save confidence intervals for one configuration."""
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)
    reasoning_file = find_latest_reasoning_file(output_dir)

    if reasoning_file is None:
        return f"skipped missing reasoning file: {output_dir}"

    confidence_file = confidence_file_for_reasoning(reasoning_file)
    reasoning_data = load_reasoning_file(reasoning_file)
    predictions, ground_truth = helpers.extract_predictions_and_truth(reasoning_data)
    point_macro_f1 = helpers.compute_macro_f1_from_predictions(predictions, ground_truth)

    if not override and confidence_is_current(confidence_file, reasoning_file, point_macro_f1):
        return f"skipped current {confidence_file.relative_to(config.PROJECT_ROOT)}"

    print()
    print(f"Computing 95% CI for {dataset}/{model}/{phrase}/{mode}/n={n}")
    print(f"  Source reasoning: {reasoning_file.relative_to(config.PROJECT_ROOT)}")
    print(f"  Loaded {len(predictions)} examples")

    macro_f1_ci = bootstrap_confidence_interval(
        predictions,
        ground_truth,
        helpers.compute_macro_f1_from_predictions,
    )

    confidence_data = {
        "macro_f1": macro_f1_ci,
        "metadata": {
            "dataset": dataset,
            "model": model,
            "phrase": phrase,
            "mode": mode,
            "n_responses": n,
            "n_examples": len(predictions),
            "reasoning_file": str(reasoning_file.relative_to(config.PROJECT_ROOT)),
            "random_seed": RANDOM_SEED,
        },
    }

    with confidence_file.open("w", encoding="utf-8") as handle:
        json.dump(confidence_data, handle, indent=2)

    ci_lower, ci_upper = macro_f1_ci["ci_95"]
    print(f"  Saved: {confidence_file.relative_to(config.PROJECT_ROOT)}")
    print(f"  Macro F1: {macro_f1_ci['point']:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
    return f"processed {confidence_file.relative_to(config.PROJECT_ROOT)}"


def process_all_combinations(override: bool = False) -> None:
    """Process every configured dataset/model/phrase/mode/n combination."""
    combinations = [
        (dataset, model, phrase, mode, n)
        for dataset, model, phrase, mode, n in itertools.product(DATASETS, MODELS, PHRASES, MODES, N_VALUES)
        if phrase != "baseline" or mode == MODES[0]
    ]

    print("Bootstrap Confidence Interval Computation")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {MODELS}")
    print(f"Phrases: {PHRASES}")
    print(f"Modes: {MODES}")
    print(f"n values: {N_VALUES}")
    print(f"Confidence level: {CONFIDENCE_LEVEL}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Combinations: {len(combinations)}")
    if override:
        print("Override: existing confidence files will be recomputed")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for index, (dataset, model, phrase, mode, n) in enumerate(combinations, 1):
        label = f"[{index}/{len(combinations)}] {dataset}/{model}/{phrase}/{mode}/n={n}"
        try:
            status = compute_and_save_confidence_intervals(
                dataset,
                model,
                phrase,
                mode,
                n,
                override=override,
            )
        except Exception as exc:
            print(f"{label}: error: {exc}")
            errors += 1
            continue

        print(f"{label}: {status}")
        if status.startswith("processed"):
            processed += 1
        else:
            skipped += 1

    print()
    print("Summary")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute fixed 95% bootstrap confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Recompute even when timestamped confidence metadata is current.",
    )
    args = parser.parse_args()
    process_all_combinations(override=args.override)


if __name__ == "__main__":
    main()
