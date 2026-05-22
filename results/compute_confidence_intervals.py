"""
Confidence Interval Computation Script

Computes bootstrap confidence intervals for evaluation metrics from reasoning JSON files.
Uses parallel processing with joblib for fast computation.

Metrics computed:
- Macro F1-score with 95% CI

Output: confidence.json in the same directory as performance.json

Usage:
    1. Configure DATASETS, MODELS, PHRASES, MODES, N_VALUES at the top of the script
    2. Run: python compute_confidence_intervals.py

The script will process all combinations and skip existing confidence.json files.
"""

import json
import multiprocessing as mp
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from joblib import Parallel, delayed

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import config and helpers
import config
import helpers

# ============================================================================
# CONFIGURATION
# ============================================================================

# Bootstrap parameters
N_BOOTSTRAP = 10000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 0
N_JOBS = -1  # Use all available CPUs

# Override flag (set via command line)
OVERRIDE = False

# Combinations to compute CIs for
# Modify these lists to compute CIs for specific configurations
DATASETS = ['d3', 'df40', 'genimage']  # Datasets to process
MODELS = ['llava-onevision-7b', 'qwen25-vl-7b', 'qwen3-vl-8b']  # Models to process
PHRASES = ['baseline', 'cot', 's2']  # Phrases to process
MODES = ['prefill']  # Phrase modes to process
N_VALUES = [1]  # n values to process

# Note: Metric computation functions are now in helpers.py
# We use helpers.compute_macro_f1_from_predictions()


# ============================================================================
# BOOTSTRAP FUNCTIONS
# ============================================================================

def _bootstrap_iteration(predictions: List[str], ground_truth: List[str],
                        metric_fn, seed: int) -> float:
    """
    Single bootstrap iteration: resample and compute metric.

    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels
        metric_fn: Metric computation function
        seed: Random seed for this iteration

    Returns:
        Metric value for this bootstrap sample
    """
    rng = np.random.RandomState(seed)
    n_samples = len(predictions)

    # Resample with replacement
    indices = rng.choice(n_samples, size=n_samples, replace=True)

    sample_preds = [predictions[i] for i in indices]
    sample_truth = [ground_truth[i] for i in indices]

    return metric_fn(sample_preds, sample_truth)


def bootstrap_confidence_interval(predictions: List[str], ground_truth: List[str],
                                  metric_fn, n_bootstrap: int = N_BOOTSTRAP,
                                  confidence_level: float = CONFIDENCE_LEVEL,
                                  random_seed: int = RANDOM_SEED,
                                  n_jobs: int = N_JOBS) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels
        metric_fn: Metric computation function
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all CPUs)

    Returns:
        Dictionary with point estimate and CI bounds
    """
    # Point estimate
    point_estimate = metric_fn(predictions, ground_truth)

    # Bootstrap sampling in parallel
    print(f"  Running {n_bootstrap} bootstrap iterations using {mp.cpu_count()} CPUs...")

    bootstrap_scores = Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_iteration)(predictions, ground_truth, metric_fn, random_seed + i)
        for i in tqdm(range(n_bootstrap), desc="  Bootstrap", ncols=80)
    )

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(bootstrap_scores, lower_percentile)
    ci_upper = np.percentile(bootstrap_scores, upper_percentile)

    return {
        'point': float(point_estimate),
        'ci_95': [float(ci_lower), float(ci_upper)],
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level
    }


# Note: Data loading functions are now in helpers.py
# We use helpers.load_reasoning_json() and helpers.extract_predictions_and_truth()


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_and_save_confidence_intervals(dataset: str, model: str, phrase: str,
                                         mode: str = 'prefill', n: int = 1, override: bool = False):
    """
    Compute confidence intervals and save to confidence.json.

    Args:
        dataset: Dataset name
        model: Model name
        phrase: Phrase name
        mode: Phrase mode (default: 'prefill')
        n: Number of responses (default: 1)
        override: Force recompute even if confidence.json exists (default: False)
    """
    output_dir = config.get_output_dir(dataset, model, phrase, mode, n)
    confidence_file = output_dir / "confidence.json"

    # Skip if already exists (unless override)
    if confidence_file.exists() and not override:
        print(f"⚠️  Confidence intervals already exist at {confidence_file}")
        print("   Skipping computation. Use --override flag to recompute.")
        return

    print(f"\n📊 Computing confidence intervals for:")
    print(f"   Dataset: {dataset}")
    print(f"   Model: {model}")
    print(f"   Phrase: {phrase}")
    print(f"   Mode: {mode}")
    print(f"   n: {n}")

    # Load reasoning data
    print("\n📁 Loading reasoning data...")
    reasoning_data = helpers.load_reasoning_json(dataset, model, phrase, mode, n)
    predictions, ground_truth = helpers.extract_predictions_and_truth(reasoning_data)

    print(f"   Loaded {len(predictions)} examples")

    # Compute bootstrap CI for macro F1
    print("\n🔄 Computing bootstrap CI for Macro F1...")
    macro_f1_ci = bootstrap_confidence_interval(
        predictions,
        ground_truth,
        helpers.compute_macro_f1_from_predictions,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED,
        n_jobs=N_JOBS
    )

    # Prepare output data
    confidence_data = {
        'macro_f1': macro_f1_ci,
        'metadata': {
            'dataset': dataset,
            'model': model,
            'phrase': phrase,
            'mode': mode,
            'n_responses': n,
            'n_examples': len(predictions)
        }
    }

    # Save to JSON
    with open(confidence_file, 'w', encoding='utf-8') as f:
        json.dump(confidence_data, f, indent=2)

    print(f"\n✅ Confidence intervals saved to: {confidence_file}")
    print(f"\n📈 Results:")
    print(f"   Macro F1: {macro_f1_ci['point']:.4f} [{macro_f1_ci['ci_95'][0]:.4f}, {macro_f1_ci['ci_95'][1]:.4f}]")


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_combinations(override: bool = False):
    """
    Process all combinations specified in configuration.

    Computes confidence intervals for all dataset/model/phrase/mode/n combinations.
    Skips existing confidence.json files automatically unless override=True.

    Args:
        override: Force recompute even if confidence.json exists
    """
    import itertools

    # Generate all combinations
    combinations = list(itertools.product(DATASETS, MODELS, PHRASES, MODES, N_VALUES))
    total = len(combinations)

    print("=" * 80)
    print("Bootstrap Confidence Interval Computation - Batch Mode")
    print("=" * 80)
    print(f"\n📋 Processing {total} combinations:")
    print(f"   Datasets: {DATASETS}")
    print(f"   Models: {MODELS}")
    print(f"   Phrases: {PHRASES}")
    print(f"   Modes: {MODES}")
    print(f"   n values: {N_VALUES}")
    if override:
        print(f"   Override: ✅ Will recompute existing files")
    print()

    # Track statistics
    processed = 0
    skipped = 0
    errors = 0

    # Process each combination
    for idx, (dataset, model, phrase, mode, n) in enumerate(combinations, 1):
        print("=" * 80)
        print(f"[{idx}/{total}] Processing: {dataset} / {model} / {phrase} / {mode} / n={n}")
        print("=" * 80)

        try:
            # Check if already exists (unless override)
            output_dir = config.get_output_dir(dataset, model, phrase, mode, n)
            confidence_file = output_dir / "confidence.json"

            if confidence_file.exists() and not override:
                print(f"⏭️  Skipping - confidence.json already exists (use --override to recompute)")
                skipped += 1
                continue

            # Check if performance/reasoning files exist
            reasoning_files = sorted(output_dir.glob("reasoning_*.json"))
            if not reasoning_files:
                print(f"⚠️  Skipping - no reasoning.json found in {output_dir}")
                skipped += 1
                continue

            # Compute CI
            compute_and_save_confidence_intervals(dataset, model, phrase, mode, n, override=override)
            processed += 1

        except Exception as e:
            print(f"❌ Error processing combination: {e}")
            errors += 1

        print()

    # Summary
    print("=" * 80)
    print("📊 Batch Processing Summary")
    print("=" * 80)
    print(f"Total combinations: {total}")
    print(f"✅ Processed: {processed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Errors: {errors}")
    print("=" * 80)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--override',
        action='store_true',
        help='Force recompute even if confidence.json already exists'
    )

    args = parser.parse_args()

    process_all_combinations(override=args.override)


if __name__ == "__main__":
    main()
