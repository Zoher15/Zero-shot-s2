"""
Generate LaTeX table for macro F1 scores across datasets and phrase modes.

Creates a publication-ready table with:
- Nested rows for phrases (baseline, cot, s2)
- Different modes (prefill, prefill-pseudo-system, prompt)
- Performance deltas (+/-) relative to prefill baseline
- Macro F1 scores formatted to 1 decimal place
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import shared configuration
import plot_config as pc

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset configuration (2k variants) - use same order as plot_config
DATASETS = ['d3-2k', 'df40-2k', 'genimage-2k']
DATASET_DISPLAY_NAMES = {
    'd3-2k': 'D3 (2k)',
    'df40-2k': 'DF40 (2k)',
    'genimage-2k': 'GenImage (2k)'
}

# Phrase and mode configuration
PHRASES = ['baseline', 'cot', 's2']
MODES = {
    'baseline': [None],  # No mode for baseline
    'cot': ['prefill', 'prefill-pseudo-system', 'prompt'],
    's2': ['prefill', 'prefill-pseudo-system', 'prompt']
}

MODE_DISPLAY_NAMES = {
    'prefill': 'Prefill',
    'prefill-pseudo-system': 'Pseudo-Prefill',
    'prompt': 'Prompt'
}

# Models to generate tables for
MODELS = pc.MODEL_ORDER

# Output directory
TABLES_DIR = Path(__file__).resolve().parent / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_latest_performance_file(dataset: str, model: str, phrase: str, mode: Optional[str] = None) -> Optional[Path]:
    """Find the latest performance_fixed JSON file for given configuration."""
    base_dir = Path(__file__).resolve().parent.parent / "output" / dataset / model

    if phrase == 'baseline':
        search_dir = base_dir / 'baseline'
    else:
        if mode is None:
            return None
        search_dir = base_dir / phrase / mode

    if not search_dir.exists():
        return None

    # Find all performance_fixed JSON files (with null handling)
    performance_files = list(search_dir.glob('performance_fixed_*.json'))

    if not performance_files:
        return None

    # Return the most recent file
    return max(performance_files, key=lambda p: p.stat().st_mtime)


def load_macro_f1(dataset: str, model: str, phrase: str, mode: Optional[str] = None) -> Optional[float]:
    """Load macro F1 score from performance JSON file."""
    perf_file = find_latest_performance_file(dataset, model, phrase, mode)

    if perf_file is None:
        return None

    try:
        with open(perf_file, 'r') as f:
            data = json.load(f)
        return data['metrics']['macro_f1']
    except (KeyError, json.JSONDecodeError, FileNotFoundError):
        return None


def format_f1_with_delta(f1: Optional[float], baseline_f1: Optional[float]) -> str:
    """Format F1 score with delta annotation relative to baseline."""
    if f1 is None:
        return "---"

    # Convert to percentage and round for display
    f1_pct = f1 * 100
    f1_rounded = round(f1_pct, 1)
    f1_str = f"{f1_rounded:.1f}"

    # Calculate delta if baseline exists
    if baseline_f1 is not None and baseline_f1 != f1:
        baseline_pct = baseline_f1 * 100
        baseline_rounded = round(baseline_pct, 1)
        # Calculate delta from rounded values for consistency with displayed numbers
        delta = f1_rounded - baseline_rounded
        if abs(delta) >= 0.05:  # Only show delta if >= 0.05
            sign = "+" if delta > 0 else ""
            delta_str = f"{{\\scriptsize ({sign}{delta:.1f})}}"
            return f"{f1_str} {delta_str}"

    return f1_str


def generate_latex_table(model: str, output_path: Path):
    """Generate LaTeX table with nested rows and performance deltas."""

    # Collect all data
    data = {}
    for phrase in PHRASES:
        data[phrase] = {}
        for mode in MODES[phrase]:
            data[phrase][mode] = {}
            for dataset in DATASETS:
                f1 = load_macro_f1(dataset, model, phrase, mode)
                data[phrase][mode][dataset] = f1

    # Start building LaTeX table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lllll}")
    lines.append("\\toprule")

    # Header
    header = "\\textbf{Phrase} & \\textbf{Type} & "
    header += " & ".join([f"\\textbf{{{DATASET_DISPLAY_NAMES[d]}}}" for d in DATASETS])
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    # Table rows
    for phrase in PHRASES:
        modes = MODES[phrase]
        num_modes = len(modes)

        if phrase == 'baseline':
            # Baseline row (no mode column)
            row = "Baseline & --- & "
            row += " & ".join([format_f1_with_delta(data[phrase][None][d], None) for d in DATASETS])
            row += " \\\\"
            lines.append(row)
            # Add midrule after baseline
            lines.append("\\midrule")
        else:
            # Multi-row for cot/s2
            phrase_display = 'CoT' if phrase == 'cot' else phrase.upper()

            for i, mode in enumerate(modes):
                # Get baseline F1 for delta calculation (prefill is the baseline)
                baseline_mode = 'prefill'

                row = ""
                if i == 0:
                    # First row: include multirow phrase
                    row += f"\\multirow{{{num_modes}}}{{*}}{{{phrase_display}}} & "
                else:
                    # Subsequent rows: empty phrase column
                    row += " & "

                # Mode column
                row += f"{MODE_DISPLAY_NAMES[mode]} & "

                # Dataset columns with deltas
                f1_values = []
                for dataset in DATASETS:
                    f1 = data[phrase][mode][dataset]
                    baseline_f1 = data[phrase][baseline_mode][dataset]
                    f1_values.append(format_f1_with_delta(f1, baseline_f1))

                row += " & ".join(f1_values)
                row += " \\\\"
                lines.append(row)

            # Add cmidrule after each phrase group (except last)
            if phrase != PHRASES[-1]:
                lines.append("\\cmidrule(lr){1-5}")

    # Table footer
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Macro F1 scores across datasets and phrase modes. "
                "Values in parentheses indicate performance change relative to Prefill baseline. "
                f"Results for {model}.}}")
    lines.append(f"\\label{{tab:macro_f1_modes_{model.replace('-', '_')}}}")
    lines.append("\\end{table}")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ LaTeX table generated: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate LaTeX tables for all models."""
    for model in MODELS:
        output_path = TABLES_DIR / f"macro_f1_modes_{model}.tex"
        generate_latex_table(model, output_path)


if __name__ == '__main__':
    main()
