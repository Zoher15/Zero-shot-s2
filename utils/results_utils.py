"""
Shared utilities for results processing scripts.
Eliminates code duplication across results generation scripts.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from collections import defaultdict
import re

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import config
from utils import helpers

logger = logging.getLogger(__name__)

# --- Common Configuration Constants ---
DATASET_DISPLAY_NAMES = {
    "d32k": "D3 (2k)",
    "df402k": "DF40 (2k)", 
    "genimage2k": "GenImage (2k)",
    "d3": "D3",
    "df40": "DF40",
    "genimage": "GenImage"
}

MODEL_DISPLAY_NAMES = {
    "qwen25-7b": "Qwen2.5-VL-7B",
    "qwen25-3b": "Qwen2.5-VL-3B", 
    "qwen25-72b": "Qwen2.5-VL-72B",
    "llama3-11b": "Llama-3.2-11B-Vision",
    "llama3-90b": "Llama-3.2-90B-Vision"
}

METHOD_DISPLAY_NAMES = {
    "zeroshot": "Standard",
    "zeroshot-cot": "Chain-of-Thought",
    "zeroshot-2-artifacts": "Zero-shot-s²",
    "zeroshot-artifacts": "Artifacts Only",
    "zeroshot-examine": "Examine",
    "zeroshot-style": "Style"
}

# --- Common Data Loading Functions ---
def load_scores_data(
    models: List[str],
    datasets: List[str], 
    methods: List[str],
    n_val: str = "1",
    score_type: str = "macro_f1"
) -> pd.DataFrame:
    """
    Load evaluation scores from CSV files into a unified DataFrame.
    
    Args:
        models: List of model names
        datasets: List of dataset names  
        methods: List of method names
        n_val: Number of sequences value
        score_type: Type of score to extract
        
    Returns:
        DataFrame with columns: model, dataset, method, score
    """
    all_data = []
    missing_files = []
    
    for model in models:
        for dataset in datasets:
            for method in methods:
                # Determine prefix based on model type
                prefix = "AI_llama" if "llama" in model.lower() else "AI_qwen"
                filename = f"{prefix}-{dataset}-{model}-{method}-n{n_val}-scores.csv"
                filepath = config.SCORES_DIR / filename
                
                df = helpers.load_scores_csv_to_dataframe(filepath)
                
                if not df.empty:
                    try:
                        target_index = f"{method}-n{n_val}"
                        if target_index in df.index:
                            score_col = score_type if score_type in df.columns else df.columns[0]
                            raw_score = df.loc[target_index, score_col]
                            score = round(float(raw_score) * 100, 1)  # Convert to percentage
                            
                            all_data.append({
                                'model': model,
                                'dataset': dataset, 
                                'method': method,
                                'score': score
                            })
                        else:
                            logger.warning(f"Index '{target_index}' not found in {filepath}")
                    except Exception as e:
                        logger.error(f"Error processing {filepath}: {e}")
                else:
                    missing_files.append(str(filepath))
    
    if missing_files:
        logger.info(f"Missing {len(missing_files)} score files")
        
    return pd.DataFrame(all_data) if all_data else pd.DataFrame()

def load_rationales_data(
    models: List[str],
    datasets: List[str],
    methods: List[str], 
    n_val: str = "1"
) -> Dict[str, List[Dict]]:
    """
    Load rationales data from JSONL files.
    
    Returns:
        Dictionary mapping (model, dataset, method) tuples to rationales lists
    """
    rationales_data = {}
    
    for model in models:
        for dataset in datasets:
            for method in methods:
                prefix = "AI_llama" if "llama" in model.lower() else "AI_qwen"
                filename = f"{prefix}-{dataset}-{model}-{method}-n{n_val}-rationales.jsonl"
                filepath = config.RESPONSES_DIR / filename
                
                if filepath.exists():
                    try:
                        rationales = helpers.load_rationales_from_file(filepath)
                        key = (model, dataset, method)
                        rationales_data[key] = rationales
                    except Exception as e:
                        logger.error(f"Error loading rationales from {filepath}: {e}")
                        
    return rationales_data

# --- LaTeX Table Generation Utilities ---
class LaTeXTableBuilder:
    """Builder class for creating LaTeX tables with consistent formatting."""
    
    def __init__(self, font_size: str = r"\scriptsize", bold_max: bool = True):
        self.font_size = font_size
        self.bold_max = bold_max
        self.table_content = []
        
    def start_table(self, columns: str, caption: str, label: str):
        """Start a new table with specified column specification."""
        self.table_content = [
            r"\begin{table}[htbp]",
            r"\centering",
            self.font_size,
            f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule"
        ]
        self._caption = caption
        self._label = label
        
    def add_header(self, headers: List[str]):
        """Add table header row."""
        header_row = " & ".join(headers) + r" \\"
        self.table_content.append(header_row)
        self.table_content.append(r"\midrule")
        
    def add_row(self, row_data: List[str], is_separator: bool = False):
        """Add a data row to the table."""
        if is_separator:
            self.table_content.append(r"\midrule")
        else:
            row = " & ".join(str(item) for item in row_data) + r" \\"
            self.table_content.append(row)
            
    def end_table(self):
        """Finalize the table."""
        self.table_content.extend([
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{self._caption}}}",
            f"\\label{{{self._label}}}",
            r"\end{table}"
        ])
        
    def get_table(self) -> str:
        """Get the complete LaTeX table as a string."""
        return "\n".join(self.table_content)
        
    def save_table(self, filepath: Union[str, Path]):
        """Save the table to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.get_table())
            
        logger.info(f"LaTeX table saved to {filepath}")

def format_score_with_bold(
    score: float, 
    max_score: float, 
    bold_threshold: float = 0.1,
    decimal_places: int = 1
) -> str:
    """
    Format a score, making it bold if it's the maximum or close to it.
    
    Args:
        score: The score to format
        max_score: The maximum score in the group
        bold_threshold: Threshold for considering a score "close to max"
        decimal_places: Number of decimal places
        
    Returns:
        Formatted score string with LaTeX bold formatting if applicable
    """
    if pd.isna(score):
        return "—"
        
    formatted = f"{score:.{decimal_places}f}"
    
    if abs(score - max_score) <= bold_threshold:
        return f"\\textbf{{{formatted}}}"
    else:
        return formatted

def escape_latex_text(text: str) -> str:
    """Escape special LaTeX characters in text."""
    if not isinstance(text, str):
        return str(text)
        
    # Replace common special characters
    replacements = {
        '&': r'\&',
        '%': r'\%', 
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    return text

# --- Data Analysis Utilities ---
def calculate_improvement_percentage(baseline: float, improved: float) -> float:
    """Calculate percentage improvement from baseline to improved score."""
    if baseline == 0:
        return float('inf') if improved > 0 else 0
    return ((improved - baseline) / baseline) * 100

def get_score_statistics(scores: List[float]) -> Dict[str, float]:
    """Calculate basic statistics for a list of scores."""
    scores_array = np.array([s for s in scores if not pd.isna(s)])
    
    if len(scores_array) == 0:
        return {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
        
    return {
        'mean': np.mean(scores_array),
        'std': np.std(scores_array),
        'min': np.min(scores_array), 
        'max': np.max(scores_array)
    }

def find_best_methods_per_dataset(
    scores_df: pd.DataFrame,
    group_by: List[str] = ['dataset'],
    score_col: str = 'score'
) -> pd.DataFrame:
    """
    Find the best performing method for each dataset/group.
    
    Args:
        scores_df: DataFrame with score data
        group_by: Columns to group by
        score_col: Column containing scores
        
    Returns:
        DataFrame with best methods per group
    """
    def get_best_method(group):
        best_idx = group[score_col].idxmax()
        return group.loc[best_idx]
        
    return scores_df.groupby(group_by).apply(get_best_method).reset_index(drop=True)

# --- File Organization Utilities ---
def organize_output_files(
    base_name: str,
    extensions: List[str] = ['.tex', '.png', '.pdf'],
    timestamp: bool = False
) -> Dict[str, Path]:
    """
    Generate organized output file paths for different formats.
    
    Args:
        base_name: Base name for the files
        extensions: List of file extensions to create paths for
        timestamp: Whether to add timestamp to filenames
        
    Returns:
        Dictionary mapping extension to Path object
    """
    from datetime import datetime
    
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{ts}"
        
    file_paths = {}
    
    for ext in extensions:
        if ext == '.tex':
            file_paths[ext] = config.TABLES_DIR / f"{base_name}{ext}"
        elif ext in ['.png', '.pdf', '.svg']:
            file_paths[ext] = config.PLOTS_DIR / f"{base_name}{ext}"
        else:
            # Default to outputs directory
            file_paths[ext] = config.OUTPUTS_DIR / f"{base_name}{ext}"
            
    return file_paths

# --- Common Validation Functions ---
def validate_required_files(
    models: List[str],
    datasets: List[str], 
    methods: List[str],
    file_type: str = "scores"
) -> Tuple[List[str], List[str]]:
    """
    Validate that required files exist for the analysis.
    
    Args:
        models, datasets, methods: Lists of identifiers
        file_type: Either "scores" or "rationales"
        
    Returns:
        Tuple of (existing_files, missing_files)
    """
    existing = []
    missing = []
    
    for model in models:
        for dataset in datasets:
            for method in methods:
                prefix = "AI_llama" if "llama" in model.lower() else "AI_qwen"
                
                if file_type == "scores":
                    filename = f"{prefix}-{dataset}-{model}-{method}-n1-scores.csv"
                    filepath = config.SCORES_DIR / filename
                elif file_type == "rationales":
                    filename = f"{prefix}-{dataset}-{model}-{method}-n1-rationales.jsonl"
                    filepath = config.RESPONSES_DIR / filename
                else:
                    raise ValueError(f"Unknown file_type: {file_type}")
                    
                if filepath.exists():
                    existing.append(str(filepath))
                else:
                    missing.append(str(filepath))
                    
    return existing, missing

# --- Setup Function for Results Scripts ---
def setup_results_script(script_name: str, log_file_path: Optional[Path] = None):
    """
    Common setup for results scripts including logging and directory creation.
    
    Args:
        script_name: Name of the script (for logging)
        log_file_path: Optional custom log file path
    """
    if log_file_path is None:
        log_file_path = config.LOGS_DIR / f"results_{script_name}.log"
        
    helpers.setup_global_logger(log_file_path)
    
    # Ensure output directories exist
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(0)
    
    logger.info(f"Starting {script_name} results generation")
    return logging.getLogger(script_name) 