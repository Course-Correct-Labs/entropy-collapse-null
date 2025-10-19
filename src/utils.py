"""
Utility functions for data loading and validation.
"""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .constants import EXTERNAL_COLS_REQUIRED, INTERNAL_COLS_REQUIRED


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    """Load and parse manifest.json from run directory."""
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found in {run_dir}. "
            "This file contains run metadata and configuration."
        )

    with open(manifest_path, "r") as f:
        return json.load(f)


def load_metrics_internal(run_dir: Path) -> pd.DataFrame:
    """
    Load metrics_internal.csv with validation.

    Expected columns:
    - prompt_id: unique identifier for each sequence
    - model_name: model used (e.g., 'microsoft/phi-2')
    - eci_raw: raw ECI value
    - eci_residualized: residualized ECI (control-adjusted)
    - early_eci_raw: ECI computed on early window only
    - effective_ranks: list of effective rank values over token windows
    - participation_ratios: list of participation ratio values
    - variances: list of variance values
    - window_starts: list of window start indices
    - window_ends: list of window end indices
    """
    csv_path = run_dir / "metrics_internal.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"metrics_internal.csv not found in {run_dir}. "
            "This file should contain internal model metrics."
        )

    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = set(INTERNAL_COLS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in metrics_internal.csv: {missing}. "
            f"Expected columns: {INTERNAL_COLS_REQUIRED}"
        )

    # Parse list columns stored as strings
    list_cols = [
        "effective_ranks",
        "participation_ratios",
        "variances",
        "window_starts",
        "window_ends",
    ]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    return df


def load_metrics_external(run_dir: Path) -> pd.DataFrame:
    """
    Load metrics_external.csv with validation.

    Expected columns:
    - prompt_id: unique identifier matching metrics_internal
    - model_name: model used
    - qa_failure: boolean indicating QA task failure
    - reasoning_failures: dict of reasoning failure types
    - delta_i_values: list of ΔI drift values
    - ngram_novelty_values: list of n-gram novelty values
    - char_entropy_values: list of character entropy values
    """
    csv_path = run_dir / "metrics_external.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"metrics_external.csv not found in {run_dir}. "
            "This file should contain external behavioral metrics."
        )

    df = pd.read_csv(csv_path)

    # Validate required columns
    missing = set(EXTERNAL_COLS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in metrics_external.csv: {missing}. "
            f"Expected columns: {EXTERNAL_COLS_REQUIRED}"
        )

    # Parse list/dict columns
    list_cols = ["delta_i_values", "ngram_novelty_values", "char_entropy_values"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)

    if "reasoning_failures" in df.columns:
        df["reasoning_failures"] = df["reasoning_failures"].apply(safe_parse_dict)

    # Ensure qa_failure is boolean
    df["qa_failure"] = df["qa_failure"].astype(bool)

    return df


def safe_parse_list(val: Any) -> List:
    """Safely parse a string representation of a list."""
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []
    return []


def safe_parse_dict(val: Any) -> Dict:
    """Safely parse a string representation of a dict."""
    if pd.isna(val) or val == "":
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    return {}


def merge_metrics(df_internal: pd.DataFrame, df_external: pd.DataFrame) -> pd.DataFrame:
    """Merge internal and external metrics on prompt_id and model_name."""
    merged = pd.merge(
        df_internal,
        df_external,
        on=["prompt_id", "model_name"],
        how="inner",
        suffixes=("", "_external"),
    )

    if len(merged) == 0:
        raise ValueError(
            "No matching rows found when merging internal and external metrics. "
            "Check that prompt_id and model_name values match between CSVs."
        )

    return merged


def subsample_data(
    df: pd.DataFrame, frac: float = 0.05, min_rows: int = 30, seed: int = 42
) -> pd.DataFrame:
    """Subsample DataFrame for smoke tests."""
    n_target = max(int(len(df) * frac), min_rows)
    n_target = min(n_target, len(df))
    return df.sample(n=n_target, random_state=seed).reset_index(drop=True)
