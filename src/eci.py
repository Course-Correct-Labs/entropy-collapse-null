"""
Epistemic Collapse Index (ECI) computation and analysis.

ECI measures the rate of change in internal metrics over token generation.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy import stats

from .constants import ECI_COLLAPSE_THRESHOLD


def compute_eci_slope(metric_values: List[float], token_indices: Optional[List[int]] = None) -> float:
    """
    Compute ECI as the slope of metric values over token generation.

    Uses linear regression: metric ~ token_index

    Args:
        metric_values: List of metric values (e.g., effective rank) over windows
        token_indices: Optional list of token indices; if None, uses 0, 1, 2, ...

    Returns:
        Slope (ECI value)
    """
    if not metric_values or len(metric_values) < 2:
        return 0.0

    y = np.array(metric_values)

    if token_indices is None:
        x = np.arange(len(y))
    else:
        x = np.array(token_indices[:len(y)])

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    return float(slope)


def residualize_eci(eci_values: np.ndarray, control_eci: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Residualize ECI values against control condition.

    If control is provided: residualized = eci - mean(control)
    If no control: residualized = eci (identity)

    Args:
        eci_values: Array of ECI values to residualize
        control_eci: Optional array of control ECI values

    Returns:
        Residualized ECI values
    """
    if control_eci is None or len(control_eci) == 0:
        return eci_values

    control_mean = np.mean(control_eci)
    return eci_values - control_mean


def classify_collapse(eci: float, threshold: float = ECI_COLLAPSE_THRESHOLD) -> bool:
    """
    Classify whether an ECI value indicates collapse.

    Args:
        eci: ECI value
        threshold: Collapse threshold (default -0.02)

    Returns:
        True if eci < threshold (collapse detected)
    """
    return eci < threshold


def compute_collapse_fraction(eci_values: np.ndarray, threshold: float = ECI_COLLAPSE_THRESHOLD) -> float:
    """
    Compute fraction of sequences with ECI below collapse threshold.

    Args:
        eci_values: Array of ECI values
        threshold: Collapse threshold

    Returns:
        Fraction [0, 1]
    """
    if len(eci_values) == 0:
        return 0.0

    n_collapsed = np.sum(eci_values < threshold)
    return n_collapsed / len(eci_values)


def compute_eci_stats(eci_values: np.ndarray) -> dict:
    """
    Compute summary statistics for ECI distribution.

    Args:
        eci_values: Array of ECI values

    Returns:
        Dict with mean, std, median, collapse_fraction
    """
    if len(eci_values) == 0:
        return {
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'collapse_fraction': 0.0,
            'n': 0,
        }

    return {
        'mean': float(np.mean(eci_values)),
        'std': float(np.std(eci_values)),
        'median': float(np.median(eci_values)),
        'collapse_fraction': compute_collapse_fraction(eci_values),
        'n': len(eci_values),
    }


def bootstrap_ci(values: np.ndarray, stat_func=np.mean, n_bootstrap: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Array of values
        stat_func: Function to compute statistic (default: np.mean)
        n_bootstrap: Number of bootstrap resamples
        alpha: Significance level (default 0.05 for 95% CI)
        seed: Random seed

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if len(values) == 0:
        return (0.0, 0.0)

    rng = np.random.RandomState(seed)
    bootstrap_stats = []

    for _ in range(n_bootstrap):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_stats.append(stat_func(resample))

    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))
