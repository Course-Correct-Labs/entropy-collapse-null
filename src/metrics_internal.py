"""
Internal metrics: effective rank, participation ratio, variance.

These metrics are computed from model hidden states and measure
internal representational structure.
"""

from typing import List
import numpy as np
import logging

# FIX: Add logging for numerical stability warnings
logger = logging.getLogger(__name__)


def compute_effective_rank(singular_values: np.ndarray) -> float:
    """
    Compute effective rank from singular values.

    Effective rank = exp(H(p)) where H is Shannon entropy
    and p is the normalized distribution of singular values.

    Args:
        singular_values: Array of singular values from SVD

    Returns:
        Effective rank (float)
    """
    if len(singular_values) == 0:
        return 0.0

    # Normalize to probability distribution
    s_squared = singular_values ** 2
    p = s_squared / s_squared.sum()

    # Avoid log(0)
    p = p[p > 0]

    # Shannon entropy
    H = -np.sum(p * np.log(p))

    # Effective rank
    return float(np.exp(H))


def compute_participation_ratio(singular_values: np.ndarray) -> float:
    """
    Compute participation ratio from singular values.

    PR = (sum(s²))² / sum(s⁴)

    Args:
        singular_values: Array of singular values from SVD

    Returns:
        Participation ratio (float), or None if computation produces inf/nan
    """
    if len(singular_values) == 0:
        return 0.0

    s_squared = singular_values ** 2
    numerator = s_squared.sum() ** 2
    denominator = (s_squared ** 2).sum()

    if denominator == 0:
        return 0.0

    # FIX: Add numerical stability guard for inf/nan values
    pr = numerator / denominator

    if not np.isfinite(pr):
        logger.warning(
            f"Participation ratio computation produced non-finite value: {pr}. "
            f"Returning None. Singular values shape: {singular_values.shape}, "
            f"numerator: {numerator}, denominator: {denominator}"
        )
        return None  # FIX: Return None for non-finite values (will be stored as empty in CSV)

    return float(pr)


def compute_variance(hidden_states: np.ndarray) -> float:
    """
    Compute variance of hidden state activations.

    Args:
        hidden_states: Array of shape (seq_len, hidden_dim)

    Returns:
        Variance (float)
    """
    return float(np.var(hidden_states))


def aggregate_trajectory(values: List[float]) -> dict:
    """
    Aggregate a trajectory of metric values.

    Args:
        values: List of metric values over time

    Returns:
        Dict with mean, std, min, max
    """
    if not values or len(values) == 0:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
    }
