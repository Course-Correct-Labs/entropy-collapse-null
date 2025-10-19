"""
Bootstrap confidence intervals for ROC-AUC and PR-AUC.
"""

from typing import Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .constants import BOOTSTRAP_N_RESAMPLES, BOOTSTRAP_SEED, CI_ALPHA


def bootstrap_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> Tuple[float, float, float]:
    """
    Compute ROC-AUC with bootstrap confidence interval.

    Args:
        y_true: True binary labels
        y_score: Predicted scores
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed
        alpha: Significance level (default 0.05 for 95% CI)

    Returns:
        Tuple of (auc, lower_ci, upper_ci)
    """
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return (0.5, 0.5, 0.5)

    # Compute base AUC
    try:
        base_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        return (0.5, 0.5, 0.5)

    # Bootstrap
    rng = np.random.RandomState(seed)
    bootstrap_aucs = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        # Skip if only one class in bootstrap sample
        if len(set(y_true_boot)) < 2:
            continue

        try:
            auc_boot = roc_auc_score(y_true_boot, y_score_boot)
            bootstrap_aucs.append(auc_boot)
        except ValueError:
            continue

    if not bootstrap_aucs:
        return (base_auc, base_auc, base_auc)

    # Compute percentile CI
    lower = np.percentile(bootstrap_aucs, alpha / 2 * 100)
    upper = np.percentile(bootstrap_aucs, (1 - alpha / 2) * 100)

    return (float(base_auc), float(lower), float(upper))


def bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = CI_ALPHA,
) -> Tuple[float, float, float]:
    """
    Compute PR-AUC (Average Precision) with bootstrap confidence interval.

    Args:
        y_true: True binary labels
        y_score: Predicted scores
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed
        alpha: Significance level

    Returns:
        Tuple of (ap, lower_ci, upper_ci)
    """
    if len(y_true) == 0 or len(set(y_true)) < 2:
        prevalence = np.mean(y_true) if len(y_true) > 0 else 0.5
        return (prevalence, prevalence, prevalence)

    # Compute base Average Precision
    try:
        base_ap = average_precision_score(y_true, y_score)
    except ValueError:
        prevalence = np.mean(y_true)
        return (prevalence, prevalence, prevalence)

    # Bootstrap
    rng = np.random.RandomState(seed)
    bootstrap_aps = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_score_boot = y_score[indices]

        if len(set(y_true_boot)) < 2:
            continue

        try:
            ap_boot = average_precision_score(y_true_boot, y_score_boot)
            bootstrap_aps.append(ap_boot)
        except ValueError:
            continue

    if not bootstrap_aps:
        return (base_ap, base_ap, base_ap)

    lower = np.percentile(bootstrap_aps, alpha / 2 * 100)
    upper = np.percentile(bootstrap_aps, (1 - alpha / 2) * 100)

    return (float(base_ap), float(lower), float(upper))


def compute_roc_curve_data(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve data.

    Args:
        y_true: True binary labels
        y_score: Predicted scores

    Returns:
        Tuple of (fpr, tpr, thresholds)
    """
    if len(y_true) == 0 or len(set(y_true)) < 2:
        return (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))

    try:
        return roc_curve(y_true, y_score)
    except ValueError:
        return (np.array([0, 1]), np.array([0, 1]), np.array([0, 1]))


def compute_pr_curve_data(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve data.

    Args:
        y_true: True binary labels
        y_score: Predicted scores

    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if len(y_true) == 0 or len(set(y_true)) < 2:
        prevalence = np.mean(y_true) if len(y_true) > 0 else 0.5
        return (np.array([prevalence, prevalence]), np.array([0, 1]), np.array([0]))

    try:
        return precision_recall_curve(y_true, y_score)
    except ValueError:
        prevalence = np.mean(y_true)
        return (np.array([prevalence, prevalence]), np.array([0, 1]), np.array([0]))


def compute_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute calibration curve (binned predicted vs observed probabilities).

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities [0, 1]
        n_bins: Number of bins

    Returns:
        Tuple of (mean_predicted_probs, fraction_positives)
    """
    if len(y_true) == 0:
        return (np.array([]), np.array([]))

    # Bin predictions
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_predicted = np.zeros(n_bins)
    fraction_positive = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            mean_predicted[i] = y_prob[mask].mean()
            fraction_positive[i] = y_true[mask].mean()

    # Filter out empty bins
    valid_bins = mean_predicted > 0
    return (mean_predicted[valid_bins], fraction_positive[valid_bins])
