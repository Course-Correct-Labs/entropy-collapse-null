"""
Figure generation for "No Evidence for Epistemic Entropy Collapse in Small Open Language Models"

This module generates all three figures from the paper:
- Figure 1: ECI histograms (phi-2 vs control)
- Figure 2: Effective rank trajectories
- Figure 3: Failure prediction (ROC + PR curves with calibration)
"""

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

from .constants import COLORS, DEFAULT_DPI, ECI_COLLAPSE_THRESHOLD, CALIBRATION_N_BINS
from .eci import classify_collapse
from .bootstrap import (
    bootstrap_roc_auc,
    bootstrap_pr_auc,
    compute_roc_curve_data,
    compute_pr_curve_data,
    compute_calibration,
)


def setup_matplotlib(dpi: int = None):
    """Configure matplotlib for publication-quality figures."""
    if dpi is None:
        dpi = DEFAULT_DPI
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
    })


def figure1_eci_histograms(df: pd.DataFrame, output_path: Path, bins: int = 30, dpi: int = None):
    """
    Generate Figure 1: ECI distribution histograms for phi-2 vs control.

    Args:
        df: Merged metrics DataFrame with 'model_name' and 'eci_residualized' columns
        output_path: Path to save figure (e.g., 'fig1_eci_histograms.png')
        bins: Number of histogram bins
        dpi: DPI for output figure
    """
    setup_matplotlib(dpi)

    # Filter data
    df_phi2 = df[df['model_name'] == 'microsoft/phi-2']
    df_control = df[df['model_name'] == 'mistralai/Mistral-7B-v0.1']

    eci_phi2 = df_phi2['eci_residualized'].values
    eci_control = df_control['eci_residualized'].values

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Phi-2 histogram
    ax = axes[0]
    ax.hist(eci_phi2, bins=bins, color=COLORS['blue'], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(ECI_COLLAPSE_THRESHOLD, color=COLORS['red'], linestyle='--', linewidth=1.5, label=f'Collapse threshold ({ECI_COLLAPSE_THRESHOLD})')
    ax.set_xlabel('Residualized ECI')
    ax.set_ylabel('Frequency')
    ax.set_title('microsoft/phi-2', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Control histogram
    ax = axes[1]
    ax.hist(eci_control, bins=bins, color=COLORS['orange'], alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvline(ECI_COLLAPSE_THRESHOLD, color=COLORS['red'], linestyle='--', linewidth=1.5, label=f'Collapse threshold ({ECI_COLLAPSE_THRESHOLD})')
    ax.set_xlabel('Residualized ECI')
    ax.set_title('mistralai/Mistral-7B-v0.1 (Control)', fontweight='bold')
    ax.legend(loc='upper left', frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_dpi = dpi if dpi is not None else DEFAULT_DPI
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
    plt.close()

    print(f"Figure 1 saved to {output_path}")


def figure2_effective_rank_trajectories(df: pd.DataFrame, output_path: Path, n_examples: int = 50, dpi: int = None):
    """
    Generate Figure 2: Effective rank trajectories over token generation.

    Shows spaghetti plot of effective_ranks trajectories for collapsed vs non-collapsed sequences.

    Args:
        df: Metrics DataFrame with 'eci_residualized', 'effective_ranks' columns
        output_path: Path to save figure
        n_examples: Number of example trajectories to plot per group
        dpi: DPI for output figure
    """
    setup_matplotlib(dpi)

    # Classify sequences
    df = df.copy()
    df['collapsed'] = df['eci_residualized'].apply(classify_collapse)

    df_collapsed = df[df['collapsed'] == True]
    df_normal = df[df['collapsed'] == False]

    # Sample trajectories
    sample_collapsed = df_collapsed.sample(n=min(n_examples, len(df_collapsed)), random_state=42)
    sample_normal = df_normal.sample(n=min(n_examples, len(df_normal)), random_state=42)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    # Plot collapsed trajectories
    ax = axes[0]
    for idx, row in sample_collapsed.iterrows():
        ranks = row['effective_ranks']
        if isinstance(ranks, list) and len(ranks) > 0:
            x = np.arange(len(ranks))
            ax.plot(x, ranks, color=COLORS['red'], alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Token window index')
    ax.set_ylabel('Effective rank')
    ax.set_title(f'Collapsed sequences (ECI < {ECI_COLLAPSE_THRESHOLD})', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot normal trajectories
    ax = axes[1]
    for idx, row in sample_normal.iterrows():
        ranks = row['effective_ranks']
        if isinstance(ranks, list) and len(ranks) > 0:
            x = np.arange(len(ranks))
            ax.plot(x, ranks, color=COLORS['blue'], alpha=0.3, linewidth=0.8)

    ax.set_xlabel('Token window index')
    ax.set_title(f'Normal sequences (ECI ≥ {ECI_COLLAPSE_THRESHOLD})', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_dpi = dpi if dpi is not None else DEFAULT_DPI
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
    plt.close()

    print(f"Figure 2 saved to {output_path}")


def figure3_failure_prediction_panel(df: pd.DataFrame, output_path: Path, dpi: int = None):
    """
    Generate Figure 3: Failure prediction panel (ROC, PR, calibration).

    Three-panel figure showing:
    - Panel A: ROC curve with AUC and bootstrap CI
    - Panel B: Precision-Recall curve with AP and bootstrap CI
    - Panel C: Calibration curve

    Args:
        df: Merged metrics with 'eci_residualized' and 'qa_failure' columns
        output_path: Path to save figure
        dpi: DPI for output figure
    """
    setup_matplotlib(dpi)

    # Prepare data
    y_true = df['qa_failure'].astype(int).values
    y_score = -df['eci_residualized'].values  # Lower ECI → higher failure risk
    y_prob = 1 / (1 + np.exp(-(-df['eci_residualized'].values * 10)))  # Sigmoid transform for calibration

    # Compute metrics
    roc_auc, roc_lower, roc_upper = bootstrap_roc_auc(y_true, y_score)
    pr_auc, pr_lower, pr_upper = bootstrap_pr_auc(y_true, y_score)

    fpr, tpr, _ = compute_roc_curve_data(y_true, y_score)
    precision, recall, _ = compute_pr_curve_data(y_true, y_score)
    mean_pred, frac_pos = compute_calibration(y_true, y_prob, n_bins=CALIBRATION_N_BINS)

    # Create figure with 3 panels
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # Panel A: ROC curve
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(fpr, tpr, color=COLORS['blue'], linewidth=2, label=f'ROC-AUC = {roc_auc:.3f} [{roc_lower:.3f}, {roc_upper:.3f}]')
    ax1.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--', linewidth=1, label='Chance')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('(A) ROC Curve', fontweight='bold', loc='left')
    ax1.legend(loc='lower right', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Panel B: Precision-Recall curve
    ax2 = fig.add_subplot(gs[1])
    baseline = np.mean(y_true)
    ax2.plot(recall, precision, color=COLORS['orange'], linewidth=2, label=f'PR-AUC = {pr_auc:.3f} [{pr_lower:.3f}, {pr_upper:.3f}]')
    ax2.axhline(baseline, color=COLORS['gray'], linestyle='--', linewidth=1, label=f'Baseline ({baseline:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('(B) Precision-Recall Curve', fontweight='bold', loc='left')
    ax2.legend(loc='upper right', frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # Panel C: Calibration curve
    ax3 = fig.add_subplot(gs[2])
    if len(mean_pred) > 0:
        ax3.plot(mean_pred, frac_pos, 'o-', color=COLORS['purple'], linewidth=2, markersize=6, label='Observed')
    ax3.plot([0, 1], [0, 1], color=COLORS['gray'], linestyle='--', linewidth=1, label='Perfect calibration')
    ax3.set_xlabel('Predicted probability')
    ax3.set_ylabel('Observed fraction of positives')
    ax3.set_title('(C) Calibration Curve', fontweight='bold', loc='left')
    ax3.legend(loc='upper left', frameon=False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    save_dpi = dpi if dpi is not None else DEFAULT_DPI
    plt.savefig(output_path, dpi=save_dpi, bbox_inches='tight')
    plt.close()

    print(f"Figure 3 saved to {output_path}")
    print(f"  ROC-AUC: {roc_auc:.3f} [{roc_lower:.3f}, {roc_upper:.3f}]")
    print(f"  PR-AUC:  {pr_auc:.3f} [{pr_lower:.3f}, {pr_upper:.3f}]")


def generate_all_figures(run_dir: Path, output_dir: Path, smoke: bool = False, dpi: int = None):
    """
    Generate all three figures from paper.

    Args:
        run_dir: Path to run directory with metrics CSVs
        output_dir: Directory to save figures
        smoke: If True, use subsampled data for fast smoke test
        dpi: DPI for output figures (default: 600)
    """
    from .utils import load_metrics_internal, load_metrics_external, merge_metrics, subsample_data

    print("Loading data...")
    df_internal = load_metrics_internal(run_dir)
    df_external = load_metrics_external(run_dir)
    df = merge_metrics(df_internal, df_external)

    if smoke:
        print("Smoke test mode: subsampling data to 5%...")
        df = subsample_data(df, frac=0.05, min_rows=30)

    print(f"Loaded {len(df)} sequences")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add suffix for smoke test figures
    suffix = '_smoke' if smoke else ''

    print("\nGenerating Figure 1: ECI histograms...")
    figure1_eci_histograms(df, output_dir / f'fig1_eci_histograms{suffix}.png', dpi=dpi)

    print("\nGenerating Figure 2: Effective rank trajectories...")
    figure2_effective_rank_trajectories(df, output_dir / f'fig2_effective_rank_trajectories{suffix}.png', dpi=dpi)

    print("\nGenerating Figure 3: Failure prediction panel...")
    figure3_failure_prediction_panel(df, output_dir / f'fig3_failure_prediction_panel{suffix}.png', dpi=dpi)

    print(f"\nAll figures saved to {output_dir}")
