"""
Constants and configuration values used across the package.
"""

# Random seed for reproducibility
DEFAULT_SEED = 42

# ECI collapse threshold (from paper)
ECI_COLLAPSE_THRESHOLD = -0.02

# Figure generation defaults
DEFAULT_DPI = 600
SMOKE_DPI = 150
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_WIDE = (18, 6)

# Color-blind safe palette (based on Wong 2011)
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'grey': '#949494',
    'black': '#000000',
}

# CSV column expectations
INTERNAL_COLS_REQUIRED = [
    'prompt_id',
    'model_name',
    'eci_raw',
    'eci_residualized',
    'early_eci_raw',
    'effective_ranks',
]

EXTERNAL_COLS_REQUIRED = [
    'prompt_id',
    'model_name',
    'qa_failure',
]

# Statistical parameters
BOOTSTRAP_N_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
CALIBRATION_N_BINS = 10
CI_ALPHA = 0.05  # for 95% confidence intervals
