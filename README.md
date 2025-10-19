# No Evidence for Epistemic Entropy Collapse in Small Open Language Models

[![CI](https://github.com/Course-Correct-Labs/entropy-collapse-null/actions/workflows/ci.yml/badge.svg)](https://github.com/Course-Correct-Labs/entropy-collapse-null/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![DOI](https://zenodo.org/badge/Course-Correct-Labs/entropy-collapse-null.svg)](https://doi.org/10.5281/zenodo.PLACEHOLDER)

**Author:** Bentley DeVilling
**Affiliation:** Course Correct Labs
**Contact:** bentley@coursecorrectlabs.com

---

## Overview

This repository contains the complete code and data to reproduce all figures and analyses from the paper:

> **"No Evidence for Epistemic Entropy Collapse in Small Open Language Models"**
> DeVilling, B. (2025). *Course Correct Labs*.

**Key Finding:** We find no evidence that small open-source language models (microsoft/phi-2, mistralai/Mistral-7B-v0.1) exhibit "epistemic entropy collapse" — a hypothesized phenomenon where hidden state representations progressively lose diversity during text generation, leading to behavioral failure.

## Results at a Glance

- **Mean ECI:** −0.001 (SD ≈ 0.025)
- **Collapse rate:** ~9.8% of sequences with ECI < −0.02
- **Predictive utility:** ROC-AUC ≈ 0.454 (95% CI [0.41, 0.50]) — near chance
- **Effective rank trajectories:** Flat across generation, no systematic decline

---

## Quick Start

### One-Command Reproduction

```bash
# Clone repository
git clone https://github.com/Course-Correct-Labs/entropy-collapse-null.git
cd entropy-collapse-null

# Set up environment
conda env create -f environment.yml
conda activate eec-null

# Reproduce all figures from paper
bash scripts/reproduce_all_figures.sh
```

**Output:** Three publication-quality figures (600 DPI) in `figures/`:
- `fig1_eci_histograms.png`
- `fig2_effective_rank_trajectories.png`
- `fig3_failure_prediction_panel.png`

---

## Installation

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate eec-null
```

### Option 2: Pip

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Installation

```bash
bash scripts/verify_environment.sh
```

**Expected output:**
```
✓ All required packages installed!
```

---

## Repository Structure

```
entropy-collapse-null/
├── src/                          # Python package
│   ├── __init__.py              # Package initialization
│   ├── constants.py             # Configuration and constants
│   ├── utils.py                 # Data loading and validation
│   ├── metrics_internal.py      # Effective rank, participation ratio
│   ├── metrics_external.py      # ΔI drift, n-gram novelty
│   ├── eci.py                   # Epistemic Collapse Index (ECI)
│   ├── bootstrap.py             # Bootstrap confidence intervals
│   ├── figures.py               # Figure generation
│   └── cli.py                   # Command-line interface
├── scripts/                      # Shell scripts
│   ├── reproduce_all_figures.sh # One-command reproduction
│   ├── run_smoke.sh             # Fast smoke test (<5 min)
│   ├── lint_check.sh            # Code quality checks
│   └── verify_environment.sh    # Environment verification
├── runs/affordable/              # Data directory
│   ├── metrics_internal.csv     # Internal model metrics
│   ├── metrics_external.csv     # External behavioral metrics
│   └── manifest.json            # Run metadata
├── figures/                      # Output directory for figures
├── paper/                        # Paper and documentation
│   └── captions.md              # Figure captions
├── data/                         # Data documentation
│   └── README.md                # Dataset description
├── .github/workflows/            # CI/CD
│   └── ci.yml                   # GitHub Actions workflow
├── environment.yml               # Conda environment
├── requirements.txt              # Pip dependencies
├── LICENSE                       # Apache 2.0 license
├── CITATION.cff                  # Citation metadata
└── README.md                     # This file
```

---

## Usage

### Generate All Figures

```bash
bash scripts/reproduce_all_figures.sh
```

**Runtime:** ~10-15 minutes on CPU
**Output:** `figures/fig1_eci_histograms.png`, `fig2_effective_rank_trajectories.png`, `fig3_failure_prediction_panel.png`

### Smoke Test (Fast Validation)

```bash
bash scripts/run_smoke.sh
```

**Runtime:** <5 minutes
**Purpose:** Validates code correctness on 5% subsample (n≈30)
**Output:** `figures/smoke/` directory

### Python API

```python
from pathlib import Path
from src.figures import generate_all_figures

# Generate all figures
generate_all_figures(
    run_dir=Path("runs/affordable"),
    output_dir=Path("figures"),
    smoke=False  # Set True for fast smoke test
)
```

### Command-Line Interface

```bash
# Full reproduction
python -m src.cli reproduce --run-dir runs/affordable --output-dir figures/

# Smoke test
python -m src.cli smoke --run-dir runs/affordable --output-dir figures/smoke/
```

---

## Figures

### Figure 1: ECI Distribution Histograms

Distribution of residualized Epistemic Collapse Index (ECI) values for microsoft/phi-2 vs control. Both models show similar distributions centered near zero, with no evidence of systematic collapse.

![Figure 1](figures/fig1_eci_histograms.png)

### Figure 2: Effective Rank Trajectories

Representative effective rank trajectories over token generation for "collapsed" vs "normal" sequences. Both groups show substantial variability with no distinctive pattern.

![Figure 2](figures/fig2_effective_rank_trajectories.png)

### Figure 3: Failure Prediction Performance

Predictive utility of ECI for identifying QA task failures. All metrics indicate near-chance performance (ROC-AUC ≈ 0.50), demonstrating that ECI does not reliably predict behavioral failure.

![Figure 3](figures/fig3_failure_prediction_panel.png)

---

## Data

### Dataset Description

The `runs/affordable/` directory contains preprocessed metrics for n=346 sequences:

- **`metrics_internal.csv`**: Internal model metrics
  - Columns: `prompt_id`, `model_name`, `eci_raw`, `eci_residualized`, `effective_ranks`, `participation_ratios`, `variances`

- **`metrics_external.csv`**: External behavioral metrics
  - Columns: `prompt_id`, `model_name`, `qa_failure`, `delta_i_values`, `ngram_novelty_values`, `char_entropy_values`

- **`manifest.json`**: Run metadata (seed, configuration)

See [`data/README.md`](data/README.md) for detailed column descriptions.

### Models Tested

- **microsoft/phi-2** (2.7B parameters): Primary model
- **mistralai/Mistral-7B-v0.1** (7.2B parameters): Control

### Data Availability

Preprocessed metrics (CSVs) are included in this repository. Raw data (hidden states, ~50GB) available upon request: bentley@coursecorrectlabs.com

---

## Methods

### Epistemic Collapse Index (ECI)

ECI measures the rate of change in representational diversity over token generation:

```
ECI = slope(effective_rank ~ token_index)
```

- **Effective rank:** Exponential of Shannon entropy over singular value spectrum
- **Negative ECI:** Declining diversity (hypothesized "collapse")
- **Threshold:** ECI < -0.02 (adopted from prior literature)

### Metrics

**Internal (from hidden states):**
- Effective rank (diversity of representations)
- Participation ratio (dimensionality)
- Variance (activation magnitude)

**External (from generated text):**
- ΔI drift (n-gram divergence)
- N-gram novelty (lexical diversity)
- Character entropy (randomness)
- QA failure (TruthfulQA correctness)

### Analysis Pipeline

1. Extract hidden states at each generation step
2. Compute internal metrics over sliding windows (128 tokens, stride 64)
3. Compute ECI as slope of effective rank trajectory
4. Residualize against control condition (Mistral-7B)
5. Evaluate predictive utility for QA failure (ROC-AUC, PR-AUC)

---

## Development

### Code Quality

```bash
# Run linter
bash scripts/lint_check.sh

# Format code
ruff format src/

# Type checking (optional)
mypy src/
```

### Testing

```bash
# Smoke test
bash scripts/run_smoke.sh

# Full reproduction
bash scripts/reproduce_all_figures.sh
```

### CI/CD

GitHub Actions runs on every push:
- Linting (ruff)
- Smoke test (<5 min)
- Full reproduction (~15 min)

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

---

## Troubleshooting

### Issue: `FileNotFoundError: manifest.json not found`

**Solution:** Ensure data files are in `runs/affordable/`:
```bash
ls runs/affordable/
# Should show: metrics_internal.csv, metrics_external.csv, manifest.json
```

### Issue: `Missing required columns in metrics_internal.csv`

**Solution:** Verify CSV headers match expected format:
```bash
head -n1 runs/affordable/metrics_internal.csv
# Should include: prompt_id, model_name, eci_raw, eci_residualized, ...
```

### Issue: Matplotlib backend errors on headless server

**Solution:** Set non-interactive backend:
```python
import matplotlib
matplotlib.use('Agg')
```

Or set environment variable:
```bash
export MPLBACKEND=Agg
```

### Issue: Smoke test fails with "No matching rows found"

**Cause:** Internal and external CSVs have mismatched `prompt_id` or `model_name` values

**Solution:** Check merge keys:
```python
import pandas as pd
df_int = pd.read_csv('runs/affordable/metrics_internal.csv')
df_ext = pd.read_csv('runs/affordable/metrics_external.csv')
print(set(df_int['prompt_id']) - set(df_ext['prompt_id']))
```

---

## Citation

If you use this code or data, please cite:

**BibTeX:**
```bibtex
@article{devilling2025entropy,
  title={No Evidence for Epistemic Entropy Collapse in Small Open Language Models},
  author={DeVilling, Bentley},
  year={2025},
  organization={Course Correct Labs}
}
```

**APA:**
```
DeVilling, B. (2025). No Evidence for Epistemic Entropy Collapse in Small Open Language Models. Course Correct Labs.
```

---

## License

**Code:** Apache License 2.0 (see [`LICENSE`](LICENSE))
**Data:** CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
**Paper/Figures:** CC BY-SA 4.0

---

## Release Checklist

When preparing a release:

- [ ] Run smoke test: `bash scripts/run_smoke.sh`
- [ ] Run full reproduction: `bash scripts/reproduce_all_figures.sh`
- [ ] Verify all figures generated correctly
- [ ] Run linter: `bash scripts/lint_check.sh`
- [ ] Check CI passes on GitHub
- [ ] Update version in `src/__init__.py` and `CITATION.cff`
- [ ] Tag release: `git tag v1.0.0 && git push --tags`
- [ ] Create GitHub release with figures attached
- [ ] Upload to Zenodo for DOI
- [ ] Update DOI badge in README.md

---

## Zenodo Upload Instructions

1. Go to https://zenodo.org/deposit/new
2. Upload release tarball or link GitHub repository
3. Fill metadata:
   - **Title:** "No Evidence for Epistemic Entropy Collapse in Small Open Language Models"
   - **Authors:** Bentley DeVilling (Course Correct Labs)
   - **Description:** See abstract from paper
   - **License:** Apache-2.0 (code), CC-BY-SA-4.0 (data/paper)
   - **Keywords:** language models, epistemic collapse, effective rank, interpretability
4. Publish to mint DOI
5. Update DOI badge in README.md:
   ```markdown
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
   ```

---

## Contact

**Bentley DeVilling**
Course Correct Labs
bentley@coursecorrectlabs.com
https://coursecorrectlabs.com

For questions, issues, or collaboration inquiries, please open a GitHub issue or email directly.

---

**Last updated:** January 2025
