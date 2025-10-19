# Release Notes - v0.1.0

**Initial Public Release**

This is the first public release of the reproducibility package for the paper "No Evidence for Epistemic Entropy Collapse in Small Open Language Models" by Bentley DeVilling (Course Correct Labs).

## What's Included

- **Complete Python package** for figure generation and analysis
- **Preprocessed data** (n=346 sequences) from phi-2 and Mistral-7B models
- **Three publication-quality figures** (600 DPI):
  - Figure 1: ECI distribution histograms
  - Figure 2: Effective rank trajectories  
  - Figure 3: Failure prediction panel (ROC + PR + calibration)
- **CI/CD pipeline** with smoke tests
- **Complete documentation** with quickstart and troubleshooting

## Key Features

- One-command reproduction: `python -m src.cli reproduce --in runs/affordable --out runs/affordable/figures --dpi 600`
- Fast smoke test (<5 min): `bash scripts/run_smoke.sh`
- Configurable DPI output (300-600)
- Bootstrap confidence intervals for ROC-AUC and PR-AUC
- Color-blind safe palette

## Results

- **Mean ECI:** −0.001 (SD ≈ 0.025)
- **Collapse rate:** ~9.8% sequences with ECI < −0.02
- **ROC-AUC:** ≈ 0.454 (95% CI [0.41, 0.50]) — near chance
- **Conclusion:** No evidence for systematic epistemic collapse

## Installation

```bash
conda env create -f environment.yml
conda activate eec-null
bash scripts/reproduce_all_figures.sh
```

## License

- **Code:** Apache-2.0
- **Data & Paper:** CC BY-SA 4.0

## Citation

```bibtex
@article{devilling2025entropy,
  title={No Evidence for Epistemic Entropy Collapse in Small Open Language Models},
  author={DeVilling, Bentley},
  year={2025},
  organization={Course Correct Labs}
}
```

## Contact

Bentley DeVilling  
Course Correct Labs  
bentley@coursecorrectlabs.com
