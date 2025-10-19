# Figure Captions

## Figure 1: ECI Distribution Histograms

**Caption:** Distribution of residualized Epistemic Collapse Index (ECI) values for microsoft/phi-2 (left, blue) and Qwen/Qwen2.5-0.5B control (right, orange). ECI measures the slope of effective rank over token generation, with negative values indicating declining representational diversity. The red dashed line marks the hypothesized collapse threshold (ECI < -0.02). Both models show similar distributions centered near zero, with no evidence of systematic collapse in either condition.

**File:** `fig1_eci_histograms.png`

---

## Figure 2: Effective Rank Trajectories

**Caption:** Representative effective rank trajectories over token generation for sequences classified as "collapsed" (left, ECI < -0.02, red) versus "normal" (right, ECI ≥ -0.02, blue). Each line represents one sequence, with n=50 examples sampled per group. Effective rank quantifies the diversity of hidden state representations, with higher values indicating more diffuse activation patterns. Both groups show substantial trajectory variability with no distinctive qualitative pattern distinguishing collapsed from normal sequences.

**File:** `fig2_effective_rank_trajectories.png`

---

## Figure 3: Failure Prediction Performance

**Caption:** Predictive utility of ECI for identifying QA task failures. **(A)** ROC curve with area under curve (AUC) and 95% bootstrap confidence interval. **(B)** Precision-Recall curve with average precision (AP) and confidence interval; dashed line indicates baseline precision (prevalence). **(C)** Calibration curve showing predicted probability (from sigmoid-transformed ECI) versus observed fraction of failures; dashed line indicates perfect calibration. All metrics indicate near-chance performance (ROC-AUC ≈ 0.50, PR-AUC ≈ baseline), with poor calibration, demonstrating that ECI does not reliably predict behavioral failure.

**File:** `fig3_failure_prediction_panel.png`

---

## Notes

- All figures generated at 600 DPI for publication quality
- Color palette designed for color-blind accessibility
- Confidence intervals computed via bootstrap resampling (1000 iterations)
- ECI collapse threshold (-0.02) adopted from prior literature
