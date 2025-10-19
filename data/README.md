# Dataset Description

This directory contains the preprocessed metrics data used in the paper "No Evidence for Epistemic Entropy Collapse in Small Open Language Models."

## Files

The data is located in `../runs/affordable/`:

- **`metrics_internal.csv`**: Internal model metrics computed from hidden states
  - `prompt_id`: Unique sequence identifier
  - `model_name`: Model used (e.g., 'microsoft/phi-2', 'Qwen/Qwen2.5-0.5B')
  - `eci_raw`: Raw Epistemic Collapse Index (slope of effective rank over tokens)
  - `eci_residualized`: Control-adjusted ECI
  - `early_eci_raw`: ECI computed on early window only
  - `effective_ranks`: List of effective rank values over sliding windows
  - `participation_ratios`: List of participation ratio values
  - `variances`: List of variance values
  - `window_starts`: List of window start token indices
  - `window_ends`: List of window end token indices

- **`metrics_external.csv`**: External behavioral metrics computed from generated text
  - `prompt_id`: Unique sequence identifier (matches internal)
  - `model_name`: Model used
  - `qa_failure`: Boolean indicating QA task failure
  - `reasoning_failures`: Dict of reasoning failure types
  - `delta_i_values`: List of ΔI drift values (n-gram divergence between windows)
  - `ngram_novelty_values`: List of n-gram novelty values
  - `char_entropy_values`: List of character entropy values

- **`manifest.json`**: Run metadata and configuration
  - Experimental settings (random seed, window sizes, models tested)
  - Data collection parameters

## Data Collection

Data was collected by running open-source language models (phi-2, Qwen2.5-0.5B) on QA prompts from the TruthfulQA dataset. Hidden states were extracted at each generation step, and metrics were computed over sliding windows.

### Models Tested

- **microsoft/phi-2** (2.7B parameters): Primary model of interest
- **Qwen/Qwen2.5-0.5B** (0.5B parameters): Control condition

### Sample Size

- n = 346 sequences total
- Both models tested on identical prompts
- Sequences length-matched (256 tokens generated per prompt)

## Reproducibility

The raw data (hidden states, generated text) is large (~50GB). This repository includes only the preprocessed metrics CSV files sufficient to reproduce all figures and analyses from the paper.

For access to raw data or to discuss data reuse, contact: bentley@coursecorrectlabs.com

## License

Data: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

You are free to share and adapt this data with attribution and share-alike requirements.

**Citation:**

```
DeVilling, B. (2025). No Evidence for Epistemic Entropy Collapse in Small Open Language Models. Course Correct Labs.
```
