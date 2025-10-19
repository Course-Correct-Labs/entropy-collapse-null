# Dataset Description

This directory contains the preprocessed metrics data used in the paper "No Evidence for Epistemic Entropy Collapse in Small Open Language Models."

## Files

The data is located in `../runs/affordable/`:

- **`metrics_internal.csv`**: Internal model metrics computed from hidden states
  - `prompt_id`: Unique sequence identifier
  - `model_name`: Model used ('microsoft/phi-2' or 'mistralai/Mistral-7B-v0.1')
  - `control`: Control condition label (always 'none' in this dataset)
  - `mode`: Generation mode (always 'natural_long')
  - `eci_raw`: Raw Epistemic Collapse Index (slope of effective rank over tokens)
  - `eci_residualized`: Control-adjusted ECI
  - `early_eci_raw`: ECI computed on early window only (first 256 tokens)
  - `effective_ranks`: List of effective rank values over sliding windows (128-token windows, 64-token stride)
  - `participation_ratios`: List of participation ratio values
  - `variances`: List of variance values
  - `window_starts`: List of window start token indices
  - `window_ends`: List of window end token indices

- **`metrics_external.csv`**: External behavioral metrics computed from generated text
  - `prompt_id`: Unique sequence identifier (matches internal)
  - `model_name`: Model used ('microsoft/phi-2' or 'mistralai/Mistral-7B-v0.1')
  - `control`: Control condition label (always 'none')
  - `mode`: Generation mode (always 'natural_long')
  - `delta_i_values`: List of ΔI drift values (n-gram divergence between windows)
  - `ngram_novelty_values`: List of n-gram novelty values
  - `char_entropy_values`: List of character entropy values
  - `qa_failure`: Boolean indicating QA task failure
  - `reasoning_failures`: Dict of reasoning failure types (JSON-encoded)

- **`manifest.json`**: Run metadata and configuration
  - Experimental settings (random seed=13, window size=128, stride=64)
  - Data collection parameters
  - Model completion status

## Data Collection

Data was collected by running open-source language models on QA prompts from the TruthfulQA dataset. Hidden states were extracted at each generation step, and metrics were computed over sliding windows (128 tokens, stride 64).

### Models Tested

- **microsoft/phi-2** (2.7B parameters): Primary model of interest
  - Status: **Complete** (200/200 sequences)
- **mistralai/Mistral-7B-v0.1** (7.2B parameters): Control condition
  - Status: **Partial** (146/200 sequences due to computational constraints)
- **google/gemma-2b** (2.5B parameters): Attempted but **failed** due to gated access restrictions
  - Status: **Not included in dataset**

### Sample Size

- **n = 346 sequences total**
  - 200 sequences from microsoft/phi-2
  - 146 sequences from mistralai/Mistral-7B-v0.1
- Both models tested on identical prompts (first 146 prompts overlap completely)
- Sequences length-matched (800 tokens generated per prompt)

### Generation Parameters

- Temperature: 0.3
- Top-p: 0.95
- Repetition penalty: 1.1
- Max new tokens: 800
- Sampling: True

## Reproducibility

The raw data (hidden states, generated text) is large (~50GB). This repository includes only the preprocessed metrics CSV files sufficient to reproduce all figures and analyses from the paper.

For access to raw data or to discuss data reuse, contact: bentley@coursecorrectlabs.com

## Schema Validation

Column names and data types are strictly validated at load time. See `src/utils.py` for schema enforcement:
- `INTERNAL_COLS_REQUIRED` defines expected columns for metrics_internal.csv
- `EXTERNAL_COLS_REQUIRED` defines expected columns for metrics_external.csv
- Missing or misnamed columns will raise `ValueError` with clear error messages

## License

Data: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

You are free to share and adapt this data with attribution and share-alike requirements.

**Citation:**

```
DeVilling, B. (2025). No Evidence for Epistemic Entropy Collapse in Small Open Language Models.
Course Correct Labs Technical Report CCTR-001.
```
