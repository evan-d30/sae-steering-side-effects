# Pre-Interventation Prediction of SAE Steering Side Effects

This repository contains experiment notebokos and figure-generation code for the paper:

**Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects**

The project tests whether side effects of sparse-autoencoder (SAE) feature steering can be forecast *before* running an intervention, using cheap feature statistics such as decoder geometry, activation statistics, co-activation structure, and direct-logit footprint.


## Repository status and experiment phases

This repo is script-based. The original experiment notebooks have been removed to keep the repository cleaner; their code cells were converted into Python scripts under `scripts/`.

The code is organized around four experiment phases:

### Phase 1: Feature-level predictor extraction

Phase 1 computes cheap, intervention-free statistics for SAE features before any steering is done.

These predictors describe properties such as:

- how often a feature activates
- how strong or sparse its activations are
- how much it co-activates with other features
- how large its direct logit effect is
- how concentrated or diffuse its downstream effects appear to be

The goal of Phase 1 is to describe each SAE feature using measurable signals that can be computed without running expensive causal interventions.

### Phase 2: Steering label construction

Phase 2 builds labels for whether each feature appears causally useful, clean, messy, or risky when used for steering.

This phase evaluates what happens when a feature is amplified and measures outcomes such as:

- whether the target behavior improves
- whether the effect is stable across contexts
- whether the feature causes collateral side effects
- whether unrelated behavior is degraded

The goal of Phase 2 is to create the ground-truth steering outcomes that Phase 1 predictors will later try to predict.

### Phase 3: Predictive evaluation

Phase 3 tests whether the Phase 1 predictors can predict the Phase 2 steering outcomes.

This is the main predictive part of the project. It asks:

> Can we tell in advance which SAE features are likely to steer cleanly, before actually steering with them?

The scripts evaluate correlations, regression models, cross-validation performance, baseline comparisons, and residualized analyses.

### Phase 4: Held-out causal screening

Phase 4 applies the learned predictors to held-out SAE features that were not used during the earlier evaluation.

The model selects features predicted to be cleaner or messier, then tests them with fresh steering runs.

The goal of Phase 4 is to check whether the predictor is practically useful for screening new features, rather than only explaining features already seen during analysis.

Some scripts were converted from the interactive cloud runs used during development, so check model paths, SAE paths, and output directories before launching long jobs on a new machine.

## Directory structure

```text
configs/                 Model and experiment configuration summaries
scripts/phase123/        Phase 1 predictor, Phase 2 steering, Phase 3 prediction scripts
scripts/phase4/          Held-out screening scripts
scripts/figures/         Figure-generation scripts
scripts/summarize_results.py
results/                 Available here or in the manuscript (Appendix + Results)
figures/                 Available here or in the manuscript (Appendix + Results)
paper/                   Manuscript file
```


## Installation

Create a fresh Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For Llama-3.1-8B and Gemma, you may need to authenticate with Hugging Face and accept model license terms:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

Do **not** commit tokens, `.env` files, Hugging Face caches, model weights, SAE checkpoints, or other private files.

## Basic reproduction flow

For each model:

1. Run the relevant Phase 1/2/3 script in `scripts/phase123/`.
2. Confirm that Phase 1 predictor files, Phase 2 steering-label files, and Phase 3 evaluation files are saved.
3. Run the corresponding Phase 4 script in `scripts/phase4/`.
4. Copy final CSV/JSON outputs into `results/<model_name>/` if you want a consolidated result folder.
5. Run `scripts/figures/figures_3_4_5_with_llama.py` to regenerate main figures.

Example:

```bash
python scripts/phase123/llama31_8b_phase123.py
python scripts/phase4/llama31_8b_phase4_screening.py
python scripts/figures/figures_3_4_5_with_llama.py
```

## Expected result files

Typical Phase 1/2/3 outputs:

```text
phase1_predictors.csv
phase2_steering_labels.csv
robust_phase2_phase3_merged.csv
robust_phase3_baseline_comparison.csv
robust_phase3_univariate_correlations.csv
robust_phase3_cv_regression_results.csv
robust_phase3_residualized_target_results.csv
robust_phase123_summary.json
```

Typical Phase 4 outputs:

```text
phase4_selected_features.csv
phase4_fresh_steering_eval.csv
phase4_group_tests.csv
phase4_screening_summary.csv
phase4_verdict.json
```

## Notes

The scripts are research artifacts converted from the original experimental runs rather than a polished Python package. For exact manuscript reproduction, use the same model checkpoints, SAE releases, feature counts, context counts, and random seeds reported in the paper/config files.

