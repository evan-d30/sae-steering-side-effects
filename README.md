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
results/                 Place generated result CSV/JSON files here
figures/                 Place generated figures here
paper/                   Optional manuscript files
```

