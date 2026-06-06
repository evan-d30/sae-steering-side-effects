# Reproduction Notes

## Hardware

Small-model experiments can usually run on Colab-class GPUs. Gemma-2-2B and Llama-3.1-8B are better run on A100/H100/H200-class GPUs.

The Llama scripts in this repository are configured for the final Llama setting used in the manuscript:

```text
model: meta-llama/Llama-3.1-8B
primary SAE: llama_scope_lxr_8x / l16r_8x
downstream SAE: llama_scope_lxr_8x / l20r_8x
hooks: blocks.16.hook_resid_post, blocks.20.hook_resid_post
batch_contexts: 16
batch_steer: 16
```

Before running full Llama experiments, perform a reconstruction sanity check to verify that the selected Llama Scope SAE reconstructs cached residual activations reasonably and that additive steering produces nonzero logit effects.

## Environment

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For gated models, log in to Hugging Face before loading the model:

```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")
```

Never commit Hugging Face tokens, API keys, model weights, SAE checkpoints, or local caches.

## Running order

For each model:

1. Run Phase 1 clean pass and predictor construction.
2. Run Phase 2 steering-label construction.
3. Run Phase 3 predictive evaluation and residualized robustness.
4. Run Phase 4 held-out screening.
5. Copy outputs into `results/<model>/`.
6. Run the figure-generation script.

## Included scripts

```text
scripts/phase123/gpt2_small_phase123.py
scripts/phase123/pythia70m_phase123.py
scripts/phase123/gemma2_2b_phase123.py
scripts/phase123/llama31_8b_phase123.py
scripts/phase4/gpt2_phase4_screening.py
scripts/phase4/pythia70m_phase4_screening.py
scripts/phase4/gemma2_2b_phase4_screening.py
scripts/phase4/llama31_8b_phase4_screening.py
scripts/figures/figures_3_4_5_with_llama.py
scripts/summarize_results.py
```

These scripts were converted from the original interactive cloud notebooks. They preserve the original code cells, but output paths may need to be edited depending on your local or cloud filesystem.

## Summarizing outputs

After copying model result folders into `results/`, run:

```bash
python scripts/summarize_results.py results/
```
