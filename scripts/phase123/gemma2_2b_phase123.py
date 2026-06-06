#!/usr/bin/env python
# Converted from notebooks/phase123/gemma2_2b_phase123.ipynb
# Original notebook removed from the repository; this script preserves the code cells for reproducibility.


################################################################################
# Cell 1
################################################################################
# Cell 1 — install dependencies

# NOTE: notebook-only command was: !pip install -q sae-lens transformer-lens huggingface_hub scipy scikit-learn tqdm datasets pyarrow


################################################################################
# Cell 2
################################################################################
# Cell 2 — imports and setup

import os
import json
import math
import random
import time
import shutil
from pathlib import Path
from getpass import getpass

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)

pd.set_option("display.max_rows", 220)
pd.set_option("display.max_columns", 220)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

OUT_DIR = Path("/content/gemma_scope_fullscale_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR:", OUT_DIR)


################################################################################
# Cell 3
################################################################################
# Cell 2.5 — optional Hugging Face token login
# Use this if downloads are slow or rate-limited.
# Do NOT hard-code your token.

USE_HF_TOKEN = True

if USE_HF_TOKEN:
    from huggingface_hub import login
    token = getpass("Paste Hugging Face token. Input is hidden: ").strip()
    if token:
        login(token=token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = token
        print("Hugging Face login complete.")
    else:
        print("No token entered. Continuing unauthenticated.")
else:
    print("Skipping Hugging Face login.")


################################################################################
# Cell 4
################################################################################
# Cell 3 — robust experiment configuration

# -----------------------------
# Model / SAE setup
# -----------------------------
MODEL_NAME = "gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"

USE_DOWNSTREAM_SAE = True
DOWNSTREAM_SAE_ID = "layer_16/width_16k/canonical"

# -----------------------------
# Real corpus setup
# -----------------------------
# This should stay True for the robust rerun.
USE_HF_DATASET = True
HF_DATASET_NAME = "Salesforce/wikitext"
HF_DATASET_CONFIG = "wikitext-103-raw-v1"
HF_DATASET_SPLIT = "train[:2%]"   # increase later if needed

MIN_DISTINCT_TEXTS = 1000         # hard stop if corpus is too small
MAX_DISTINCT_TEXTS = 8000

CONTEXT_LEN = 48
N_CONTEXTS = 2048                 # increase to 4096 if runtime allows

# -----------------------------
# Feature sample
# -----------------------------
N_FEATURES = 300
MIN_FINAL_ACT_FREQ = 0.001
MAX_FINAL_ACT_FREQ = 0.50
FEATURE_SAMPLE_SEED = 0

# -----------------------------
# Diverse contexts per feature
# -----------------------------
CONTEXTS_PER_TYPE = 16            # top + random + low => 48 mixed contexts per feature
CONTEXT_TYPES = ["top", "random", "low", "mixed"]

# -----------------------------
# Clamp modes
# -----------------------------
# feature_q95: original per-feature clamp
# fixed_global: same target clamp value for all features
# fixed_global_add: always add the same positive amount; avoids sign-flip artifacts
CLAMP_MODES = ["feature_q95", "fixed_global", "fixed_global_add"]
CLAMP_QUANTILE = 0.95
MIN_CLAMP_VALUE = 1.0

# The fixed global clamp is computed later as median feature_q95 over selected features.
FIXED_GLOBAL_CLAMP_VALUE = None

# -----------------------------
# Runtime
# -----------------------------
BATCH_CONTEXTS = 2
BATCH_STEER = 2
SAVE_EVERY_FEATURES = 5

# -----------------------------
# Metrics
# -----------------------------
TOPK_CONCENTRATION_KS = [10, 50, 100, 500]
ABS_LOGIT_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]
DOWNSTREAM_PANEL_SIZE = 2048
DOWNSTREAM_ABS_THRESHOLDS = [0.01, 0.05, 0.10]
COMPUTE_PCA = False  # Set True only if runtime is okay; SVD is expensive.

# -----------------------------
# Gemma memory/runtime controls
# -----------------------------
MODEL_DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Direct-logit predictors are optional for Gemma.
# Gemma's final logit softcap means absolute d_f @ W_U scale is not directly comparable to GPT-2.
# Shape metrics are still useful. If OOM occurs, set this to False.
COMPUTE_DIRECT_LOGIT_PREDICTORS = True
DIRECT_LOGIT_BATCH_FEATURES = 16

random.seed(FEATURE_SAMPLE_SEED)
np.random.seed(FEATURE_SAMPLE_SEED)
torch.manual_seed(FEATURE_SAMPLE_SEED)

CONFIG = {
    "MODEL_NAME": MODEL_NAME,
    "SAE_RELEASE": SAE_RELEASE,
    "SAE_ID": SAE_ID,
    "USE_DOWNSTREAM_SAE": USE_DOWNSTREAM_SAE,
    "DOWNSTREAM_SAE_ID": DOWNSTREAM_SAE_ID,
    "USE_HF_DATASET": USE_HF_DATASET,
    "HF_DATASET_NAME": HF_DATASET_NAME,
    "HF_DATASET_CONFIG": HF_DATASET_CONFIG,
    "HF_DATASET_SPLIT": HF_DATASET_SPLIT,
    "MIN_DISTINCT_TEXTS": MIN_DISTINCT_TEXTS,
    "MAX_DISTINCT_TEXTS": MAX_DISTINCT_TEXTS,
    "CONTEXT_LEN": CONTEXT_LEN,
    "N_CONTEXTS": N_CONTEXTS,
    "N_FEATURES": N_FEATURES,
    "CONTEXTS_PER_TYPE": CONTEXTS_PER_TYPE,
    "CLAMP_MODES": CLAMP_MODES,
    "CLAMP_QUANTILE": CLAMP_QUANTILE,
    "MIN_CLAMP_VALUE": MIN_CLAMP_VALUE,
    "BATCH_CONTEXTS": BATCH_CONTEXTS,
    "BATCH_STEER": BATCH_STEER,
    "COMPUTE_PCA": COMPUTE_PCA,
    "MODEL_DTYPE": str(MODEL_DTYPE),
    "COMPUTE_DIRECT_LOGIT_PREDICTORS": COMPUTE_DIRECT_LOGIT_PREDICTORS,
    "DIRECT_LOGIT_BATCH_FEATURES": DIRECT_LOGIT_BATCH_FEATURES,
}

print(json.dumps(CONFIG, indent=2))


################################################################################
# Cell 5
################################################################################
# Cell 4 — load model and Gemma Scope SAEs with fixed hook-name mapping

# Gemma-2-2B may require:
# 1. accepting the Gemma license on Hugging Face,
# 2. logging in with a Hugging Face token,
# 3. a GPU runtime with enough memory.
#
# The Gemma Scope SAE ID is a loading path, not the actual TransformerLens hook name.
# This cell maps SAE IDs like "layer_12/width_16k/canonical" to hooks like
# "blocks.12.hook_resid_post".

model_kwargs = {}
try:
    model_kwargs["dtype"] = MODEL_DTYPE
except NameError:
    pass

try:
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, **model_kwargs)
except TypeError:
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)

model.eval()

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device,
)
sae.eval()

def sae_encode(x, which_sae=sae):
    if hasattr(which_sae, "encode"):
        return which_sae.encode(x)
    raise AttributeError("SAE object does not expose .encode(). Check SAELens version.")

def get_decoder_matrix(which_sae):
    W = which_sae.W_dec.detach()
    d_model = model.cfg.d_model
    if W.shape[-1] == d_model:
        return W
    elif W.shape[0] == d_model:
        return W.T
    else:
        raise ValueError(f"Cannot infer W_dec orientation. shape={tuple(W.shape)}, d_model={d_model}")

def infer_gemma_resid_hook(sae_id, model, cfg=None, preferred="hook_resid_post"):
    import re

    hook_names = set(model.hook_dict.keys())

    # First try config names if valid.
    possible_config_names = []
    try:
        possible_config_names.append(sae.cfg.hook_name)
    except Exception:
        pass
    if isinstance(cfg, dict):
        for key in ["hook_name", "hook_point", "act_name"]:
            if key in cfg:
                possible_config_names.append(cfg[key])

    for name in possible_config_names:
        if isinstance(name, str) and name in hook_names:
            return name

    match = re.search(r"layer_(\d+)", sae_id)
    if match is None:
        raise ValueError(f"Could not parse layer number from SAE_ID={sae_id}")

    layer = int(match.group(1))

    # Gemma Scope residual-stream SAEs are canonically at resid_post.
    candidates = [
        f"blocks.{layer}.{preferred}",
        f"blocks.{layer}.hook_resid_post",
        f"blocks.{layer}.hook_resid_pre",
        f"blocks.{layer}.hook_resid_mid",
    ]

    for cand in candidates:
        if cand in hook_names:
            return cand

    raise KeyError(
        f"Could not find a valid hook for SAE_ID={sae_id}. "
        f"Tried {candidates}. First model hooks: {list(hook_names)[:20]}"
    )

HOOK_NAME = infer_gemma_resid_hook(SAE_ID, model, cfg=cfg_dict, preferred="hook_resid_post")

W_DEC = get_decoder_matrix(sae).to(device)
D_SAE, D_MODEL = W_DEC.shape

downstream_sae = None
DOWNSTREAM_HOOK_NAME = None

if USE_DOWNSTREAM_SAE:
    try:
        downstream_sae, downstream_cfg_dict, downstream_sparsity = SAE.from_pretrained(
            release=SAE_RELEASE,
            sae_id=DOWNSTREAM_SAE_ID,
            device=device,
        )
        downstream_sae.eval()
        DOWNSTREAM_HOOK_NAME = infer_gemma_resid_hook(
            DOWNSTREAM_SAE_ID,
            model,
            cfg=downstream_cfg_dict,
            preferred="hook_resid_post",
        )

        print("Loaded downstream SAE:", DOWNSTREAM_SAE_ID)
        print("Downstream hook:", DOWNSTREAM_HOOK_NAME)
    except Exception as e:
        print("Could not load downstream SAE. Continuing without downstream-feature targets.")
        print("Error:", repr(e))
        downstream_sae = None
        DOWNSTREAM_HOOK_NAME = None

print("Model:", MODEL_NAME)
print("Primary SAE:", SAE_RELEASE, SAE_ID)
print("Primary hook:", HOOK_NAME)
print("Downstream SAE:", DOWNSTREAM_SAE_ID)
print("Downstream hook:", DOWNSTREAM_HOOK_NAME)
print("W_DEC:", tuple(W_DEC.shape))
print("D_SAE:", D_SAE, "D_MODEL:", D_MODEL)
print("Model dtype request:", model_kwargs.get("dtype", None))

hook_names = set(model.hook_dict.keys())
print("Primary hook exists in model:", HOOK_NAME in hook_names)
if DOWNSTREAM_HOOK_NAME is not None:
    print("Downstream hook exists in model:", DOWNSTREAM_HOOK_NAME in hook_names)

assert HOOK_NAME in hook_names, f"Primary hook not found: {HOOK_NAME}"
if DOWNSTREAM_HOOK_NAME is not None:
    assert DOWNSTREAM_HOOK_NAME in hook_names, f"Downstream hook not found: {DOWNSTREAM_HOOK_NAME}"


################################################################################
# Cell 6
################################################################################
# Cell 5 — load real corpus and create token contexts
# This cell intentionally refuses to silently use the toy fallback.

from datasets import load_dataset

if not USE_HF_DATASET:
    raise ValueError("For robust rerun, USE_HF_DATASET must be True.")

print("Loading dataset...")
ds = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG, split=HF_DATASET_SPLIT)

texts = []
seen = set()

for x in ds:
    txt = x.get("text", "")
    if not isinstance(txt, str):
        continue
    txt = " ".join(txt.strip().split())
    if len(txt) < 100:
        continue
    if txt in seen:
        continue
    seen.add(txt)
    texts.append(txt)
    if len(texts) >= MAX_DISTINCT_TEXTS:
        break

print("distinct usable texts:", len(texts))

if len(texts) < MIN_DISTINCT_TEXTS:
    raise ValueError(
        f"Real corpus too small: only {len(texts)} distinct texts. "
        f"Need at least {MIN_DISTINCT_TEXTS}. Increase HF_DATASET_SPLIT or use another dataset."
    )

random.shuffle(texts)

# Tokenize into one long stream, chunk into fixed contexts.
all_ids = []
for txt in tqdm(texts, desc="tokenizing corpus"):
    ids = model.tokenizer.encode(txt, add_special_tokens=False)
    if len(ids) > 0:
        all_ids.extend(ids + [model.tokenizer.eos_token_id])

bos_id = model.tokenizer.bos_token_id
if bos_id is None:
    bos_id = model.tokenizer.eos_token_id

chunks = []
stride = CONTEXT_LEN - 1

for start in range(0, max(0, len(all_ids) - (CONTEXT_LEN - 1)), stride):
    body = all_ids[start:start + (CONTEXT_LEN - 1)]
    if len(body) == CONTEXT_LEN - 1:
        chunks.append([bos_id] + body)
    if len(chunks) >= N_CONTEXTS:
        break

if len(chunks) < N_CONTEXTS:
    raise ValueError(f"Only created {len(chunks)} chunks, need {N_CONTEXTS}. Increase dataset split.")

tokens = torch.tensor(chunks, dtype=torch.long, device=device)

print("tokens shape:", tuple(tokens.shape))
print("example decoded context:")
print(model.tokenizer.decode(tokens[0].detach().cpu().tolist()[:120]))


################################################################################
# Cell 7
################################################################################
# Cell 6 — collect clean logits, final primary SAE activations, and optional downstream features
# Important: we batch this to avoid materializing full-sequence logits for all contexts.

ACTIVE_THRESH = 1e-6

clean_final_logits_chunks = []
final_primary_feat_chunks = []
downstream_final_feat_chunks = []

names_filter = [HOOK_NAME]
if downstream_sae is not None and DOWNSTREAM_HOOK_NAME is not None:
    names_filter.append(DOWNSTREAM_HOOK_NAME)

for i in tqdm(range(0, tokens.shape[0], BATCH_CONTEXTS), desc="clean pass"):
    batch = tokens[i:i+BATCH_CONTEXTS]
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch, names_filter=names_filter)

    clean_final_logits_chunks.append(logits[:, -1, :].detach().cpu())

    primary_feat = sae_encode(cache[HOOK_NAME], sae)[:, -1, :]
    final_primary_feat_chunks.append(primary_feat.detach().cpu())

    if downstream_sae is not None and DOWNSTREAM_HOOK_NAME in cache:
        downstream_feat = sae_encode(cache[DOWNSTREAM_HOOK_NAME], downstream_sae)[:, -1, :]
        downstream_final_feat_chunks.append(downstream_feat.detach().cpu())

clean_final_logits = torch.cat(clean_final_logits_chunks, dim=0)
final_feat_primary = torch.cat(final_primary_feat_chunks, dim=0)

if len(downstream_final_feat_chunks) > 0:
    feat_downstream_clean = torch.cat(downstream_final_feat_chunks, dim=0)
else:
    feat_downstream_clean = None

print("clean_final_logits:", tuple(clean_final_logits.shape))
print("final_feat_primary:", tuple(final_feat_primary.shape))
print("feat_downstream_clean:", None if feat_downstream_clean is None else tuple(feat_downstream_clean.shape))


################################################################################
# Cell 8
################################################################################
# Cell 7 — select feature sample and compute fixed global clamp

final_active = (final_feat_primary > ACTIVE_THRESH).float()
final_act_freq = final_active.mean(dim=0).numpy()
final_act_mean = final_feat_primary.mean(dim=0).numpy()
final_act_std = final_feat_primary.std(dim=0).numpy()
final_act_max = final_feat_primary.max(dim=0).values.numpy()
final_act_q95 = torch.quantile(final_feat_primary.float(), CLAMP_QUANTILE, dim=0).numpy()

feature_stats = pd.DataFrame({
    "feature": np.arange(D_SAE),
    "final_act_freq": final_act_freq,
    "final_act_mean": final_act_mean,
    "final_act_std": final_act_std,
    "final_act_max": final_act_max,
    "final_act_q95": final_act_q95,
})

eligible = feature_stats[
    (feature_stats["final_act_freq"] >= MIN_FINAL_ACT_FREQ)
    & (feature_stats["final_act_freq"] <= MAX_FINAL_ACT_FREQ)
].copy()

eligible = eligible.sort_values("final_act_freq", ascending=False).reset_index(drop=True)

print("eligible features:", len(eligible))
display(eligible.head(20))

if len(eligible) == 0:
    raise ValueError("No eligible features. Increase N_CONTEXTS or lower MIN_FINAL_ACT_FREQ.")

if len(eligible) <= N_FEATURES:
    selected_features = eligible["feature"].astype(int).tolist()
else:
    idxs = np.linspace(0, len(eligible) - 1, N_FEATURES).astype(int)
    selected_features = eligible.iloc[idxs]["feature"].astype(int).tolist()

selected_feature_stats = feature_stats[feature_stats["feature"].isin(selected_features)].copy().reset_index(drop=True)

# Fixed global clamp: same for every feature.
selected_q95 = selected_feature_stats["final_act_q95"].clip(lower=MIN_CLAMP_VALUE).values
FIXED_GLOBAL_CLAMP_VALUE = float(np.median(selected_q95))

print("selected features:", len(selected_features))
print("fixed global clamp value:", FIXED_GLOBAL_CLAMP_VALUE)
print(selected_features[:30])

selected_feature_stats["fixed_global_clamp_value"] = FIXED_GLOBAL_CLAMP_VALUE
selected_feature_stats_path = OUT_DIR / "robust_selected_feature_stats.csv"
selected_feature_stats.to_csv(selected_feature_stats_path, index=False)
print("saved:", selected_feature_stats_path)


################################################################################
# Cell 9
################################################################################
# Cell 8 — choose downstream feature panel

downstream_panel = None

if feat_downstream_clean is not None:
    downstream_active_freq = (feat_downstream_clean > ACTIVE_THRESH).float().mean(dim=0)
    panel_size = min(DOWNSTREAM_PANEL_SIZE, feat_downstream_clean.shape[-1])
    downstream_panel = torch.topk(downstream_active_freq, k=panel_size).indices.detach().cpu()
    print("downstream panel size:", len(downstream_panel))
else:
    print("No downstream panel.")


################################################################################
# Cell 10
################################################################################
# Cell 9 — context selection per feature

rng = np.random.default_rng(FEATURE_SAMPLE_SEED)

def get_context_indices_for_feature(feat):
    vals = final_feat_primary[:, int(feat)].float()
    n = len(vals)
    k = min(CONTEXTS_PER_TYPE, n)

    top_idx = torch.topk(vals, k=k).indices.detach().cpu().numpy()
    low_idx = torch.topk(-vals, k=k).indices.detach().cpu().numpy()

    exclude = set(top_idx.tolist()) | set(low_idx.tolist())
    available = np.array([i for i in range(n) if i not in exclude])
    if len(available) >= k:
        random_idx = rng.choice(available, size=k, replace=False)
    else:
        random_idx = rng.choice(np.arange(n), size=k, replace=False)

    mixed_idx = np.unique(np.concatenate([top_idx, random_idx, low_idx]))

    return {
        "top": top_idx,
        "random": random_idx,
        "low": low_idx,
        "mixed": mixed_idx,
    }

# Quick sanity check.
example_feat = selected_features[0]
idxs = get_context_indices_for_feature(example_feat)
for name, arr in idxs.items():
    vals = final_feat_primary[arr, example_feat]
    print(name, len(arr), "mean activation:", float(vals.mean()))


################################################################################
# Cell 11
################################################################################
# Cell 10 — steering helper with TransformerLens API-safe hook
# Includes amplify-only mode: fixed_global_add.

def run_feature_clamp(feature_idx, clamp_value, batch_tokens, clamp_mode="feature_q95", return_downstream=False):
    feature_idx = int(feature_idx)
    direction = W_DEC[feature_idx].detach()

    def steering_hook(act, hook):
        feat = sae_encode(act, sae)
        current = feat[:, -1, feature_idx]

        if clamp_mode == "fixed_global_add":
            # Amplify-only intervention:
            # always add +C * d_f, regardless of current activation.
            # This avoids sign flips where high-current contexts get suppressed.
            delta = torch.full_like(current, float(clamp_value)).to(act.dtype)
        else:
            # Clamp-style intervention:
            # set the feature toward the requested target activation.
            delta = (clamp_value - current).to(act.dtype)

        act = act.clone()
        act[:, -1, :] = act[:, -1, :] + delta[:, None] * direction.to(act.dtype)[None, :]
        return act

    if return_downstream and downstream_sae is not None and DOWNSTREAM_HOOK_NAME is not None:
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(HOOK_NAME, steering_hook)]):
                steered_logits, steered_cache = model.run_with_cache(
                    batch_tokens,
                    names_filter=[DOWNSTREAM_HOOK_NAME],
                )
        downstream_acts = steered_cache[DOWNSTREAM_HOOK_NAME]
        downstream_feat = sae_encode(downstream_acts, downstream_sae)[:, -1, :]
        if downstream_panel is not None:
            downstream_feat = downstream_feat[:, downstream_panel.to(downstream_feat.device)]
        return steered_logits[:, -1, :].detach(), downstream_feat.detach()

    else:
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(HOOK_NAME, steering_hook)]):
                steered_logits = model(batch_tokens)
        return steered_logits[:, -1, :].detach(), None


################################################################################
# Cell 12
################################################################################
# Cell 11 — metric helpers

def cosine_to_mean(delta_matrix):
    X = delta_matrix.float()
    row_norms = torch.norm(X, dim=-1)
    mean_effect = X.mean(dim=0)
    mean_norm = torch.norm(mean_effect)
    valid = (row_norms > 1e-8) & (mean_norm > 1e-8)
    if valid.sum().item() == 0:
        return {
            "stability_to_mean_signed_cos": np.nan,
            "stability_to_mean_abs_cos": np.nan,
            "n_valid_effects": 0,
        }
    cos = F.cosine_similarity(X[valid], mean_effect[None, :], dim=-1)
    return {
        "stability_to_mean_signed_cos": float(cos.mean().item()),
        "stability_to_mean_abs_cos": float(cos.abs().mean().item()),
        "n_valid_effects": int(valid.sum().item()),
    }

def mean_pairwise_cosine(delta_matrix):
    X = delta_matrix.float()
    norms = torch.norm(X, dim=-1)
    valid = norms > 1e-8
    X = X[valid]
    n = X.shape[0]
    if n < 2:
        return {
            "pairwise_cos_signed_mean": np.nan,
            "pairwise_cos_abs_mean": np.nan,
        }
    Xn = F.normalize(X, dim=-1)
    sim = Xn @ Xn.T
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    vals = sim[mask]
    return {
        "pairwise_cos_signed_mean": float(vals.mean().item()),
        "pairwise_cos_abs_mean": float(vals.abs().mean().item()),
    }

def pca_first_component_fraction(delta_matrix):
    if not COMPUTE_PCA:
        return np.nan
    X = delta_matrix.float()
    if X.shape[0] < 3:
        return np.nan
    X = X - X.mean(dim=0, keepdim=True)
    try:
        s = torch.linalg.svdvals(X)
        var = s ** 2
        return float((var[0] / var.sum().clamp(min=1e-12)).item())
    except Exception:
        return np.nan

def logit_collateral_metrics(delta_logits, clean_logits, steered_logits):
    delta = delta_logits.float()
    clean = clean_logits.float()
    steered = steered_logits.float()

    abs_delta = delta.abs()
    effect_l2 = torch.norm(delta, dim=-1)
    effect_l1 = abs_delta.sum(dim=-1)
    effect_linf = abs_delta.max(dim=-1).values

    out = {
        "effect_l2_mean": float(effect_l2.mean().item()),
        "effect_l2_std": float(effect_l2.std().item()),
        "effect_l1_mean": float(effect_l1.mean().item()),
        "effect_linf_mean": float(effect_linf.mean().item()),
        "effect_cv": float((effect_l2.std() / effect_l2.mean().clamp(min=1e-8)).item()),
    }

    clean_logp = F.log_softmax(clean, dim=-1)
    steered_logp = F.log_softmax(steered, dim=-1)
    clean_p = clean_logp.exp()
    steered_p = steered_logp.exp()

    out["kl_clean_to_steered_mean"] = float((clean_p * (clean_logp - steered_logp)).sum(dim=-1).mean().item())
    out["kl_steered_to_clean_mean"] = float((steered_p * (steered_logp - clean_logp)).sum(dim=-1).mean().item())

    for thr in ABS_LOGIT_THRESHOLDS:
        out[f"logit_count_abs_delta_gt_{thr}"] = float((abs_delta > thr).float().sum(dim=-1).mean().item())

    total_abs = abs_delta.sum(dim=-1).clamp(min=1e-8)

    for k in TOPK_CONCENTRATION_KS:
        kk = min(k, abs_delta.shape[-1])
        topk_sum = torch.topk(abs_delta, k=kk, dim=-1).values.sum(dim=-1)
        out[f"top{k}_abs_delta_mass_frac"] = float((topk_sum / total_abs).mean().item())

    prob = abs_delta / total_abs[:, None]
    entropy = -(prob.clamp(min=1e-12) * torch.log(prob.clamp(min=1e-12))).sum(dim=-1)
    out["logit_delta_entropy_mean"] = float(entropy.mean().item())
    out["logit_delta_effective_vocab_mean"] = float(torch.exp(entropy).mean().item())

    return out

def downstream_collateral_metrics(delta_feat):
    if delta_feat is None:
        return {}
    delta = delta_feat.float()
    abs_delta = delta.abs()

    l2 = torch.norm(delta, dim=-1)
    l1 = abs_delta.sum(dim=-1)
    linf = abs_delta.max(dim=-1).values
    total_abs = abs_delta.sum(dim=-1).clamp(min=1e-8)
    prob = abs_delta / total_abs[:, None]
    entropy = -(prob.clamp(min=1e-12) * torch.log(prob.clamp(min=1e-12))).sum(dim=-1)

    out = {
        "downstream_feat_l2_mean": float(l2.mean().item()),
        "downstream_feat_l1_mean": float(l1.mean().item()),
        "downstream_feat_linf_mean": float(linf.mean().item()),
        "downstream_feat_entropy_mean": float(entropy.mean().item()),
        "downstream_feat_effective_moved_mean": float(torch.exp(entropy).mean().item()),
    }

    for thr in DOWNSTREAM_ABS_THRESHOLDS:
        out[f"downstream_feat_count_abs_delta_gt_{thr}"] = float((abs_delta > thr).float().sum(dim=-1).mean().item())

    return out


################################################################################
# Cell 13
################################################################################
# Cell 12 — Phase 1 robust label sweep
# Builds labels for feature × clamp_mode × context_type.

label_rows = []
start_time = time.time()

for feat_i, feat in enumerate(tqdm(selected_features, desc="Phase 1 robust labels")):
    feat = int(feat)
    feat_vals = final_feat_primary[:, feat].float()

    feature_q95_clamp = float(torch.quantile(feat_vals, CLAMP_QUANTILE).item())
    feature_q95_clamp = max(feature_q95_clamp, MIN_CLAMP_VALUE)

    clamp_values = {
        "feature_q95": feature_q95_clamp,
        "fixed_global": FIXED_GLOBAL_CLAMP_VALUE,
        "fixed_global_add": FIXED_GLOBAL_CLAMP_VALUE,
    }

    context_indices_by_type = get_context_indices_for_feature(feat)

    for clamp_mode in CLAMP_MODES:
        clamp_value = float(clamp_values[clamp_mode])

        for context_type in CONTEXT_TYPES:
            ctx_idx_np = context_indices_by_type[context_type]
            ctx_idx = torch.tensor(ctx_idx_np, dtype=torch.long)

            ctx_tokens = tokens[ctx_idx.to(tokens.device)]
            clean_logits_ctx = clean_final_logits[ctx_idx].detach().cpu()
            clean_feat_vals_ctx = final_feat_primary[ctx_idx, feat].detach().cpu()

            steered_logits_chunks = []
            downstream_chunks = []

            for b in range(0, ctx_tokens.shape[0], BATCH_STEER):
                batch_tokens = ctx_tokens[b:b+BATCH_STEER]
                steered_logits, downstream_feat = run_feature_clamp(
                    feat,
                    clamp_value,
                    batch_tokens,
                    clamp_mode=clamp_mode,
                    return_downstream=(downstream_panel is not None),
                )
                steered_logits_chunks.append(steered_logits.detach().cpu())

                if downstream_feat is not None:
                    downstream_chunks.append(downstream_feat.detach().cpu())

            steered_logits_ctx = torch.cat(steered_logits_chunks, dim=0)
            delta_logits = steered_logits_ctx - clean_logits_ctx

            if downstream_panel is not None and feat_downstream_clean is not None and len(downstream_chunks) > 0:
                steered_downstream_panel = torch.cat(downstream_chunks, dim=0)
                clean_downstream_panel = feat_downstream_clean[ctx_idx][:, downstream_panel].detach().cpu()
                delta_downstream = steered_downstream_panel - clean_downstream_panel
            else:
                delta_downstream = None

            row = {
                "feature": feat,
                "clamp_mode": clamp_mode,
                "context_type": context_type,
                "n_contexts": int(ctx_tokens.shape[0]),
                "clamp_value": clamp_value,
                "feature_q95_clamp_value": feature_q95_clamp,
                "fixed_global_clamp_value": FIXED_GLOBAL_CLAMP_VALUE,
                "natural_final_act_mean_contexts": float(clean_feat_vals_ctx.mean().item()),
                "natural_final_act_min_contexts": float(clean_feat_vals_ctx.min().item()),
                "natural_final_act_max_contexts": float(clean_feat_vals_ctx.max().item()),
                "final_act_freq_global": float(final_act_freq[feat]),
                "final_act_mean_global": float(final_act_mean[feat]),
                "final_act_std_global": float(final_act_std[feat]),
                "final_act_max_global": float(final_act_max[feat]),
                "final_act_q95_global": float(final_act_q95[feat]),
            }

            row.update(cosine_to_mean(delta_logits))
            row.update(mean_pairwise_cosine(delta_logits))
            row["pca1_variance_frac_logits"] = pca_first_component_fraction(delta_logits)
            row.update(logit_collateral_metrics(delta_logits, clean_logits_ctx, steered_logits_ctx))

            if delta_downstream is not None:
                row.update(downstream_collateral_metrics(delta_downstream))
                row["pca1_variance_frac_downstream_feats"] = pca_first_component_fraction(delta_downstream)

            mean_abs_delta = delta_logits.abs().mean(dim=0)
            top_ids = torch.topk(mean_abs_delta, k=10).indices.tolist()
            row["top10_mean_abs_delta_tokens"] = repr([model.to_string(tid) for tid in top_ids])

            label_rows.append(row)

    if (feat_i + 1) % SAVE_EVERY_FEATURES == 0:
        partial_df = pd.DataFrame(label_rows)
        partial_path = OUT_DIR / "robust_phase1_labels_long_PARTIAL.csv"
        partial_df.to_csv(partial_path, index=False)
        print(f"Saved partial after {feat_i+1} features -> {partial_path}")

labels_long_df = pd.DataFrame(label_rows)
elapsed = time.time() - start_time

labels_long_path = OUT_DIR / "robust_phase1_labels_long.csv"
labels_long_df.to_csv(labels_long_path, index=False)

print("Done Phase 1 robust labels.")
print("elapsed seconds:", round(elapsed, 1))
print("labels_long shape:", labels_long_df.shape)
print("saved:", labels_long_path)
display(labels_long_df.head())


################################################################################
# Cell 14
################################################################################
# Cell 13 — make feature-level wide label table

id_cols = ["feature"]
value_cols = [
    c for c in labels_long_df.columns
    if c not in [
        "feature", "clamp_mode", "context_type", "top10_mean_abs_delta_tokens"
    ]
    and labels_long_df[c].dtype != "object"
]

wide_parts = []
for (clamp_mode, context_type), group in labels_long_df.groupby(["clamp_mode", "context_type"]):
    g = group[["feature"] + value_cols].copy()
    suffix = f"__{clamp_mode}__{context_type}"
    rename = {c: c + suffix for c in value_cols}
    g = g.rename(columns=rename)
    wide_parts.append(g)

labels_wide_df = wide_parts[0]
for part in wide_parts[1:]:
    labels_wide_df = labels_wide_df.merge(part, on="feature", how="outer")

labels_wide_path = OUT_DIR / "robust_phase1_labels_wide.csv"
labels_wide_df.to_csv(labels_wide_path, index=False)

print("labels_wide shape:", labels_wide_df.shape)
print("saved:", labels_wide_path)
display(labels_wide_df.head())


################################################################################
# Cell 15
################################################################################
# Cell 14 — Phase 1 label quality diagnostics

primary_quality_cols = [
    "stability_to_mean_abs_cos",
    "stability_to_mean_signed_cos",
    "effect_l2_mean",
    "effect_cv",
    "kl_clean_to_steered_mean",
    "top50_abs_delta_mass_frac",
    "logit_delta_effective_vocab_mean",
    "downstream_feat_l2_mean",
    "downstream_feat_effective_moved_mean",
]

quality_rows = []
for clamp_mode in CLAMP_MODES:
    for context_type in CONTEXT_TYPES:
        sub = labels_long_df[(labels_long_df["clamp_mode"] == clamp_mode) & (labels_long_df["context_type"] == context_type)]
        for col in primary_quality_cols:
            if col in sub.columns:
                vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) > 0:
                    quality_rows.append({
                        "clamp_mode": clamp_mode,
                        "context_type": context_type,
                        "label": col,
                        "mean": vals.mean(),
                        "std": vals.std(),
                        "q10": vals.quantile(0.10),
                        "q90": vals.quantile(0.90),
                        "range_q90_q10": vals.quantile(0.90) - vals.quantile(0.10),
                    })

quality_df = pd.DataFrame(quality_rows)
quality_path = OUT_DIR / "robust_phase1_label_quality.csv"
quality_df.to_csv(quality_path, index=False)

display(quality_df.sort_values("range_q90_q10", ascending=False).head(80))
print("saved:", quality_path)


################################################################################
# Cell 16
################################################################################
# Cell 15 — decoder geometry predictors

TOPK_CROWDING = 20

W_DEC_FLOAT = W_DEC.float()
W_NORM = F.normalize(W_DEC_FLOAT, dim=-1)

geom_rows = []

for feat in tqdm(selected_features, desc="decoder geometry"):
    feat = int(feat)
    v = W_NORM[feat]
    sims = W_NORM @ v
    sims[feat] = 0.0
    abs_sims = sims.abs()
    top_vals = torch.topk(abs_sims, k=min(TOPK_CROWDING, D_SAE - 1)).values

    raw_dec = W_DEC_FLOAT[feat]
    geom_rows.append({
        "feature": feat,
        "decoder_norm": float(torch.norm(raw_dec).item()),
        "crowding_max_abs_cos": float(abs_sims.max().item()),
        "crowding_topk_mean_abs_cos": float(top_vals.mean().item()),
        "crowding_topk_sum_abs_cos": float(top_vals.sum().item()),
    })

geom_df = pd.DataFrame(geom_rows)
display(geom_df.head())


################################################################################
# Cell 17
################################################################################
# Cell 16 — encoder-decoder alignment if available

align_rows = []

W_enc = None
if hasattr(sae, "W_enc"):
    W_enc = sae.W_enc.detach()
elif hasattr(sae, "W_encoders"):
    W_enc = sae.W_encoders.detach()

if W_enc is not None:
    W = W_enc.float().to(device)
    print("W_enc shape:", tuple(W.shape))

    if W.shape[0] == model.cfg.d_model:
        W_ENC_FEATURE = W.T
    elif W.shape[-1] == model.cfg.d_model:
        W_ENC_FEATURE = W
    else:
        W_ENC_FEATURE = None

    if W_ENC_FEATURE is not None:
        for feat in selected_features:
            enc = W_ENC_FEATURE[int(feat)]
            dec = W_DEC_FLOAT[int(feat)]
            align_rows.append({
                "feature": int(feat),
                "encoder_decoder_cos": float(F.cosine_similarity(enc[None, :], dec[None, :], dim=-1).item()),
                "encoder_norm": float(torch.norm(enc).item()),
            })

if len(align_rows) == 0:
    print("No usable encoder matrix found; filling with NaN.")
    align_df = pd.DataFrame({"feature": selected_features, "encoder_decoder_cos": np.nan, "encoder_norm": np.nan})
else:
    align_df = pd.DataFrame(align_rows)

display(align_df.head())


################################################################################
# Cell 18
################################################################################
# Cell 17 — direct-logit footprint predictors
# Gemma note:
# Gemma-2 has a final logit softcap. These direct-logit predictors use pre-softcap d_f @ W_U.
# Use shape metrics like entropy/top-k mass; do NOT compare absolute direct-logit scales to GPT-2.

if not COMPUTE_DIRECT_LOGIT_PREDICTORS:
    print("Skipping direct-logit predictors.")
    direct_df = pd.DataFrame({"feature": selected_features})
else:
    W_U = model.W_U.detach().float().to(device)  # [d_model, vocab]
    VOCAB = W_U.shape[-1]
    DIRECT_TOPK = [10, 50, 100, 500]

    direct_rows = []

    for start in tqdm(range(0, len(selected_features), DIRECT_LOGIT_BATCH_FEATURES), desc="direct logit footprint batches"):
        feats_batch = selected_features[start:start + DIRECT_LOGIT_BATCH_FEATURES]
        D = W_DEC_FLOAT[feats_batch].to(device)  # [batch_feats, d_model]
        direct_logits_batch = D @ W_U  # [batch_feats, vocab]
        abs_batch = direct_logits_batch.abs()
        total_abs_batch = abs_batch.sum(dim=-1).clamp(min=1e-12)
        prob_batch = abs_batch / total_abs_batch[:, None]
        entropy_batch = -(prob_batch.clamp(min=1e-12) * torch.log(prob_batch.clamp(min=1e-12))).sum(dim=-1)
        effective_vocab_batch = torch.exp(entropy_batch)

        for j, feat in enumerate(feats_batch):
            abs_logits = abs_batch[j]
            total_abs = total_abs_batch[j]
            direct_logits = direct_logits_batch[j]

            row = {
                "feature": int(feat),
                "direct_logit_l2": float(torch.norm(direct_logits).item()),
                "direct_logit_linf": float(abs_logits.max().item()),
                "direct_logit_entropy": float(entropy_batch[j].item()),
                "direct_logit_effective_vocab": float(effective_vocab_batch[j].item()),
            }

            for k in DIRECT_TOPK:
                kk = min(k, VOCAB)
                row[f"direct_top{k}_abs_mass_frac"] = float((torch.topk(abs_logits, k=kk).values.sum() / total_abs).item())

            top_ids = torch.topk(abs_logits, k=10).indices.detach().cpu().tolist()
            row["direct_top10_tokens"] = repr([model.to_string(tid) for tid in top_ids])
            direct_rows.append(row)

        del direct_logits_batch, abs_batch, prob_batch
        torch.cuda.empty_cache()

    direct_df = pd.DataFrame(direct_rows)
    display(direct_df.head())


################################################################################
# Cell 19
################################################################################
# Cell 18 — activation/coactivation predictors on real corpus

selected_features_t = torch.tensor(selected_features, device=device, dtype=torch.long)

# First get all-feature activation frequency for coactivation panel.
active_counts_all = torch.zeros(D_SAE, device=device)
total_positions = 0

for i in tqdm(range(0, tokens.shape[0], BATCH_CONTEXTS), desc="all feature activation freq"):
    batch = tokens[i:i+BATCH_CONTEXTS]
    with torch.no_grad():
        _, cache = model.run_with_cache(batch, names_filter=[HOOK_NAME])
        feats = sae_encode(cache[HOOK_NAME], sae)
    flat = feats.reshape(-1, feats.shape[-1])
    active_counts_all += (flat > ACTIVE_THRESH).float().sum(dim=0)
    total_positions += flat.shape[0]

act_freq_all = active_counts_all / total_positions
panel_features = torch.topk(act_freq_all, k=min(2048, D_SAE)).indices.detach()

# Selected feature activation moments and coactivation.
N_SEL = len(selected_features)
sum_x = torch.zeros(N_SEL, device=device)
sum_x2 = torch.zeros(N_SEL, device=device)
sum_x3 = torch.zeros(N_SEL, device=device)
sum_x4 = torch.zeros(N_SEL, device=device)
max_x = torch.zeros(N_SEL, device=device)
active_count_sel = torch.zeros(N_SEL, device=device)

sum_mass = torch.zeros(N_SEL, device=device)
sum_xlogx = torch.zeros(N_SEL, device=device)

coact_counts = torch.zeros((N_SEL, len(panel_features)), device=device)
total_positions = 0

for i in tqdm(range(0, tokens.shape[0], BATCH_CONTEXTS), desc="selected activation/coactivation"):
    batch = tokens[i:i+BATCH_CONTEXTS]
    with torch.no_grad():
        _, cache = model.run_with_cache(batch, names_filter=[HOOK_NAME])
        feats = sae_encode(cache[HOOK_NAME], sae)

    flat = feats.reshape(-1, feats.shape[-1]).float()
    sel = flat[:, selected_features_t]
    panel = flat[:, panel_features]

    sel_pos = sel.clamp(min=0)
    sel_active = sel > ACTIVE_THRESH
    panel_active = panel > ACTIVE_THRESH

    sum_x += sel.sum(dim=0)
    sum_x2 += (sel ** 2).sum(dim=0)
    sum_x3 += (sel ** 3).sum(dim=0)
    sum_x4 += (sel ** 4).sum(dim=0)
    max_x = torch.maximum(max_x, sel.max(dim=0).values)
    active_count_sel += sel_active.float().sum(dim=0)

    sum_mass += sel_pos.sum(dim=0)
    sum_xlogx += (sel_pos.clamp(min=1e-12) * torch.log(sel_pos.clamp(min=1e-12))).sum(dim=0)

    coact_counts += sel_active.float().T @ panel_active.float()

    total_positions += sel.shape[0]

mean_x = sum_x / total_positions
mean_x2 = sum_x2 / total_positions
var_x = (mean_x2 - mean_x ** 2).clamp(min=1e-12)
std_x = torch.sqrt(var_x)

mean_x3 = sum_x3 / total_positions
mean_x4 = sum_x4 / total_positions
central4 = mean_x4 - 4 * mean_x * mean_x3 + 6 * (mean_x ** 2) * mean_x2 - 3 * (mean_x ** 4)
kurtosis = central4 / (var_x ** 2).clamp(min=1e-12)

token_act_freq = active_count_sel / total_positions
p = token_act_freq.clamp(1e-8, 1 - 1e-8)
binary_entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

activation_entropy = torch.log(sum_mass.clamp(min=1e-12)) - (sum_xlogx / sum_mass.clamp(min=1e-12))
activation_entropy_norm = activation_entropy / math.log(total_positions)

coact_entropy_vals = []
coact_count_mean_vals = []

for r in range(N_SEL):
    counts = coact_counts[r]
    total = counts.sum()
    if total.item() <= 0:
        coact_entropy_vals.append(0.0)
        coact_count_mean_vals.append(0.0)
    else:
        q = (counts / total).clamp(min=1e-12)
        ent = -(q * torch.log(q)).sum() / math.log(len(panel_features))
        coact_entropy_vals.append(float(ent.item()))
        coact_count_mean_vals.append(float((total / active_count_sel[r].clamp(min=1)).item()))

act_rows = []
for j, feat in enumerate(selected_features):
    act_rows.append({
        "feature": int(feat),
        "token_act_freq": float(token_act_freq[j].item()),
        "token_act_mean": float(mean_x[j].item()),
        "token_act_std": float(std_x[j].item()),
        "token_act_max": float(max_x[j].item()),
        "token_binary_entropy": float(binary_entropy[j].item()),
        "token_activation_entropy_norm": float(activation_entropy_norm[j].item()),
        "token_act_kurtosis": float(kurtosis[j].item()),
        "coact_entropy_norm": coact_entropy_vals[j],
        "coact_count_mean": coact_count_mean_vals[j],
    })

activation_df = pd.DataFrame(act_rows)
display(activation_df.head())


################################################################################
# Cell 20
################################################################################
# Cell 19 — combine predictors and merge labels

predictors_df = (
    pd.DataFrame({"feature": selected_features})
    .merge(selected_feature_stats.add_prefix("phase1_").rename(columns={"phase1_feature": "feature"}), on="feature", how="left")
    .merge(geom_df, on="feature", how="left")
    .merge(align_df, on="feature", how="left")
    .merge(direct_df, on="feature", how="left")
    .merge(activation_df, on="feature", how="left")
)

merged_df = labels_wide_df.merge(predictors_df, on="feature", how="left")

predictors_path = OUT_DIR / "robust_phase2_predictors.csv"
merged_path = OUT_DIR / "robust_phase2_phase3_merged.csv"

predictors_df.to_csv(predictors_path, index=False)
merged_df.to_csv(merged_path, index=False)

print("predictors:", predictors_df.shape, predictors_path)
print("merged:", merged_df.shape, merged_path)
display(predictors_df.head())
display(merged_df.head())


################################################################################
# Cell 21
################################################################################
# Cell 20 — define predictor sets and target sets

candidate_predictor_cols = [
    c for c in predictors_df.columns
    if c != "feature" and predictors_df[c].dtype != "object"
]

freq_predictors = [c for c in candidate_predictor_cols if "freq" in c]
activation_magnitude_predictors = [
    c for c in candidate_predictor_cols
    if any(s in c for s in ["act_mean", "act_std", "act_max", "act_q95"])
]
activation_shape_predictors = [
    c for c in candidate_predictor_cols
    if any(s in c for s in ["entropy", "kurtosis"]) and "direct_logit" not in c
]
crowding_predictors = [c for c in candidate_predictor_cols if "crowding" in c]
geometry_predictors = [
    c for c in candidate_predictor_cols
    if any(s in c for s in ["crowding", "decoder_norm", "encoder_decoder", "encoder_norm"])
]
direct_logit_predictors = [c for c in candidate_predictor_cols if "direct_" in c]
coactivation_predictors = [c for c in candidate_predictor_cols if "coact" in c]

no_magnitude_predictors = [
    c for c in candidate_predictor_cols
    if c not in activation_magnitude_predictors
]

compact_no_magnitude_predictors = [
    c for c in [
        "phase1_final_act_freq",
        "token_act_freq",
        "token_binary_entropy",
        "token_activation_entropy_norm",
        "token_act_kurtosis",
        "coact_entropy_norm",
        "coact_count_mean",
        "decoder_norm",
        "crowding_max_abs_cos",
        "crowding_topk_mean_abs_cos",
        "crowding_topk_sum_abs_cos",
        "direct_logit_l2",
        "direct_logit_linf",
        "direct_logit_effective_vocab",
        "direct_top50_abs_mass_frac",
        "encoder_decoder_cos",
    ] if c in candidate_predictor_cols
]

predictor_sets = {
    "freq_only": freq_predictors,
    "activation_magnitude_only": activation_magnitude_predictors,
    "activation_shape_only": activation_shape_predictors,
    "crowding_only": crowding_predictors,
    "geometry_only": geometry_predictors,
    "direct_logit_only": direct_logit_predictors,
    "coactivation_only": coactivation_predictors,
    "compact_no_magnitude": compact_no_magnitude_predictors,
    "full_no_magnitude": no_magnitude_predictors,
    "full_all": candidate_predictor_cols,
}
predictor_sets = {k: v for k, v in predictor_sets.items() if len(v) > 0}

# Primary targets are deliberately fixed_global_add + mixed to avoid top-context and sign-flip clamp confounds.
primary_targets = [
    "stability_to_mean_abs_cos__fixed_global_add__mixed",
    "stability_to_mean_signed_cos__fixed_global_add__mixed",
    "pairwise_cos_abs_mean__fixed_global_add__mixed",
    "pairwise_cos_signed_mean__fixed_global_add__mixed",
    "effect_cv__fixed_global_add__mixed",
    "downstream_feat_l2_mean__fixed_global_add__mixed",
    "downstream_feat_effective_moved_mean__fixed_global_add__mixed",
    "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
]

secondary_targets = [
    c for c in merged_df.columns
    if (
        c.startswith("stability_to_mean")
        or c.startswith("pairwise_cos")
        or c.startswith("effect_l2_mean")
        or c.startswith("effect_cv")
        or c.startswith("downstream_feat")
        or c.startswith("kl_clean")
    )
]

primary_targets = [c for c in primary_targets if c in merged_df.columns]
secondary_targets = [c for c in secondary_targets if c in merged_df.columns]

print("predictor sets:")
for k, v in predictor_sets.items():
    print(k, len(v))
print("\nprimary targets:")
for t in primary_targets:
    print(t)
print("\nsecondary target count:", len(secondary_targets))


################################################################################
# Cell 22
################################################################################
# Cell 21 — evaluation helpers

def clean_xy(df, predictors, target):
    cols = predictors + [target]
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub[predictors].astype(float).values
    y = sub[target].astype(float).values
    return X, y, sub

def cv_regression_eval(df, predictors, target, model_kind="ridge", n_splits=5, seed=0):
    X, y, sub = clean_xy(df, predictors, target)
    if len(y) < n_splits * 5 or np.std(y) < 1e-12:
        return None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    if model_kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif model_kind == "gbr":
        model = GradientBoostingRegressor(random_state=seed, max_depth=2, n_estimators=100, learning_rate=0.05)
    else:
        raise ValueError(model_kind)

    y_pred = cross_val_predict(model, X, y, cv=kf)
    return {
        "target": target,
        "model_kind": model_kind,
        "n": len(y),
        "n_predictors": len(predictors),
        "cv_r2": r2_score(y, y_pred),
        "cv_mae": mean_absolute_error(y, y_pred),
        "cv_spearman": spearmanr(y, y_pred).statistic,
    }

def residualize_target(df, target, controls):
    # Returns a series of target residualized against controls using ridge on all data.
    # This is for diagnostic robustness, not final held-out claim.
    cols = [target] + controls
    sub = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X = sub[controls].astype(float).values
    y = sub[target].astype(float).values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    resid = y - model.predict(X)
    out = pd.Series(index=sub.index, data=resid, name=target + "__residualized")
    return out


################################################################################
# Cell 23
################################################################################
# Cell 22 — univariate correlations

corr_rows = []
all_targets_for_corr = primary_targets + [t for t in secondary_targets if t not in primary_targets]

for pred in candidate_predictor_cols:
    for target in all_targets_for_corr:
        sub = merged_df[[pred, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 20 or sub[pred].std() < 1e-12 or sub[target].std() < 1e-12:
            continue
        sp_r, sp_p = spearmanr(sub[pred], sub[target])
        pe_r, pe_p = pearsonr(sub[pred], sub[target])
        corr_rows.append({
            "predictor": pred,
            "target": target,
            "target_is_primary": target in primary_targets,
            "spearman_r": sp_r,
            "spearman_p": sp_p,
            "pearson_r": pe_r,
            "pearson_p": pe_p,
            "abs_spearman": abs(sp_r),
            "n": len(sub),
        })

corr_df = pd.DataFrame(corr_rows).sort_values("abs_spearman", ascending=False).reset_index(drop=True)
corr_path = OUT_DIR / "robust_phase3_univariate_correlations.csv"
corr_df.to_csv(corr_path, index=False)

print("Top primary-target correlations:")
display(corr_df[corr_df["target_is_primary"]].head(50))

print("\nTop overall correlations:")
display(corr_df.head(50))

print("saved:", corr_path)


################################################################################
# Cell 24
################################################################################
# Cell 23 — held-out regression evaluation

eval_rows = []

# Primary targets: ridge + gbr.
# Secondary targets: ridge only to avoid fishing too hard.
for target in primary_targets:
    for set_name, preds in predictor_sets.items():
        for model_kind in ["ridge", "gbr"]:
            out = cv_regression_eval(merged_df, preds, target, model_kind=model_kind, n_splits=5, seed=0)
            if out is not None:
                out["predictor_set"] = set_name
                out["target_is_primary"] = True
                eval_rows.append(out)

for target in secondary_targets:
    if target in primary_targets:
        continue
    for set_name, preds in predictor_sets.items():
        out = cv_regression_eval(merged_df, preds, target, model_kind="ridge", n_splits=5, seed=0)
        if out is not None:
            out["predictor_set"] = set_name
            out["target_is_primary"] = False
            eval_rows.append(out)

eval_df = pd.DataFrame(eval_rows).sort_values(["target_is_primary", "target", "cv_spearman"], ascending=[False, True, False])
eval_path = OUT_DIR / "robust_phase3_cv_regression_results.csv"
eval_df.to_csv(eval_path, index=False)

print("Primary target CV results:")
display(eval_df[eval_df["target_is_primary"]])

print("\nTop secondary target CV results:")
display(eval_df[~eval_df["target_is_primary"]].sort_values("cv_spearman", ascending=False).head(50))

print("saved:", eval_path)


################################################################################
# Cell 25
################################################################################
# Cell 24 — baseline comparisons for primary targets

compare_rows = []

for target in primary_targets:
    sub = eval_df[(eval_df["target"] == target) & (eval_df["model_kind"] == "ridge")]

    def get(set_name, metric):
        row = sub[sub["predictor_set"] == set_name]
        return float(row.iloc[0][metric]) if len(row) else np.nan

    compare_rows.append({
        "target": target,
        "freq_cv_spearman": get("freq_only", "cv_spearman"),
        "activation_magnitude_cv_spearman": get("activation_magnitude_only", "cv_spearman"),
        "crowding_cv_spearman": get("crowding_only", "cv_spearman"),
        "geometry_cv_spearman": get("geometry_only", "cv_spearman"),
        "coactivation_cv_spearman": get("coactivation_only", "cv_spearman"),
        "direct_logit_cv_spearman": get("direct_logit_only", "cv_spearman"),
        "compact_no_magnitude_cv_spearman": get("compact_no_magnitude", "cv_spearman"),
        "full_no_magnitude_cv_spearman": get("full_no_magnitude", "cv_spearman"),
        "full_all_cv_spearman": get("full_all", "cv_spearman"),
        "full_no_magnitude_minus_freq": get("full_no_magnitude", "cv_spearman") - get("freq_only", "cv_spearman"),
        "full_no_magnitude_minus_activation_magnitude": get("full_no_magnitude", "cv_spearman") - get("activation_magnitude_only", "cv_spearman"),
        "full_all_minus_activation_magnitude": get("full_all", "cv_spearman") - get("activation_magnitude_only", "cv_spearman"),
    })

baseline_compare_df = pd.DataFrame(compare_rows)
baseline_path = OUT_DIR / "robust_phase3_baseline_comparison.csv"
baseline_compare_df.to_csv(baseline_path, index=False)

display(baseline_compare_df)
print("saved:", baseline_path)


################################################################################
# Cell 26
################################################################################
# Cell 25 — residualized-target robustness
# Diagnostic: remove effect-size and activation-magnitude effects, then test prediction of residual stability.

resid_rows = []

controls = [
    c for c in [
        "effect_l2_mean__fixed_global_add__mixed",
        "effect_cv__fixed_global_add__mixed",
        "clamp_value__fixed_global_add__mixed",
        "natural_final_act_mean_contexts__fixed_global_add__mixed",
        "natural_final_act_max_contexts__fixed_global_add__mixed",
    ] if c in merged_df.columns
]

resid_targets = [
    t for t in [
        "stability_to_mean_abs_cos__fixed_global_add__mixed",
        "stability_to_mean_signed_cos__fixed_global_add__mixed",
        "pairwise_cos_abs_mean__fixed_global_add__mixed",
        "pairwise_cos_signed_mean__fixed_global_add__mixed",
    ] if t in merged_df.columns
]

print("Residualization controls:", controls)
print("Residualization targets:", resid_targets)

resid_df = merged_df.copy()

for target in resid_targets:
    if len(controls) == 0:
        continue

    resid_series = residualize_target(resid_df, target, controls)
    resid_col = target + "__resid_effect_activation"
    resid_df[resid_col] = np.nan
    resid_df.loc[resid_series.index, resid_col] = resid_series.values

    for set_name in ["geometry_only", "coactivation_only", "compact_no_magnitude", "full_no_magnitude", "full_all"]:
        if set_name not in predictor_sets:
            continue
        out = cv_regression_eval(resid_df, predictor_sets[set_name], resid_col, model_kind="ridge", n_splits=5, seed=0)
        if out is not None:
            out["original_target"] = target
            out["residual_target"] = resid_col
            out["predictor_set"] = set_name
            resid_rows.append(out)

resid_eval_df = pd.DataFrame(resid_rows)
resid_path = OUT_DIR / "robust_phase3_residualized_target_results.csv"
resid_eval_df.to_csv(resid_path, index=False)

display(resid_eval_df.sort_values("cv_spearman", ascending=False) if len(resid_eval_df) else resid_eval_df)
print("saved:", resid_path)


################################################################################
# Cell 27
################################################################################
# Cell 26 — plots for strongest primary relationships

plots_dir = OUT_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

top_primary_pairs = corr_df[corr_df["target_is_primary"]].head(8)[["predictor", "target"]].values.tolist()

for pred, target in top_primary_pairs:
    sub = merged_df[[pred, target]].replace([np.inf, -np.inf], np.nan).dropna()
    plt.figure(figsize=(6, 4))
    plt.scatter(sub[pred], sub[target], alpha=0.7)
    plt.xlabel(pred)
    plt.ylabel(target)
    sp = spearmanr(sub[pred], sub[target]).statistic
    plt.title(f"{pred} vs {target}\nSpearman={sp:.3f}")
    fname = f"{pred[:35]}__vs__{target[:45]}.png".replace("/", "_")
    plt.tight_layout()
    plt.savefig(plots_dir / fname, dpi=150)
    plt.show()

print("saved plots to:", plots_dir)


################################################################################
# Cell 28
################################################################################
# Cell 27 — robust rerun verdict

primary_corr = corr_df[corr_df["target_is_primary"]].copy()
best_primary_corr = primary_corr.iloc[0] if len(primary_corr) else None

if len(baseline_compare_df):
    best_full_no_mag_minus_freq = baseline_compare_df["full_no_magnitude_minus_freq"].max()
    best_full_no_mag_minus_actmag = baseline_compare_df["full_no_magnitude_minus_activation_magnitude"].max()
    mean_full_no_mag_minus_freq = baseline_compare_df["full_no_magnitude_minus_freq"].mean()
else:
    best_full_no_mag_minus_freq = np.nan
    best_full_no_mag_minus_actmag = np.nan
    mean_full_no_mag_minus_freq = np.nan

summary = {
    "n_features": len(selected_features),
    "n_contexts": int(tokens.shape[0]),
    "corpus_distinct_texts": len(texts),
    "fixed_global_clamp_value": FIXED_GLOBAL_CLAMP_VALUE,
    "primary_clamp_mode": "fixed_global_add",
    "best_primary_corr": None if best_primary_corr is None else {
        "predictor": best_primary_corr["predictor"],
        "target": best_primary_corr["target"],
        "spearman_r": float(best_primary_corr["spearman_r"]),
        "spearman_p": float(best_primary_corr["spearman_p"]),
    },
    "best_full_no_magnitude_minus_freq": None if np.isnan(best_full_no_mag_minus_freq) else float(best_full_no_mag_minus_freq),
    "best_full_no_magnitude_minus_activation_magnitude": None if np.isnan(best_full_no_mag_minus_actmag) else float(best_full_no_mag_minus_actmag),
    "mean_full_no_magnitude_minus_freq": None if np.isnan(mean_full_no_mag_minus_freq) else float(mean_full_no_mag_minus_freq),
    "config": CONFIG,
}

summary_path = OUT_DIR / "robust_phase123_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print("=" * 80)
print("GEMMA FULL-SCALE PHASE 1/2/3 VERDICT")
print("=" * 80)
print(f"n features: {len(selected_features)}")
print(f"n contexts: {tokens.shape[0]}")
print(f"distinct corpus texts: {len(texts)}")
print(f"fixed global/add clamp value: {FIXED_GLOBAL_CLAMP_VALUE:.4f}")
print("primary clamp mode: fixed_global_add")
print("-" * 80)

if best_primary_corr is not None:
    print("Best primary-target univariate relationship:")
    print(best_primary_corr[["predictor", "target", "spearman_r", "spearman_p"]])
else:
    print("No primary-target correlation found.")

print("-" * 80)
print(f"Best full_no_magnitude - freq CV Spearman improvement: {best_full_no_mag_minus_freq:.3f}")
print(f"Best full_no_magnitude - activation_magnitude CV Spearman improvement: {best_full_no_mag_minus_actmag:.3f}")
print(f"Mean full_no_magnitude - freq improvement across primary targets: {mean_full_no_mag_minus_freq:.3f}")
print("-" * 80)

if (
    best_primary_corr is not None
    and abs(best_primary_corr["spearman_r"]) >= 0.25
    and best_full_no_mag_minus_freq >= 0.05
    and best_full_no_mag_minus_actmag >= 0.00
):
    print("VERDICT: GEMMA FULL-SCALE SIGNAL SURVIVED.")
    print("Next: Phase 4 screening demo.")
elif (
    best_primary_corr is not None
    and abs(best_primary_corr["spearman_r"]) >= 0.20
    and best_full_no_mag_minus_freq >= 0.00
):
    print("VERDICT: GEMMA FULL-SCALE PROMISING BUT BORDERLINE.")
    print("Next: inspect targets/predictors, maybe increase contexts/features before Gemma.")
else:
    print("VERDICT: WEAK GEMMA FULL-SCALE RESULT AFTER ROBUST CONTROLS.")
    print("Next: do not move to Gemma/Phase 4 yet. Revisit labels or steering design.")

print("saved summary:", summary_path)


################################################################################
# Cell 29
################################################################################
# Cell 28 — save outputs to Google Drive

SAVE_TO_DRIVE = True
DRIVE_OUT = "/content/drive/MyDrive/SAE Prediction/gemma_fullscale_phase123_outputs"

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

    drive_out = Path(DRIVE_OUT)
    drive_out.mkdir(parents=True, exist_ok=True)

    for path in OUT_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, drive_out / path.name)
            print("copied:", path.name)

    # Copy plots
    if (OUT_DIR / "plots").exists():
        plot_out = drive_out / "plots"
        plot_out.mkdir(exist_ok=True)
        for path in (OUT_DIR / "plots").glob("*.png"):
            shutil.copy2(path, plot_out / path.name)

    print("Saved to:", drive_out)
else:
    print("SAVE_TO_DRIVE is False.")
