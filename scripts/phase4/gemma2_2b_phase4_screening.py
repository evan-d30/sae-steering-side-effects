#!/usr/bin/env python
# Converted from notebooks/phase4/gemma2_2b_phase4_screening.ipynb
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
import shutil
from pathlib import Path
from getpass import getpass

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from datasets import load_dataset

from transformer_lens import HookedTransformer
from sae_lens import SAE

torch.set_grad_enabled(False)

pd.set_option("display.max_rows", 240)
pd.set_option("display.max_columns", 240)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

OUT_DIR = Path("/content/gemma_phase4_both_predictors_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR:", OUT_DIR)


################################################################################
# Cell 3
################################################################################
# Cell 2.5 — optional Hugging Face token login
# Gemma may require a Hugging Face token and accepted model access.
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
# Cell 3 — locate full-scale Gemma Phase 1/2/3 outputs

# Expected Drive folder from the full-scale Gemma notebook:
USE_GOOGLE_DRIVE = True
DRIVE_FOLDER = "/content/drive/MyDrive/SAE Prediction/gemma_fullscale_phase123_outputs"

# If you uploaded files directly to Colab, set USE_GOOGLE_DRIVE=False and upload to /content.
UPLOAD_FOLDER = "/content"

if USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path(DRIVE_FOLDER)
else:
    BASE_DIR = Path(UPLOAD_FOLDER)

MERGED_PATH = BASE_DIR / "robust_phase2_phase3_merged.csv"
SUMMARY_PATH = BASE_DIR / "robust_phase123_summary.json"

print("BASE_DIR:", BASE_DIR)
print("merged exists :", MERGED_PATH.exists(), MERGED_PATH)
print("summary exists:", SUMMARY_PATH.exists(), SUMMARY_PATH)

assert MERGED_PATH.exists(), f"Missing merged dataset: {MERGED_PATH}"
assert SUMMARY_PATH.exists(), f"Missing summary file: {SUMMARY_PATH}"


################################################################################
# Cell 5
################################################################################
# Cell 4 — load merged labels/predictors and metadata

merged_df = pd.read_csv(MERGED_PATH)

with open(SUMMARY_PATH, "r") as f:
    summary = json.load(f)

config = summary["config"]

print("merged shape:", merged_df.shape)
print("summary:")
print(json.dumps(summary, indent=2)[:3000])

assert "gemma" in config.get("MODEL_NAME", "").lower(), "This notebook is for Gemma outputs."

display(merged_df.head())


################################################################################
# Cell 6
################################################################################
# Cell 5 — define targets, clean score, and SAFE predictor sets

STABILITY_COL = "stability_to_mean_abs_cos__fixed_global_add__mixed"
SIGNED_STABILITY_COL = "stability_to_mean_signed_cos__fixed_global_add__mixed"
COLLATERAL_COL = "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed"
EFFECT_COL = "effect_l2_mean__fixed_global_add__mixed"

required_cols = ["feature", STABILITY_COL, COLLATERAL_COL, EFFECT_COL]
missing = [c for c in required_cols if c not in merged_df.columns]
assert not missing, f"Missing required columns: {missing}"

def zscore_np(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)

work_df = merged_df.copy()
work_df["phase4_collateral_per_effect"] = (
    np.log1p(work_df[COLLATERAL_COL]) / (work_df[EFFECT_COL].abs() + 1e-8)
)
work_df["phase4_clean_score"] = (
    zscore_np(work_df[STABILITY_COL])
    - zscore_np(work_df["phase4_collateral_per_effect"])
)

# Geometry-only predictor set.
# Drop crowding_topk_sum_abs_cos because it is redundant with mean when K is fixed.
geometry_predictors = [
    c for c in [
        "decoder_norm",
        "encoder_decoder_cos",
        "encoder_norm",
        "crowding_max_abs_cos",
        "crowding_topk_mean_abs_cos",
    ]
    if c in work_df.columns
]

# Safer full_no_magnitude set:
# Only include genuine pre-intervention predictors.
# Exclude label-wide columns containing "__" because those are Phase 1 outcome/metadata columns.
allowed_full_no_magnitude = [
    # frequency / shape, not magnitude
    "phase1_final_act_freq",
    "token_act_freq",
    "token_binary_entropy",
    "token_activation_entropy_norm",
    "token_act_kurtosis",

    # coactivation
    "coact_entropy_norm",
    "coact_count_mean",

    # geometry
    "decoder_norm",
    "encoder_decoder_cos",
    "encoder_norm",
    "crowding_max_abs_cos",
    "crowding_topk_mean_abs_cos",

    # direct-logit shape / footprint, if computed
    "direct_logit_l2",
    "direct_logit_linf",
    "direct_logit_entropy",
    "direct_logit_effective_vocab",
    "direct_top10_abs_mass_frac",
    "direct_top50_abs_mass_frac",
    "direct_top100_abs_mass_frac",
    "direct_top500_abs_mass_frac",
]

full_no_magnitude_predictors = [
    c for c in allowed_full_no_magnitude
    if c in work_df.columns
]

def clean_predictor_list(df, cols, min_nonnull_frac=0.80):
    """Keep numeric predictors with enough finite values and nonzero variance."""
    good = []
    dropped = []

    for c in cols:
        if c not in df.columns:
            dropped.append((c, "missing"))
            continue
        if df[c].dtype == "object":
            dropped.append((c, "object"))
            continue

        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnull_frac = vals.notna().mean()

        if nonnull_frac < min_nonnull_frac:
            dropped.append((c, f"too_many_nan:{nonnull_frac:.2f}"))
            continue
        if vals.dropna().std() < 1e-12:
            dropped.append((c, "zero_variance"))
            continue

        good.append(c)

    return good, dropped

geometry_predictors, dropped_geo = clean_predictor_list(work_df, geometry_predictors)
full_no_magnitude_predictors, dropped_full = clean_predictor_list(work_df, full_no_magnitude_predictors)

PREDICTOR_SETS = {
    "geometry_only": geometry_predictors,
    "full_no_magnitude": full_no_magnitude_predictors,
}

for name, preds in PREDICTOR_SETS.items():
    print("\nPredictor set:", name)
    print("n predictors:", len(preds))
    print(preds)
    assert len(preds) > 0, f"No usable predictors found for {name}"

print("\nDropped geometry predictors:")
print(dropped_geo)

print("\nDropped full_no_magnitude predictors:")
print(dropped_full)

# Usable feature table across all predictors.
needed = ["feature", "phase4_clean_score", EFFECT_COL] + sorted(set(sum(PREDICTOR_SETS.values(), [])))
phase4_df = work_df[needed].replace([np.inf, -np.inf], np.nan).copy()
phase4_df["feature"] = phase4_df["feature"].astype(int)

print("\nusable feature rows before per-set filtering:", len(phase4_df))
display(phase4_df.head())


################################################################################
# Cell 7
################################################################################
# Cell 6 — load Gemma model and Gemma Scope SAEs

MODEL_NAME = config.get("MODEL_NAME", "gemma-2-2b")
SAE_RELEASE = config.get("SAE_RELEASE", "gemma-scope-2b-pt-res-canonical")
SAE_ID = config.get("SAE_ID", "layer_12/width_16k/canonical")
USE_DOWNSTREAM_SAE = config.get("USE_DOWNSTREAM_SAE", True)
DOWNSTREAM_SAE_ID = config.get("DOWNSTREAM_SAE_ID", "layer_16/width_16k/canonical")

MODEL_DTYPE_STR = config.get("MODEL_DTYPE", "torch.bfloat16")
if "bfloat16" in MODEL_DTYPE_STR and torch.cuda.is_available():
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32

model_kwargs = {"dtype": MODEL_DTYPE}

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

def infer_hook_name(sae_id, model, cfg=None):
    import re
    hook_names = set(model.hook_dict.keys())

    # Config names first.
    candidates = []
    if isinstance(cfg, dict):
        for key in ["hook_name", "hook_point", "act_name"]:
            if key in cfg:
                candidates.append(cfg[key])

    try:
        candidates.append(sae.cfg.hook_name)
    except Exception:
        pass

    for cand in candidates:
        if isinstance(cand, str) and cand in hook_names:
            return cand

    # Gemma Scope mapping.
    m = re.search(r"layer_(\d+)", str(sae_id))
    if m is not None:
        layer = int(m.group(1))
        for cand in [
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.hook_resid_pre",
            f"blocks.{layer}.hook_resid_mid",
        ]:
            if cand in hook_names:
                return cand

    if sae_id in hook_names:
        return sae_id

    raise KeyError(f"Could not infer hook for {sae_id}. First hooks: {list(hook_names)[:20]}")

HOOK_NAME = infer_hook_name(SAE_ID, model, cfg=cfg_dict)

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
        DOWNSTREAM_HOOK_NAME = infer_hook_name(DOWNSTREAM_SAE_ID, model, cfg=downstream_cfg_dict)
    except Exception as e:
        print("Could not load downstream SAE. Continuing without downstream-feature metrics.")
        print("Error:", repr(e))
        downstream_sae = None
        DOWNSTREAM_HOOK_NAME = None

print("Model:", MODEL_NAME)
print("Primary hook:", HOOK_NAME)
print("Downstream hook:", DOWNSTREAM_HOOK_NAME)
print("W_DEC shape:", tuple(W_DEC.shape))
print("D_SAE:", D_SAE)
print("primary hook exists:", HOOK_NAME in model.hook_dict)
if DOWNSTREAM_HOOK_NAME is not None:
    print("downstream hook exists:", DOWNSTREAM_HOOK_NAME in model.hook_dict)


################################################################################
# Cell 8
################################################################################
# Cell 7 — build fresh held-out Gemma evaluation contexts

# Use validation split so Phase 4 contexts are fresh relative to train[:...] Phase 1/2/3 runs.
HF_DATASET_NAME = "Salesforce/wikitext"
HF_DATASET_CONFIG = "wikitext-103-raw-v1"
HF_DATASET_SPLIT = "validation"

CONTEXT_LEN = int(config.get("CONTEXT_LEN", 48))

# Start moderate. Increase to 512 after checking runtime if desired.
N_HELDOUT_CONTEXTS = 768
N_EVAL_CONTEXTS = 256
MIN_DISTINCT_TEXTS = 500

BATCH_CONTEXTS = 2
BATCH_STEER = 2

print("Loading held-out dataset...")
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

print("distinct heldout texts:", len(texts))
if len(texts) < MIN_DISTINCT_TEXTS:
    raise ValueError(f"Too few distinct held-out texts: {len(texts)}")

random.seed(123)
random.shuffle(texts)

all_ids = []
for txt in tqdm(texts, desc="tokenizing heldout"):
    ids = model.tokenizer.encode(txt, add_special_tokens=False)
    if ids:
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
    if len(chunks) >= N_HELDOUT_CONTEXTS:
        break

assert len(chunks) >= N_EVAL_CONTEXTS, f"Only made {len(chunks)} contexts."

tokens = torch.tensor(chunks, dtype=torch.long, device=device)

rng = np.random.default_rng(123)
eval_idx = rng.choice(np.arange(tokens.shape[0]), size=N_EVAL_CONTEXTS, replace=False)
eval_idx_t = torch.tensor(eval_idx, dtype=torch.long, device=device)
eval_tokens = tokens[eval_idx_t]

print("heldout tokens:", tuple(tokens.shape))
print("eval tokens:", tuple(eval_tokens.shape))
print(model.tokenizer.decode(eval_tokens[0].detach().cpu().tolist()))


################################################################################
# Cell 9
################################################################################
# Cell 8 — clean pass on held-out evaluation contexts

clean_final_logits_chunks = []
downstream_final_feat_chunks = []

names_filter = []
if DOWNSTREAM_HOOK_NAME is not None:
    names_filter.append(DOWNSTREAM_HOOK_NAME)

for i in tqdm(range(0, eval_tokens.shape[0], BATCH_CONTEXTS), desc="heldout clean pass"):
    batch = eval_tokens[i:i+BATCH_CONTEXTS]
    with torch.no_grad():
        if len(names_filter) > 0:
            logits, cache = model.run_with_cache(batch, names_filter=names_filter)
        else:
            logits = model(batch)
            cache = {}

    clean_final_logits_chunks.append(logits[:, -1, :].detach().cpu())

    if downstream_sae is not None and DOWNSTREAM_HOOK_NAME in cache:
        downstream_feat = sae_encode(cache[DOWNSTREAM_HOOK_NAME], downstream_sae)[:, -1, :]
        downstream_final_feat_chunks.append(downstream_feat.detach().cpu())

clean_final_logits = torch.cat(clean_final_logits_chunks, dim=0)

if len(downstream_final_feat_chunks) > 0:
    feat_downstream_clean = torch.cat(downstream_final_feat_chunks, dim=0)
else:
    feat_downstream_clean = None

print("clean_final_logits:", tuple(clean_final_logits.shape))
print("feat_downstream_clean:", None if feat_downstream_clean is None else tuple(feat_downstream_clean.shape))


################################################################################
# Cell 10
################################################################################
# Cell 9 — downstream panel and steering/metric helpers

ACTIVE_THRESH = 1e-6
DOWNSTREAM_PANEL_SIZE = 1024
DOWNSTREAM_ABS_THRESHOLDS = [0.01, 0.05, 0.10]
ABS_LOGIT_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]

downstream_panel = None
if feat_downstream_clean is not None:
    downstream_active_freq = (feat_downstream_clean > ACTIVE_THRESH).float().mean(dim=0)
    downstream_panel = torch.topk(
        downstream_active_freq,
        k=min(DOWNSTREAM_PANEL_SIZE, feat_downstream_clean.shape[-1])
    ).indices.detach().cpu()
    print("downstream panel size:", len(downstream_panel))

FIXED_GLOBAL_CLAMP_VALUE = float(summary.get("fixed_global_clamp_value", 1.0))
print("Fixed global add value:", FIXED_GLOBAL_CLAMP_VALUE)

def run_feature_add(feature_idx, batch_tokens, add_value=FIXED_GLOBAL_CLAMP_VALUE, return_downstream=True):
    feature_idx = int(feature_idx)
    direction = W_DEC[feature_idx].detach()

    def steering_hook(act, hook):
        delta = torch.full((act.shape[0],), float(add_value), device=act.device, dtype=act.dtype)
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

def logit_metrics(delta_logits, clean_logits, steered_logits):
    delta = delta_logits.float()
    clean = clean_logits.float()
    steered = steered_logits.float()
    abs_delta = delta.abs()

    effect_l2 = torch.norm(delta, dim=-1)
    out = {
        "effect_l2_mean": float(effect_l2.mean().item()),
        "effect_l2_std": float(effect_l2.std().item()),
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

    return out

def downstream_metrics(delta_feat):
    if delta_feat is None:
        return {}

    delta = delta_feat.float()
    abs_delta = delta.abs()
    l2 = torch.norm(delta, dim=-1)
    total_abs = abs_delta.sum(dim=-1).clamp(min=1e-8)
    prob = abs_delta / total_abs[:, None]
    entropy = -(prob.clamp(min=1e-12) * torch.log(prob.clamp(min=1e-12))).sum(dim=-1)

    out = {
        "downstream_feat_l2_mean": float(l2.mean().item()),
        "downstream_feat_effective_moved_mean": float(torch.exp(entropy).mean().item()),
    }

    for thr in DOWNSTREAM_ABS_THRESHOLDS:
        out[f"downstream_feat_count_abs_delta_gt_{thr}"] = float((abs_delta > thr).float().sum(dim=-1).mean().item())

    return out


################################################################################
# Cell 11
################################################################################
# Cell 10 — selection function for a predictor set, robust to missing/NaN columns

TEST_SIZE = 0.40
SELECTION_SEED = 0
N_PER_GROUP = 25

USE_EFFECT_MATCHING = True
EFFECT_MATCH_Q_LOW = 0.05
EFFECT_MATCH_Q_HIGH = 0.95

def make_selection_for_predictor_set(predictor_set_name, predictor_cols):
    # Re-clean predictor cols inside this function so one bad column cannot kill the whole set.
    predictor_cols, dropped = clean_predictor_list(work_df, predictor_cols, min_nonnull_frac=0.80)

    print(f"\n{predictor_set_name}: usable predictors after cleaning = {len(predictor_cols)}")
    print(predictor_cols)
    if dropped:
        print(f"{predictor_set_name}: dropped predictors:", dropped)

    assert len(predictor_cols) > 0, f"No usable predictors for {predictor_set_name}"

    needed = ["feature", "phase4_clean_score", EFFECT_COL] + predictor_cols
    df = work_df[needed].replace([np.inf, -np.inf], np.nan).copy()

    for c in predictor_cols + ["phase4_clean_score", EFFECT_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().copy()
    df["feature"] = df["feature"].astype(int)

    print(f"{predictor_set_name}: usable rows after dropna = {len(df)}")
    assert len(df) >= 30, f"Too few usable rows for {predictor_set_name}: {len(df)}"

    train_df, pool_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SELECTION_SEED,
    )

    clean_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    clean_model.fit(train_df[predictor_cols].values, train_df["phase4_clean_score"].values)

    pool_df = pool_df.copy()
    pool_df["pred_clean_score"] = clean_model.predict(pool_df[predictor_cols].values)

    rho = spearmanr(pool_df["phase4_clean_score"], pool_df["pred_clean_score"]).statistic

    effect_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    effect_model.fit(train_df[predictor_cols].values, train_df[EFFECT_COL].values)
    pool_df["pred_effect_l2"] = effect_model.predict(pool_df[predictor_cols].values)

    candidate_df = pool_df.copy()

    if USE_EFFECT_MATCHING:
        lo = candidate_df["pred_effect_l2"].quantile(EFFECT_MATCH_Q_LOW)
        hi = candidate_df["pred_effect_l2"].quantile(EFFECT_MATCH_Q_HIGH)

        candidate_df = candidate_df[
            (candidate_df["pred_effect_l2"] >= lo)
            & (candidate_df["pred_effect_l2"] <= hi)
        ].copy()
    else:
        lo, hi = np.nan, np.nan

    print(f"{predictor_set_name}: candidate rows after effect matching = {len(candidate_df)}")

    n_per_group = min(N_PER_GROUP, len(candidate_df) // 3)
    assert n_per_group >= 5, f"Too few candidates for {predictor_set_name}: {len(candidate_df)}"

    clean_sel = candidate_df.sort_values("pred_clean_score", ascending=False).head(n_per_group).copy()
    messy_sel = candidate_df.sort_values("pred_clean_score", ascending=True).head(n_per_group).copy()

    used = set(clean_sel["feature"].tolist()) | set(messy_sel["feature"].tolist())
    remaining = candidate_df[~candidate_df["feature"].isin(used)].copy()

    random_sel = remaining.sample(n=n_per_group, random_state=SELECTION_SEED).copy()

    clean_sel["selection_group"] = "predicted_clean"
    messy_sel["selection_group"] = "predicted_messy"
    random_sel["selection_group"] = "random_control"

    selected_df = pd.concat([clean_sel, messy_sel, random_sel], ignore_index=True)
    selected_df["predictor_set"] = predictor_set_name
    selected_df["heldout_pool_spearman_pred_vs_true_clean_score"] = rho
    selected_df["effect_match_q_low"] = lo
    selected_df["effect_match_q_high"] = hi
    selected_df["n_predictors_used"] = len(predictor_cols)

    return selected_df

selection_dfs = []

for name, preds in PREDICTOR_SETS.items():
    sel = make_selection_for_predictor_set(name, preds)
    selection_dfs.append(sel)

    print("\nSelection:", name)
    print("n per group:", sel.groupby("selection_group").size().to_dict())
    print("held-out pool Spearman:", sel["heldout_pool_spearman_pred_vs_true_clean_score"].iloc[0])
    display(
        sel.groupby("selection_group")[
            ["pred_clean_score", "phase4_clean_score", "pred_effect_l2", EFFECT_COL]
        ].agg(["mean", "std", "min", "max"])
    )

all_selected_df = pd.concat(selection_dfs, ignore_index=True)

selected_path = OUT_DIR / "gemma_phase4_both_selected_features.csv"
all_selected_df.to_csv(selected_path, index=False)

print("saved:", selected_path)
display(all_selected_df.head())


################################################################################
# Cell 12
################################################################################
# Cell 11 — evaluate selected features for both predictor sets

eval_rows = []

for _, sel_row in tqdm(all_selected_df.iterrows(), total=len(all_selected_df), desc="Gemma Phase 4 eval"):
    feat = int(sel_row["feature"])
    group = sel_row["selection_group"]
    predictor_set = sel_row["predictor_set"]

    steered_logits_chunks = []
    downstream_chunks = []

    for b in range(0, eval_tokens.shape[0], BATCH_STEER):
        batch_tokens = eval_tokens[b:b+BATCH_STEER]
        steered_logits, downstream_feat = run_feature_add(
            feat,
            batch_tokens,
            add_value=FIXED_GLOBAL_CLAMP_VALUE,
            return_downstream=(downstream_panel is not None),
        )
        steered_logits_chunks.append(steered_logits.detach().cpu())
        if downstream_feat is not None:
            downstream_chunks.append(downstream_feat.detach().cpu())

    steered_logits = torch.cat(steered_logits_chunks, dim=0)
    delta_logits = steered_logits - clean_final_logits

    if downstream_panel is not None and feat_downstream_clean is not None and len(downstream_chunks) > 0:
        steered_downstream = torch.cat(downstream_chunks, dim=0)
        clean_downstream = feat_downstream_clean[:, downstream_panel].detach().cpu()
        delta_downstream = steered_downstream - clean_downstream
    else:
        delta_downstream = None

    row = {
        "feature": feat,
        "predictor_set": predictor_set,
        "selection_group": group,
        "pred_clean_score": float(sel_row["pred_clean_score"]),
        "phase1_clean_score": float(sel_row["phase4_clean_score"]),
        "pred_effect_l2": float(sel_row["pred_effect_l2"]),
        "phase1_effect_l2": float(sel_row[EFFECT_COL]),
        "heldout_pool_spearman_pred_vs_true_clean_score": float(sel_row["heldout_pool_spearman_pred_vs_true_clean_score"]),
        "n_eval_contexts": int(eval_tokens.shape[0]),
        "fixed_global_add_value": FIXED_GLOBAL_CLAMP_VALUE,
    }

    row.update(cosine_to_mean(delta_logits))
    row.update(logit_metrics(delta_logits, clean_final_logits, steered_logits))
    row.update(downstream_metrics(delta_downstream))

    if "downstream_feat_count_abs_delta_gt_0.05" in row:
        row["downstream_count_0.05_per_effect_l2"] = row["downstream_feat_count_abs_delta_gt_0.05"] / (row["effect_l2_mean"] + 1e-8)
    if "downstream_feat_l2_mean" in row:
        row["downstream_l2_per_effect_l2"] = row["downstream_feat_l2_mean"] / (row["effect_l2_mean"] + 1e-8)
    row["kl_clean_to_steered_per_effect_l2"] = row["kl_clean_to_steered_mean"] / (row["effect_l2_mean"] + 1e-8)

    eval_rows.append(row)

phase4_eval_df = pd.DataFrame(eval_rows)

eval_path = OUT_DIR / "gemma_phase4_both_fresh_steering_eval.csv"
phase4_eval_df.to_csv(eval_path, index=False)

print("saved:", eval_path)
display(phase4_eval_df.head())


################################################################################
# Cell 13
################################################################################
# Cell 12 — group summaries and statistical tests for both predictor sets

summary_metrics = [
    "stability_to_mean_abs_cos",
    "stability_to_mean_signed_cos",
    "effect_l2_mean",
    "effect_cv",
    "kl_clean_to_steered_mean",
    "kl_clean_to_steered_per_effect_l2",
    "downstream_feat_l2_mean",
    "downstream_l2_per_effect_l2",
    "downstream_feat_effective_moved_mean",
    "downstream_feat_count_abs_delta_gt_0.05",
    "downstream_count_0.05_per_effect_l2",
]

summary_metrics = [m for m in summary_metrics if m in phase4_eval_df.columns]

group_summary = phase4_eval_df.groupby(["predictor_set", "selection_group"])[summary_metrics].agg(["mean", "std", "median", "count"])
summary_path = OUT_DIR / "gemma_phase4_both_group_summary.csv"
group_summary.to_csv(summary_path)

print("Group summary:")
display(group_summary)
print("saved:", summary_path)

test_rows = []

for predictor_set, ps_df in phase4_eval_df.groupby("predictor_set"):
    clean = ps_df[ps_df["selection_group"] == "predicted_clean"]
    messy = ps_df[ps_df["selection_group"] == "predicted_messy"]
    random_group = ps_df[ps_df["selection_group"] == "random_control"]

    for metric in summary_metrics:
        for a_name, a_df, b_name, b_df in [
            ("clean", clean, "messy", messy),
            ("clean", clean, "random", random_group),
            ("messy", messy, "random", random_group),
        ]:
            a = a_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            b = b_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) < 3 or len(b) < 3:
                continue
            try:
                stat, p = mannwhitneyu(a, b, alternative="two-sided")
            except Exception:
                stat, p = np.nan, np.nan

            test_rows.append({
                "predictor_set": predictor_set,
                "metric": metric,
                "group_a": a_name,
                "group_b": b_name,
                "mean_a": a.mean(),
                "mean_b": b.mean(),
                "median_a": a.median(),
                "median_b": b.median(),
                "diff_mean_a_minus_b": a.mean() - b.mean(),
                "mannwhitney_p": p,
                "n_a": len(a),
                "n_b": len(b),
            })

tests_df = pd.DataFrame(test_rows)
tests_path = OUT_DIR / "gemma_phase4_both_group_tests.csv"
tests_df.to_csv(tests_path, index=False)

print("Tests:")
display(tests_df)
print("saved:", tests_path)


################################################################################
# Cell 14
################################################################################
# Cell 13 — verdict for both predictor sets

verdict_rows = []

def group_mean(df, predictor_set, metric, group):
    sub = df[(df["predictor_set"] == predictor_set) & (df["selection_group"] == group)]
    if metric not in sub.columns:
        return np.nan
    return sub[metric].replace([np.inf, -np.inf], np.nan).mean()

for predictor_set in sorted(phase4_eval_df["predictor_set"].unique()):
    clean_stab = group_mean(phase4_eval_df, predictor_set, "stability_to_mean_abs_cos", "predicted_clean")
    messy_stab = group_mean(phase4_eval_df, predictor_set, "stability_to_mean_abs_cos", "predicted_messy")
    rand_stab = group_mean(phase4_eval_df, predictor_set, "stability_to_mean_abs_cos", "random_control")

    clean_coll = group_mean(phase4_eval_df, predictor_set, "downstream_count_0.05_per_effect_l2", "predicted_clean")
    messy_coll = group_mean(phase4_eval_df, predictor_set, "downstream_count_0.05_per_effect_l2", "predicted_messy")
    rand_coll = group_mean(phase4_eval_df, predictor_set, "downstream_count_0.05_per_effect_l2", "random_control")

    clean_eff = group_mean(phase4_eval_df, predictor_set, "effect_l2_mean", "predicted_clean")
    messy_eff = group_mean(phase4_eval_df, predictor_set, "effect_l2_mean", "predicted_messy")
    rand_eff = group_mean(phase4_eval_df, predictor_set, "effect_l2_mean", "random_control")

    success_stability = (not np.isnan(clean_stab)) and (not np.isnan(messy_stab)) and clean_stab > messy_stab
    success_collateral = (not np.isnan(clean_coll)) and (not np.isnan(messy_coll)) and clean_coll < messy_coll

    if success_stability and success_collateral:
        label = "strong"
    elif success_stability or success_collateral:
        label = "partial"
    else:
        label = "weak"

    verdict_rows.append({
        "model": MODEL_NAME,
        "predictor_set": predictor_set,
        "n_per_group": int(phase4_eval_df[phase4_eval_df["predictor_set"] == predictor_set].groupby("selection_group").size().min()),
        "n_eval_contexts": int(N_EVAL_CONTEXTS),
        "clean_stability_abs_mean": clean_stab,
        "messy_stability_abs_mean": messy_stab,
        "random_stability_abs_mean": rand_stab,
        "clean_minus_messy_stability_abs": clean_stab - messy_stab,
        "clean_collateral_per_effect": clean_coll,
        "messy_collateral_per_effect": messy_coll,
        "random_collateral_per_effect": rand_coll,
        "clean_minus_messy_collateral_per_effect": clean_coll - messy_coll,
        "clean_effect_l2": clean_eff,
        "messy_effect_l2": messy_eff,
        "random_effect_l2": rand_eff,
        "phase4_result": label,
    })

verdict_df = pd.DataFrame(verdict_rows)

verdict_path_csv = OUT_DIR / "gemma_phase4_both_verdict.csv"
verdict_path_json = OUT_DIR / "gemma_phase4_both_verdict.json"

verdict_df.to_csv(verdict_path_csv, index=False)
with open(verdict_path_json, "w") as f:
    json.dump(verdict_rows, f, indent=2)

print("=" * 80)
print("GEMMA PHASE 4 BOTH-PREDICTOR VERDICT")
print("=" * 80)
display(verdict_df)
print("saved:", verdict_path_csv)
print("saved:", verdict_path_json)


################################################################################
# Cell 15
################################################################################
# Cell 14 — plots

plots_dir = OUT_DIR / "plots"
plots_dir.mkdir(exist_ok=True)

plot_metrics = [
    "stability_to_mean_abs_cos",
    "downstream_count_0.05_per_effect_l2",
    "downstream_feat_l2_mean",
    "effect_l2_mean",
]

plot_metrics = [m for m in plot_metrics if m in phase4_eval_df.columns]

for predictor_set in sorted(phase4_eval_df["predictor_set"].unique()):
    ps_df = phase4_eval_df[phase4_eval_df["predictor_set"] == predictor_set]
    for metric in plot_metrics:
        plt.figure(figsize=(7, 4))
        groups = ["predicted_clean", "random_control", "predicted_messy"]
        data = [
            ps_df[ps_df["selection_group"] == g][metric].dropna().values
            for g in groups
        ]
        plt.boxplot(data, labels=groups)
        plt.ylabel(metric)
        plt.title(f"Gemma Phase 4 {predictor_set}: {metric}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        out = plots_dir / f"gemma_phase4_{predictor_set}_{metric}.png"
        plt.savefig(out, dpi=150)
        plt.show()

print("saved plots to:", plots_dir)


################################################################################
# Cell 16
################################################################################
# Cell 15 — save outputs to Google Drive

SAVE_TO_DRIVE = True
DRIVE_OUT = "/content/drive/MyDrive/SAE Prediction/gemma_phase4_both_predictors_outputs"

if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")

    drive_out = Path(DRIVE_OUT)
    drive_out.mkdir(parents=True, exist_ok=True)

    for path in OUT_DIR.glob("*"):
        if path.is_file():
            shutil.copy2(path, drive_out / path.name)
            print("copied:", path.name)

    if (OUT_DIR / "plots").exists():
        plot_out = drive_out / "plots"
        plot_out.mkdir(exist_ok=True)
        for path in (OUT_DIR / "plots").glob("*.png"):
            shutil.copy2(path, plot_out / path.name)

    print("Saved to:", drive_out)
else:
    print("SAVE_TO_DRIVE is False.")
