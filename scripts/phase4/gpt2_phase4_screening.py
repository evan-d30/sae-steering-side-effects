#!/usr/bin/env python
# Converted from notebooks/phase4/gpt2_phase4_screening.ipynb
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

pd.set_option("display.max_rows", 220)
pd.set_option("display.max_columns", 220)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

OUT_DIR = Path("/content/gpt2_phase4_full_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("OUT_DIR:", OUT_DIR)


################################################################################
# Cell 3
################################################################################
# Cell 2.5 — optional Hugging Face token login
# Use this if downloads are slow/rate-limited or for gated models.

USE_HF_TOKEN = False

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
# Cell 3 — locate robust Phase 1/2/3 outputs

# For GPT-2 robust run, likely:
# /content/drive/MyDrive/SAE Prediction/robust_phase123_outputs
#
# For Gemma full-scale run, likely:
# /content/drive/MyDrive/SAE Prediction/gemma_fullscale_phase123_outputs

USE_GOOGLE_DRIVE = True
DRIVE_FOLDER = "/content/drive/MyDrive/SAE Prediction/robust_phase123_outputs"

# If you uploaded files directly to Colab, set USE_GOOGLE_DRIVE=False and upload the files to /content.
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

display(merged_df.head())

assert config.get("MODEL_NAME", "") == "gpt2-small", "This notebook is the full GPT-2 Phase 4 notebook. Use the Gemma-adapted version for Gemma outputs."


################################################################################
# Cell 6
################################################################################
# Cell 5 — define Phase 4 target and predictor set

# Primary Phase 4 goal:
# Select features predicted to be stable and low-collateral.
#
# Clean score = standardized stability - standardized collateral_per_effect
# Higher = predicted cleaner.

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
work_df["phase4_collateral_per_effect"] = np.log1p(work_df[COLLATERAL_COL]) / (work_df[EFFECT_COL].abs() + 1e-8)
work_df["phase4_clean_score"] = (
    zscore_np(work_df[STABILITY_COL])
    - zscore_np(work_df["phase4_collateral_per_effect"])
)

# Geometry-only predictor set.
# Drop crowding_topk_sum_abs_cos because it is linearly redundant with mean when K is fixed.
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

# Full no-magnitude predictor set.
# Excludes obvious activation magnitude columns to avoid selecting features based only on activation scale.
activation_magnitude_keywords = ["act_mean", "act_std", "act_max", "act_q95", "clamp"]
candidate_predictor_cols = [
    c for c in work_df.columns
    if c != "feature"
    and work_df[c].dtype != "object"
    and not any(lbl in c for lbl in [
        "stability_to_mean",
        "pairwise_cos",
        "effect_l2",
        "effect_l1",
        "effect_linf",
        "effect_cv",
        "kl_",
        "logit_count",
        "top10_abs_delta",
        "top50_abs_delta",
        "top100_abs_delta",
        "top500_abs_delta",
        "logit_delta",
        "downstream_feat",
        "phase4_clean_score",
        "phase4_collateral_per_effect",
        "natural_final_act",
    ])
]

full_no_magnitude_predictors = [
    c for c in candidate_predictor_cols
    if not any(k in c for k in activation_magnitude_keywords)
]

PREDICTOR_SET_NAME = "geometry_only"  # options: "geometry_only", "full_no_magnitude"

if PREDICTOR_SET_NAME == "geometry_only":
    predictor_cols = geometry_predictors
elif PREDICTOR_SET_NAME == "full_no_magnitude":
    predictor_cols = full_no_magnitude_predictors
else:
    raise ValueError(PREDICTOR_SET_NAME)

print("Predictor set:", PREDICTOR_SET_NAME)
print("n predictors:", len(predictor_cols))
print(predictor_cols)

assert len(predictor_cols) > 0, "No predictors found."

# Keep clean rows only.
needed = ["feature", "phase4_clean_score", EFFECT_COL] + predictor_cols
phase4_df = work_df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
phase4_df["feature"] = phase4_df["feature"].astype(int)

print("usable features:", len(phase4_df))
display(phase4_df[["feature", "phase4_clean_score", EFFECT_COL] + predictor_cols].head())


################################################################################
# Cell 7
################################################################################
# Cell 6 — train predictor and select clean/messy/random held-out features

# We split by feature.
# Train set teaches the screening predictor.
# Candidate pool is held out from training; selected clean/messy/random features come only from this pool.

TEST_SIZE = 0.40
SELECTION_SEED = 0
N_PER_GROUP = 50

# Effect matching reduces the trivial confound where clean features only look clean because they have weaker effects.
# We fit an effect predictor on train features and restrict candidate pool to central predicted effect range.
USE_EFFECT_MATCHING = True
EFFECT_MATCH_Q_LOW = 0.10
EFFECT_MATCH_Q_HIGH = 0.90

train_df, pool_df = train_test_split(
    phase4_df,
    test_size=TEST_SIZE,
    random_state=SELECTION_SEED,
)

# Clean-score predictor.
clean_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
clean_model.fit(train_df[predictor_cols].values, train_df["phase4_clean_score"].values)

pool_df = pool_df.copy()
pool_df["pred_clean_score"] = clean_model.predict(pool_df[predictor_cols].values)

# Diagnostic: train-pool relationship using existing Phase 1 labels, not used for selection except after prediction.
rho = spearmanr(pool_df["phase4_clean_score"], pool_df["pred_clean_score"]).statistic
print(f"Held-out pool Spearman(pred_clean_score, true_clean_score) = {rho:.3f}")

# Effect-size predictor for optional matching.
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
    print(f"Effect matching: kept {len(candidate_df)} pool features between predicted effect {lo:.3f} and {hi:.3f}")

max_possible = len(candidate_df) // 3
if max_possible < N_PER_GROUP:
    print(f"Reducing N_PER_GROUP from {N_PER_GROUP} to {max_possible}")
    N_PER_GROUP = max_possible

assert N_PER_GROUP >= 5, "Too few candidate features. Increase pool size or lower N_PER_GROUP."

clean_sel = candidate_df.sort_values("pred_clean_score", ascending=False).head(N_PER_GROUP).copy()
messy_sel = candidate_df.sort_values("pred_clean_score", ascending=True).head(N_PER_GROUP).copy()

used = set(clean_sel["feature"].tolist()) | set(messy_sel["feature"].tolist())
remaining = candidate_df[~candidate_df["feature"].isin(used)].copy()
random_sel = remaining.sample(n=N_PER_GROUP, random_state=SELECTION_SEED).copy()

clean_sel["selection_group"] = "predicted_clean"
messy_sel["selection_group"] = "predicted_messy"
random_sel["selection_group"] = "random_control"

selected_df = pd.concat([clean_sel, messy_sel, random_sel], ignore_index=True)

selected_path = OUT_DIR / "gpt2_phase4_selected_features.csv"
selected_df.to_csv(selected_path, index=False)

print("Selected GPT-2 Phase 4 features by group:")
display(selected_df.groupby("selection_group")[["pred_clean_score", "phase4_clean_score", "pred_effect_l2", EFFECT_COL]].agg(["mean", "std", "min", "max"]))
print("saved:", selected_path)
display(selected_df[["feature", "selection_group", "pred_clean_score", "phase4_clean_score", "pred_effect_l2", EFFECT_COL]].head(20))


################################################################################
# Cell 8
################################################################################
# Cell 7 — load model and SAEs for fresh held-out steering evaluation

MODEL_NAME = config.get("MODEL_NAME", "gpt2-small")
SAE_RELEASE = config.get("SAE_RELEASE", "gpt2-small-res-jb")
SAE_ID = config.get("SAE_ID", "blocks.8.hook_resid_pre")
USE_DOWNSTREAM_SAE = config.get("USE_DOWNSTREAM_SAE", True)
DOWNSTREAM_SAE_ID = config.get("DOWNSTREAM_SAE_ID", "blocks.10.hook_resid_pre")

MODEL_DTYPE_STR = config.get("MODEL_DTYPE", None)
if MODEL_DTYPE_STR and "bfloat16" in MODEL_DTYPE_STR and torch.cuda.is_available():
    MODEL_DTYPE = torch.bfloat16
else:
    MODEL_DTYPE = torch.float32

model_kwargs = {}
if "gemma" in MODEL_NAME.lower():
    model_kwargs["dtype"] = MODEL_DTYPE

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

def infer_hook_name(sae_id, model, cfg=None, default_hook=None):
    import re

    hook_names = set(model.hook_dict.keys())

    # Direct config hook first.
    candidates = []
    if default_hook is not None:
        candidates.append(default_hook)

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

    # Gemma Scope ID mapping.
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

    # GPT-style fallback if SAE_ID already is hook.
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
# Cell 9
################################################################################
# Cell 8 — build fresh held-out corpus contexts

# Use validation split by default so Phase 4 contexts are fresh relative to train[:...] Phase 1 runs.
HF_DATASET_NAME = "Salesforce/wikitext"
HF_DATASET_CONFIG = "wikitext-103-raw-v1"
HF_DATASET_SPLIT = "validation"

CONTEXT_LEN = int(config.get("CONTEXT_LEN", 48))
N_HELDOUT_CONTEXTS = 1024
N_EVAL_CONTEXTS = 512
MIN_DISTINCT_TEXTS = 500

BATCH_CONTEXTS = 16
BATCH_STEER = 16

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

# Fixed random evaluation subset.
rng = np.random.default_rng(123)
eval_idx = rng.choice(np.arange(tokens.shape[0]), size=N_EVAL_CONTEXTS, replace=False)
eval_idx_t = torch.tensor(eval_idx, dtype=torch.long, device=device)
eval_tokens = tokens[eval_idx_t]

print("heldout tokens:", tuple(tokens.shape))
print("eval tokens:", tuple(eval_tokens.shape))
print(model.tokenizer.decode(eval_tokens[0].detach().cpu().tolist()))


################################################################################
# Cell 10
################################################################################
# Cell 9 — clean pass on held-out evaluation contexts

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
# Cell 11
################################################################################
# Cell 10 — choose downstream panel and define steering/metric helpers

ACTIVE_THRESH = 1e-6
DOWNSTREAM_PANEL_SIZE = 2048 if "gemma" not in MODEL_NAME.lower() else 1024
DOWNSTREAM_ABS_THRESHOLDS = [0.01, 0.05, 0.10]
ABS_LOGIT_THRESHOLDS = [0.05, 0.10, 0.20, 0.50]

downstream_panel = None
if feat_downstream_clean is not None:
    downstream_active_freq = (feat_downstream_clean > ACTIVE_THRESH).float().mean(dim=0)
    downstream_panel = torch.topk(downstream_active_freq, k=min(DOWNSTREAM_PANEL_SIZE, feat_downstream_clean.shape[-1])).indices.detach().cpu()
    print("downstream panel size:", len(downstream_panel))

# Clamp value from robust run.
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
# Cell 12
################################################################################
# Cell 11 — run Phase 4 fresh steering evaluation

eval_rows = []

for _, sel_row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Phase 4 selected feature eval"):
    feat = int(sel_row["feature"])
    group = sel_row["selection_group"]

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
        "selection_group": group,
        "pred_clean_score": float(sel_row["pred_clean_score"]),
        "phase1_clean_score": float(sel_row["phase4_clean_score"]),
        "pred_effect_l2": float(sel_row["pred_effect_l2"]),
        "phase1_effect_l2": float(sel_row[EFFECT_COL]),
        "n_eval_contexts": int(eval_tokens.shape[0]),
        "fixed_global_add_value": FIXED_GLOBAL_CLAMP_VALUE,
    }

    row.update(cosine_to_mean(delta_logits))
    row.update(logit_metrics(delta_logits, clean_final_logits, steered_logits))
    row.update(downstream_metrics(delta_downstream))

    # Normalized collateral: lower is better.
    if "downstream_feat_count_abs_delta_gt_0.05" in row:
        row["downstream_count_0.05_per_effect_l2"] = row["downstream_feat_count_abs_delta_gt_0.05"] / (row["effect_l2_mean"] + 1e-8)
    if "downstream_feat_l2_mean" in row:
        row["downstream_l2_per_effect_l2"] = row["downstream_feat_l2_mean"] / (row["effect_l2_mean"] + 1e-8)
    row["kl_clean_to_steered_per_effect_l2"] = row["kl_clean_to_steered_mean"] / (row["effect_l2_mean"] + 1e-8)

    eval_rows.append(row)

phase4_eval_df = pd.DataFrame(eval_rows)

eval_path = OUT_DIR / "gpt2_phase4_fresh_steering_eval.csv"
phase4_eval_df.to_csv(eval_path, index=False)

print("saved:", eval_path)
display(phase4_eval_df.head())


################################################################################
# Cell 13
################################################################################
# Cell 12 — group summary and statistical comparisons

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

group_summary = phase4_eval_df.groupby("selection_group")[summary_metrics].agg(["mean", "std", "median", "count"])
summary_path = OUT_DIR / "gpt2_phase4_group_summary.csv"
group_summary.to_csv(summary_path)

print("Group summary:")
display(group_summary)
print("saved:", summary_path)

# Pairwise clean vs messy tests.
test_rows = []

clean = phase4_eval_df[phase4_eval_df["selection_group"] == "predicted_clean"]
messy = phase4_eval_df[phase4_eval_df["selection_group"] == "predicted_messy"]
random_group = phase4_eval_df[phase4_eval_df["selection_group"] == "random_control"]

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
tests_path = OUT_DIR / "gpt2_phase4_group_tests.csv"
tests_df.to_csv(tests_path, index=False)

print("Clean vs messy / random tests:")
display(tests_df)
print("saved:", tests_path)


################################################################################
# Cell 14
################################################################################
# Cell 13 — Phase 4 verdict

# Desired pattern:
# predicted_clean should have:
# - higher stability than predicted_messy,
# - lower downstream collateral per effect than predicted_messy,
# - ideally similar effect_l2, or at least better collateral/effect ratio.

def group_mean(metric, group):
    sub = phase4_eval_df[phase4_eval_df["selection_group"] == group]
    if metric not in sub.columns:
        return np.nan
    return sub[metric].replace([np.inf, -np.inf], np.nan).mean()

clean_stab = group_mean("stability_to_mean_abs_cos", "predicted_clean")
messy_stab = group_mean("stability_to_mean_abs_cos", "predicted_messy")
rand_stab = group_mean("stability_to_mean_abs_cos", "random_control")

clean_coll = group_mean("downstream_count_0.05_per_effect_l2", "predicted_clean")
messy_coll = group_mean("downstream_count_0.05_per_effect_l2", "predicted_messy")
rand_coll = group_mean("downstream_count_0.05_per_effect_l2", "random_control")

clean_eff = group_mean("effect_l2_mean", "predicted_clean")
messy_eff = group_mean("effect_l2_mean", "predicted_messy")
rand_eff = group_mean("effect_l2_mean", "random_control")

verdict = {
    "model": MODEL_NAME,
    "phase4_scale": "full_gpt2",
    "predictor_set": PREDICTOR_SET_NAME,
    "n_per_group": int(N_PER_GROUP),
    "n_eval_contexts": int(N_EVAL_CONTEXTS),
    "clean_stability_abs_mean": None if np.isnan(clean_stab) else float(clean_stab),
    "messy_stability_abs_mean": None if np.isnan(messy_stab) else float(messy_stab),
    "random_stability_abs_mean": None if np.isnan(rand_stab) else float(rand_stab),
    "clean_minus_messy_stability_abs": None if np.isnan(clean_stab) or np.isnan(messy_stab) else float(clean_stab - messy_stab),
    "clean_collateral_per_effect": None if np.isnan(clean_coll) else float(clean_coll),
    "messy_collateral_per_effect": None if np.isnan(messy_coll) else float(messy_coll),
    "random_collateral_per_effect": None if np.isnan(rand_coll) else float(rand_coll),
    "clean_minus_messy_collateral_per_effect": None if np.isnan(clean_coll) or np.isnan(messy_coll) else float(clean_coll - messy_coll),
    "clean_effect_l2": None if np.isnan(clean_eff) else float(clean_eff),
    "messy_effect_l2": None if np.isnan(messy_eff) else float(messy_eff),
    "random_effect_l2": None if np.isnan(rand_eff) else float(rand_eff),
}

verdict_path = OUT_DIR / "gpt2_phase4_verdict.json"
with open(verdict_path, "w") as f:
    json.dump(verdict, f, indent=2)

print("=" * 80)
print("FULL GPT-2 PHASE 4 SCREENING VERDICT")
print("=" * 80)
print(json.dumps(verdict, indent=2))
print("-" * 80)

success_stability = (not np.isnan(clean_stab)) and (not np.isnan(messy_stab)) and clean_stab > messy_stab
success_collateral = (not np.isnan(clean_coll)) and (not np.isnan(messy_coll)) and clean_coll < messy_coll

if success_stability and success_collateral:
    print("VERDICT: STRONG PHASE 4 RESULT.")
    print("Predicted-clean features are more stable and lower-collateral-per-effect than predicted-messy features.")
elif success_stability or success_collateral:
    print("VERDICT: PARTIAL PHASE 4 RESULT.")
    print("The screen helps on one main axis but not both. Inspect group summaries.")
else:
    print("VERDICT: WEAK PHASE 4 RESULT.")
    print("Predicted-clean features did not clearly outperform predicted-messy features on fresh held-out contexts.")

print("saved:", verdict_path)


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

for metric in plot_metrics:
    plt.figure(figsize=(7, 4))
    groups = ["predicted_clean", "random_control", "predicted_messy"]
    data = [
        phase4_eval_df[phase4_eval_df["selection_group"] == g][metric].dropna().values
        for g in groups
    ]
    plt.boxplot(data, labels=groups)
    plt.ylabel(metric)
    plt.title(f"Phase 4: {metric}")
    plt.xticks(rotation=15)
    plt.tight_layout()
    out = plots_dir / f"phase4_{metric}.png"
    plt.savefig(out, dpi=150)
    plt.show()

print("saved plots to:", plots_dir)


################################################################################
# Cell 16
################################################################################
# Cell 15 — save outputs to Google Drive

SAVE_TO_DRIVE = True
DRIVE_OUT = "/content/drive/MyDrive/SAE Prediction/gpt2_phase4_full_outputs"

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
