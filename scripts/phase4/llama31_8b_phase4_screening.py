#!/usr/bin/env python
# Converted from notebooks/phase4/llama31_8b_phase4_screening.ipynb
# Original notebook removed from the repository; this script preserves the code cells for reproducibility.


################################################################################
# Cell 1
################################################################################
# Cell 1 — Phase 4 setup and feature selection

from pathlib import Path
import json, random, gc, math, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from datasets import load_dataset

torch.set_grad_enabled(False)

PHASE4_DIR = OUT_DIR / "llama31_phase4_outputs"
PHASE4_DIR.mkdir(parents=True, exist_ok=True)

PHASE4_CONFIG = dict(
    RANDOM_SEED=1,
    TRAIN_FRAC=0.70,
    GROUP_SIZE=25,
    FRESH_N_CONTEXTS=256,
    CONTEXT_LEN=CONFIG["CONTEXT_LEN"],
    BATCH_SIZE=CONFIG["BATCH_STEER"],
    DOWNSTREAM_PANEL_SIZE=CONFIG["DOWNSTREAM_PANEL_SIZE"],
    DOWNSTREAM_DELTA_THRESH=CONFIG["DOWNSTREAM_DELTA_THRESH"],
    ALPHA=CONFIG["FIXED_GLOBAL_CLAMP_VALUE"],
)

print("Phase 4 output dir:", PHASE4_DIR)
print("Phase 4 config:", PHASE4_CONFIG)

merged_path = OUT_DIR / "robust_phase2_phase3_merged.csv"
assert merged_path.exists(), f"Missing merged file: {merged_path}"

merged = pd.read_csv(merged_path)
print("Merged shape:", merged.shape)

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of these columns found: {candidates}")
    return None

ABS_STAB_COL = find_col(merged, ["stability_to_mean_abs_cos__fixed_global_add__mixed"])
SIGNED_STAB_COL = find_col(merged, ["stability_to_mean_signed_cos__fixed_global_add__mixed"])
COLLATERAL_COL = find_col(merged, [
    "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
    "downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed",
])
EFFECT_L2_COL = find_col(merged, ["effect_l2_mean__fixed_global_add__mixed"])

print("Using columns:")
print(" abs stability:", ABS_STAB_COL)
print(" signed stability:", SIGNED_STAB_COL)
print(" collateral:", COLLATERAL_COL)
print(" effect L2:", EFFECT_L2_COL)

EPS = 1e-8
merged["phase4_collateral_per_effect"] = merged[COLLATERAL_COL] / (merged[EFFECT_L2_COL] + EPS)

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + EPS)

merged["phase4_clean_score"] = zscore(merged[ABS_STAB_COL]) - zscore(np.log1p(merged["phase4_collateral_per_effect"]))

def available(cols):
    return [c for c in cols if c in merged.columns]

PREDICTOR_SETS = {
    "frequency_only": available(["phase1_final_act_freq", "token_act_freq"]),
    "activation_magnitude_only": available(["token_act_mean", "token_act_std", "token_act_max"]),
    "geometry_only": available([
        "decoder_norm", "encoder_norm", "encoder_decoder_cos",
        "crowding_topk_mean_abs_cos", "crowding_topk_sum_abs_cos", "crowding_max_abs_cos",
    ]),
    "direct_logit_only": available([
        "direct_logit_l2", "direct_logit_linf", "direct_logit_entropy",
        "direct_logit_top10_mass_frac", "direct_logit_top100_mass_frac",
    ]),
    "coactivation_only": available(["coact_count_mean", "coact_entropy_norm", "coact_top20_mass"]),
}

PREDICTOR_SETS["full_no_magnitude"] = available([
    "phase1_final_act_freq", "token_binary_entropy", "token_activation_entropy_norm", "token_act_kurtosis",
    "decoder_norm", "encoder_norm", "encoder_decoder_cos",
    "crowding_topk_mean_abs_cos", "crowding_topk_sum_abs_cos", "crowding_max_abs_cos",
    "coact_count_mean", "coact_entropy_norm", "coact_top20_mass",
    "direct_logit_l2", "direct_logit_linf", "direct_logit_entropy",
    "direct_logit_top10_mass_frac", "direct_logit_top100_mass_frac",
])

SELECTORS = ["full_no_magnitude", "geometry_only", "direct_logit_only"]
for name in SELECTORS:
    print(name, len(PREDICTOR_SETS[name]), PREDICTOR_SETS[name])

rng = np.random.default_rng(PHASE4_CONFIG["RANDOM_SEED"])
needed_cols = ["feature", "phase4_clean_score", EFFECT_L2_COL] + sorted(set(sum([PREDICTOR_SETS[s] for s in SELECTORS], [])))
phase4_df = merged[needed_cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
phase4_df["feature"] = phase4_df["feature"].astype(int)
print("Usable rows:", phase4_df.shape)

train_idx, pool_idx = train_test_split(
    np.arange(len(phase4_df)),
    train_size=PHASE4_CONFIG["TRAIN_FRAC"],
    random_state=PHASE4_CONFIG["RANDOM_SEED"],
    shuffle=True,
)
train_df = phase4_df.iloc[train_idx].copy()
pool_df = phase4_df.iloc[pool_idx].copy()
print("Train features:", len(train_df))
print("Held-out pool features:", len(pool_df))

def fit_predict_selector(selector_name):
    pred_cols = PREDICTOR_SETS[selector_name]
    assert len(pred_cols) > 0, f"No predictor columns for {selector_name}"
    clean_model = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["RIDGE_ALPHA"]))
    clean_model.fit(train_df[pred_cols], train_df["phase4_clean_score"])
    effect_model = make_pipeline(StandardScaler(), Ridge(alpha=CONFIG["RIDGE_ALPHA"]))
    effect_model.fit(train_df[pred_cols], train_df[EFFECT_L2_COL])
    out = pool_df.copy()
    out["selector"] = selector_name
    out["pred_clean_score"] = clean_model.predict(out[pred_cols])
    out["pred_effect_l2"] = effect_model.predict(out[pred_cols])
    return out

def select_groups(pred_df, group_size):
    df = pred_df.copy()
    lo, hi = df["pred_effect_l2"].quantile([0.10, 0.90])
    effect_pool = df[(df["pred_effect_l2"] >= lo) & (df["pred_effect_l2"] <= hi)].copy()
    if len(effect_pool) < group_size * 3:
        lo, hi = df["pred_effect_l2"].quantile([0.05, 0.95])
        effect_pool = df[(df["pred_effect_l2"] >= lo) & (df["pred_effect_l2"] <= hi)].copy()
    if len(effect_pool) < group_size * 3:
        effect_pool = df.copy()
    effect_pool = effect_pool.sort_values("pred_clean_score", ascending=False)
    clean = effect_pool.head(group_size).copy(); clean["selection_group"] = "predicted_clean"
    messy = effect_pool.tail(group_size).copy(); messy["selection_group"] = "predicted_messy"
    used = set(clean["feature"]) | set(messy["feature"])
    remaining = effect_pool[~effect_pool["feature"].isin(used)].copy()
    if len(remaining) >= group_size:
        random_control = remaining.sample(n=group_size, random_state=PHASE4_CONFIG["RANDOM_SEED"]).copy()
    else:
        broader_remaining = df[~df["feature"].isin(used)].copy()
        random_control = broader_remaining.sample(n=min(group_size, len(broader_remaining)), random_state=PHASE4_CONFIG["RANDOM_SEED"]).copy()
    random_control["selection_group"] = "random_control"
    return pd.concat([clean, random_control, messy], ignore_index=True)

selected_all = []
for selector in SELECTORS:
    pred_df = fit_predict_selector(selector)
    selected_all.append(select_groups(pred_df, PHASE4_CONFIG["GROUP_SIZE"]))
selected_features = pd.concat(selected_all, ignore_index=True)
selected_path = PHASE4_DIR / "llama31_phase4_selected_features.csv"
selected_features.to_csv(selected_path, index=False)
print("Selected features saved:", selected_path)
display(selected_features.groupby(["selector", "selection_group"]).size())
display(selected_features.head())


################################################################################
# Cell 2
################################################################################
# Cell 2 — Fresh validation contexts

def normalize_text(t):
    return " ".join(str(t).split())

raw_val = load_dataset(CONFIG["HF_DATASET_NAME"], CONFIG["HF_DATASET_CONFIG"], split="validation")

texts = []
seen = set()
for row in raw_val:
    t = normalize_text(row["text"])
    if len(t) < 40:
        continue
    if t in seen:
        continue
    seen.add(t)
    texts.append(t)
    if len(texts) >= 3000:
        break

fresh_tokens = []
for t in tqdm(texts, desc="fresh tokenizing"):
    toks = model.to_tokens(t, prepend_bos=True).squeeze(0)
    if toks.numel() >= PHASE4_CONFIG["CONTEXT_LEN"]:
        fresh_tokens.append(toks[:PHASE4_CONFIG["CONTEXT_LEN"]])
    if len(fresh_tokens) >= PHASE4_CONFIG["FRESH_N_CONTEXTS"]:
        break

fresh_tokens = torch.stack(fresh_tokens).to(CONFIG["DEVICE"])
print("Fresh tokens:", tuple(fresh_tokens.shape))
assert fresh_tokens.shape[0] == PHASE4_CONFIG["FRESH_N_CONTEXTS"], "Not enough fresh contexts"
assert fresh_tokens.shape[1] == PHASE4_CONFIG["CONTEXT_LEN"], "Wrong context length"


################################################################################
# Cell 3
################################################################################
# Cell 3 — Precompute clean logits and downstream panel

PRIMARY_HOOK = CONFIG["HOOK_NAME"]
DOWNSTREAM_HOOK = CONFIG["DOWNSTREAM_HOOK_NAME"]
DEVICE = CONFIG["DEVICE"]
BATCH = PHASE4_CONFIG["BATCH_SIZE"]

model.eval(); sae.eval(); downstream_sae.eval()

def get_sae_device(s):
    return next(s.parameters()).device

sae_device = get_sae_device(sae)
downstream_sae_device = get_sae_device(downstream_sae)
clean_logits_cpu = []
clean_downstream_z_cpu = []

@torch.no_grad()
def run_clean_batch(tokens_b):
    captured = {}
    def save_downstream(acts, hook):
        captured["downstream"] = acts[:, -1, :].detach()
        return acts
    logits = model.run_with_hooks(tokens_b, fwd_hooks=[(DOWNSTREAM_HOOK, save_downstream)])[:, -1, :].detach()
    down_acts = captured["downstream"].to(downstream_sae_device)
    down_z = downstream_sae.encode(down_acts).detach().float().cpu()
    return logits.detach().to(torch.float16).cpu(), down_z

for start in tqdm(range(0, fresh_tokens.shape[0], BATCH), desc="clean fresh pass"):
    end = min(start + BATCH, fresh_tokens.shape[0])
    logits_b, down_z_b = run_clean_batch(fresh_tokens[start:end])
    clean_logits_cpu.append(logits_b)
    clean_downstream_z_cpu.append(down_z_b)

clean_logits_cpu = torch.cat(clean_logits_cpu, dim=0)
clean_downstream_z_cpu = torch.cat(clean_downstream_z_cpu, dim=0)
print("clean logits:", tuple(clean_logits_cpu.shape))
print("clean downstream z:", tuple(clean_downstream_z_cpu.shape))
panel_size = min(PHASE4_CONFIG["DOWNSTREAM_PANEL_SIZE"], clean_downstream_z_cpu.shape[1])
panel = torch.topk(clean_downstream_z_cpu.mean(dim=0), k=panel_size).indices.cpu()
print("downstream panel size:", len(panel))
torch.cuda.empty_cache()


################################################################################
# Cell 4
################################################################################
# Cell 4 — Evaluate held-out selected features on fresh contexts

def get_decoder_matrix(s):
    W_dec = s.W_dec.detach()
    d_sae = getattr(s.cfg, "d_sae", None)
    d_in = getattr(s.cfg, "d_in", None)
    if d_sae is not None and W_dec.shape[0] == d_sae:
        return W_dec
    if d_sae is not None and W_dec.shape[1] == d_sae:
        return W_dec.T
    if d_in is not None and W_dec.shape[1] == d_in:
        return W_dec
    if d_in is not None and W_dec.shape[0] == d_in:
        return W_dec.T
    return W_dec

W_dec_primary = get_decoder_matrix(sae).to(DEVICE)

@torch.no_grad()
def run_steered_batch(tokens_b, feature_id):
    d_f = W_dec_primary[int(feature_id)].detach().to(DEVICE)
    captured = {}
    def add_feature_hook(acts, hook):
        acts = acts.clone()
        acts[:, -1, :] = acts[:, -1, :] + PHASE4_CONFIG["ALPHA"] * d_f.to(acts.dtype)
        return acts
    def save_downstream(acts, hook):
        captured["downstream"] = acts[:, -1, :].detach()
        return acts
    logits = model.run_with_hooks(
        tokens_b,
        fwd_hooks=[(PRIMARY_HOOK, add_feature_hook), (DOWNSTREAM_HOOK, save_downstream)],
    )[:, -1, :].detach()
    down_acts = captured["downstream"].to(downstream_sae_device)
    down_z = downstream_sae.encode(down_acts).detach().float()
    return logits, down_z

@torch.no_grad()
def evaluate_feature_fresh(feature_id):
    n = fresh_tokens.shape[0]
    vocab = clean_logits_cpu.shape[1]
    sum_delta = torch.zeros(vocab, device=DEVICE, dtype=torch.float32)
    effect_l2_sum = 0.0
    effect_linf_sum = 0.0
    kl_sum = 0.0
    downstream_count_sum = 0.0

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        toks_b = fresh_tokens[start:end]
        clean_logits_b = clean_logits_cpu[start:end].to(DEVICE).float()
        clean_down_panel_b = clean_downstream_z_cpu[start:end][:, panel].to(DEVICE).float()
        steered_logits_b, steered_down_z_b = run_steered_batch(toks_b, feature_id)
        steered_logits_b = steered_logits_b.float()
        delta = steered_logits_b - clean_logits_b
        sum_delta += delta.sum(dim=0)
        effect_l2_sum += delta.norm(dim=-1).sum().item()
        effect_linf_sum += delta.abs().max(dim=-1).values.sum().item()
        logp = F.log_softmax(clean_logits_b, dim=-1)
        logq = F.log_softmax(steered_logits_b, dim=-1)
        p = logp.exp()
        kl_sum += (p * (logp - logq)).sum(dim=-1).sum().item()
        steered_panel_b = steered_down_z_b[:, panel].to(DEVICE).float()
        dz = steered_panel_b - clean_down_panel_b
        downstream_count_sum += (dz.abs() > PHASE4_CONFIG["DOWNSTREAM_DELTA_THRESH"]).sum(dim=1).sum().item()

    mean_delta = sum_delta / n
    mean_delta_norm = mean_delta.norm() + 1e-8
    signed_cos_sum = 0.0
    abs_cos_sum = 0.0

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        toks_b = fresh_tokens[start:end]
        clean_logits_b = clean_logits_cpu[start:end].to(DEVICE).float()
        steered_logits_b, _ = run_steered_batch(toks_b, feature_id)
        steered_logits_b = steered_logits_b.float()
        delta = steered_logits_b - clean_logits_b
        delta_norm = delta.norm(dim=-1) + 1e-8
        cos = (delta @ mean_delta) / (delta_norm * mean_delta_norm)
        signed_cos_sum += cos.sum().item()
        abs_cos_sum += cos.abs().sum().item()

    effect_l2_mean = effect_l2_sum / n
    downstream_count_mean = downstream_count_sum / n
    kl_mean = kl_sum / n
    return {
        "feature": int(feature_id),
        "fresh_n_contexts": int(n),
        "stability_to_mean_signed_cos": signed_cos_sum / n,
        "stability_to_mean_abs_cos": abs_cos_sum / n,
        "effect_l2_mean": effect_l2_mean,
        "effect_linf_mean": effect_linf_sum / n,
        "downstream_feat_count_abs_delta_gt_0.05": downstream_count_mean,
        "downstream_count_0.05_per_effect_l2": downstream_count_mean / (effect_l2_mean + 1e-8),
        "kl_clean_to_steered_mean": kl_mean,
        "kl_clean_to_steered_per_effect_l2": kl_mean / (effect_l2_mean + 1e-8),
    }

unique_features = sorted(selected_features["feature"].astype(int).unique().tolist())
print("Unique selected features to evaluate:", len(unique_features))
fresh_rows = []
for f in tqdm(unique_features, desc="fresh steering eval"):
    fresh_rows.append(evaluate_feature_fresh(f))
    if len(fresh_rows) % 10 == 0:
        pd.DataFrame(fresh_rows).to_csv(PHASE4_DIR / "llama31_phase4_fresh_steering_eval_partial.csv", index=False)

fresh_eval = pd.DataFrame(fresh_rows)
fresh_eval_path = PHASE4_DIR / "llama31_phase4_fresh_steering_eval.csv"
fresh_eval.to_csv(fresh_eval_path, index=False)
print("Saved fresh eval:", fresh_eval_path)
display(fresh_eval.head())


################################################################################
# Cell 5
################################################################################
# Cell 5 — Group tests and verdicts

eval_long = selected_features.merge(fresh_eval, on="feature", how="left")
eval_long_path = PHASE4_DIR / "llama31_phase4_selected_with_fresh_eval.csv"
eval_long.to_csv(eval_long_path, index=False)
print("Saved selected + eval:", eval_long_path)

TEST_METRICS = [
    "stability_to_mean_abs_cos",
    "stability_to_mean_signed_cos",
    "downstream_count_0.05_per_effect_l2",
    "effect_l2_mean",
    "kl_clean_to_steered_per_effect_l2",
]

def group_test(df, selector, metric, group_a, group_b):
    a = df[(df["selector"] == selector) & (df["selection_group"] == group_a)][metric].dropna().values
    b = df[(df["selector"] == selector) & (df["selection_group"] == group_b)][metric].dropna().values
    if len(a) == 0 or len(b) == 0:
        stat, p = np.nan, np.nan
    else:
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "selector": selector,
        "metric": metric,
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "mean_a": float(np.mean(a)) if len(a) else np.nan,
        "mean_b": float(np.mean(b)) if len(b) else np.nan,
        "median_a": float(np.median(a)) if len(a) else np.nan,
        "median_b": float(np.median(b)) if len(b) else np.nan,
        "diff_mean_a_minus_b": float(np.mean(a) - np.mean(b)) if len(a) and len(b) else np.nan,
        "mannwhitney_u": float(stat) if not np.isnan(stat) else np.nan,
        "mannwhitney_p": float(p) if not np.isnan(p) else np.nan,
    }

test_rows = []
for selector in SELECTORS:
    for metric in TEST_METRICS:
        test_rows.append(group_test(eval_long, selector, metric, "predicted_clean", "predicted_messy"))
        test_rows.append(group_test(eval_long, selector, metric, "predicted_clean", "random_control"))
        test_rows.append(group_test(eval_long, selector, metric, "predicted_messy", "random_control"))

tests_df = pd.DataFrame(test_rows)
tests_path = PHASE4_DIR / "llama31_phase4_group_tests.csv"
tests_df.to_csv(tests_path, index=False)
print("Saved tests:", tests_path)
display(tests_df.head(20))

def selector_summary(selector):
    sub = eval_long[eval_long["selector"] == selector].copy()
    clean = sub[sub["selection_group"] == "predicted_clean"]
    messy = sub[sub["selection_group"] == "predicted_messy"]
    def mean_metric(g, m):
        return float(g[m].mean())
    stab_clean = mean_metric(clean, "stability_to_mean_abs_cos")
    stab_messy = mean_metric(messy, "stability_to_mean_abs_cos")
    coll_clean = mean_metric(clean, "downstream_count_0.05_per_effect_l2")
    coll_messy = mean_metric(messy, "downstream_count_0.05_per_effect_l2")
    eff_clean = mean_metric(clean, "effect_l2_mean")
    eff_messy = mean_metric(messy, "effect_l2_mean")
    stab_test = tests_df[(tests_df["selector"] == selector) & (tests_df["metric"] == "stability_to_mean_abs_cos") & (tests_df["group_a"] == "predicted_clean") & (tests_df["group_b"] == "predicted_messy")].iloc[0]
    coll_test = tests_df[(tests_df["selector"] == selector) & (tests_df["metric"] == "downstream_count_0.05_per_effect_l2") & (tests_df["group_a"] == "predicted_clean") & (tests_df["group_b"] == "predicted_messy")].iloc[0]
    effect_test = tests_df[(tests_df["selector"] == selector) & (tests_df["metric"] == "effect_l2_mean") & (tests_df["group_a"] == "predicted_clean") & (tests_df["group_b"] == "predicted_messy")].iloc[0]
    stability_good = stab_clean > stab_messy and float(stab_test["mannwhitney_p"]) < 0.05
    collateral_good = coll_clean < coll_messy and float(coll_test["mannwhitney_p"]) < 0.05
    if stability_good and collateral_good:
        verdict = "strong"
    elif stability_good:
        verdict = "partial — stability only"
    elif collateral_good:
        verdict = "partial — collateral only"
    else:
        verdict = "weak"
    return {
        "model": "Llama-3.1-8B",
        "selector": selector,
        "n_per_group": int(clean.shape[0]),
        "fresh_contexts": int(PHASE4_CONFIG["FRESH_N_CONTEXTS"]),
        "clean_stability": stab_clean,
        "messy_stability": stab_messy,
        "p_stability": float(stab_test["mannwhitney_p"]),
        "clean_collateral_per_effect": coll_clean,
        "messy_collateral_per_effect": coll_messy,
        "p_collateral": float(coll_test["mannwhitney_p"]),
        "clean_effect_l2": eff_clean,
        "messy_effect_l2": eff_messy,
        "p_effect_l2": float(effect_test["mannwhitney_p"]),
        "verdict": verdict,
    }

summary_rows = [selector_summary(selector) for selector in SELECTORS]
summary_df = pd.DataFrame(summary_rows)
summary_csv = PHASE4_DIR / "llama31_phase4_screening_summary.csv"
summary_json = PHASE4_DIR / "llama31_phase4_verdict.json"
summary_df.to_csv(summary_csv, index=False)
with open(summary_json, "w") as f:
    json.dump(summary_rows, f, indent=2)
print("Saved summary:", summary_csv)
print("Saved verdict:", summary_json)
display(summary_df)
