#!/usr/bin/env python
# Converted from notebooks/figures/figures_3_4_5_with_llama.ipynb
# Original notebook removed from the repository; this script preserves the code cells for reproducibility.


################################################################################
# Cell 1
################################################################################
# Cell 1 — mount Drive and imports

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = Path('/content/drive/MyDrive/SAE Prediction')
FIG_DIR = BASE / 'paper_figures_with_llama'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    'GPT-2-small': BASE / 'robust_phase123_outputs',
    'Pythia-70M': BASE / 'pythia70m_phase123_outputs',
    'Gemma-2-2B': BASE / 'gemma_fullscale_phase123_outputs',
    'Llama-3.1-8B': BASE / 'llama31_8b_phase123_outputs',
}

PHASE4_DIRS = {
    'GPT-2-small': BASE / 'gpt2_phase4_full_outputs',
    'Pythia-70M': BASE / 'pythia70m_phase4_outputs',
    'Gemma-2-2B': BASE / 'gemma_phase4_both_predictors_outputs',
    'Llama-3.1-8B': BASE / 'llama31_8b_phase123_outputs',
}

MODEL_ORDER = ['GPT-2-small', 'Pythia-70M', 'Gemma-2-2B', 'Llama-3.1-8B']

print('Saving figures to:', FIG_DIR)
print('\nPhase 1/2/3 folders:')
for name, folder in MODEL_DIRS.items():
    print(f'{name:14s} -> {folder} | exists={folder.exists()}')

print('\nPhase 4 folders:')
for name, folder in PHASE4_DIRS.items():
    print(f'{name:14s} -> {folder} | exists={folder.exists()}')


################################################################################
# Cell 2
################################################################################
# Cell 2 — helper functions

def find_file(folder, patterns, recursive=False):
    folder = Path(folder)
    if not folder.exists():
        return None
    globber = folder.rglob if recursive else folder.glob
    for pattern in patterns:
        matches = sorted(globber(pattern))
        if matches:
            return matches[0]
    return None


def load_csv(folder, patterns, required=True, recursive=False):
    p = find_file(folder, patterns, recursive=recursive)
    if p is None:
        if required:
            raise FileNotFoundError(f'Missing CSV in {folder}: {patterns}')
        return None
    print('Loaded CSV:', p)
    return pd.read_csv(p)


def load_json_file(path):
    with open(path, 'r') as f:
        return json.load(f)


def choose_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f'None of these columns found: {candidates}')


def short_target_name(t):
    return (
        t.replace('__fixed_global_add__mixed', '')
         .replace('downstream_feat_count_abs_delta_gt_0.05', 'collateral count')
         .replace('downstream_feat_count_abs_delta_gt_0_05', 'collateral count')
         .replace('downstream_count_0.05_per_effect_l2', 'collateral/effect')
         .replace('downstream_count_0_05_per_effect_l2', 'collateral/effect')
         .replace('stability_to_mean_signed_cos', 'signed stability')
         .replace('stability_to_mean_abs_cos', 'abs stability')
         .replace('effect_l2_mean', 'effect L2')
    )


def check_required_files():
    print('Checking key Phase 1/2/3 files...')
    for model, folder in MODEL_DIRS.items():
        b = find_file(folder, ['robust_phase3_baseline_comparison.csv', '*baseline_comparison*.csv'])
        m = find_file(folder, ['robust_phase2_phase3_merged.csv', '*phase2_phase3_merged*.csv'])
        print(f'{model:14s} baseline={b is not None} merged={m is not None}')
    print('\nChecking Phase 4 folders...')
    for model, folder in PHASE4_DIRS.items():
        files = list(Path(folder).glob('*')) if Path(folder).exists() else []
        print(f'{model:14s} n_files={len(files)}')

check_required_files()


################################################################################
# Cell 3
################################################################################
# Cell 3 — Figure 3: predictive improvement including Llama

fig3_rows = []

for model_name in MODEL_ORDER:
    folder = MODEL_DIRS[model_name]
    baseline_df = load_csv(
        folder,
        ['robust_phase3_baseline_comparison.csv', 'robust_phase3_baseline_comparison*.csv', '*baseline_comparison*.csv'],
        required=True,
    )

    fig3_rows.append({
        'model': model_name,
        'mean_no_mag_minus_freq': baseline_df['full_no_magnitude_minus_freq'].mean(),
        'mean_no_mag_minus_actmag': baseline_df['full_no_magnitude_minus_activation_magnitude'].mean(),
    })

fig3_df = pd.DataFrame(fig3_rows)
fig3_df['model'] = pd.Categorical(fig3_df['model'], categories=MODEL_ORDER, ordered=True)
fig3_df = fig3_df.sort_values('model')

display(fig3_df)

x = np.arange(len(fig3_df))
width = 0.35

plt.figure(figsize=(7.2, 3.6))
plt.bar(x - width/2, fig3_df['mean_no_mag_minus_freq'], width, label='vs frequency-only')
plt.bar(x + width/2, fig3_df['mean_no_mag_minus_actmag'], width, label='vs activation-magnitude-only')
plt.axhline(0, linewidth=0.8)
plt.xticks(x, fig3_df['model'], rotation=15, ha='right')
plt.ylabel('Mean CV Spearman improvement')
plt.xlabel('Model / SAE setting')
plt.title('Predictive improvement of no-magnitude feature statistics')
plt.legend(frameon=False)
plt.tight_layout()

out_png = FIG_DIR / 'figure3_predictive_improvement_with_llama.png'
out_pdf = FIG_DIR / 'figure3_predictive_improvement_with_llama.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.show()

print('Saved:', out_png)
print('Saved:', out_pdf)


################################################################################
# Cell 4
################################################################################
# Cell 4 — Figure 4: representative univariate relationships including Llama

REPRESENTATIVE_RELATIONSHIPS = {
    'GPT-2-small': {
        'predictor': 'crowding_topk_mean_abs_cos',
        'target_candidates': [
            'downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed',
            'downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed',
        ],
        'display': 'crowding → collateral',
    },
    'Pythia-70M': {
        'predictor': 'direct_logit_l2',
        'target_candidates': [
            'downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed',
            'downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed',
        ],
        'display': 'direct-logit $L_2$ → collateral',
    },
    'Gemma-2-2B': {
        'predictor': 'encoder_norm',
        'target_candidates': [
            'stability_to_mean_signed_cos__fixed_global_add__mixed',
        ],
        'display': 'encoder norm → signed stability',
    },
    'Llama-3.1-8B': {
        'predictor': 'direct_logit_l2',
        'target_candidates': [
            'downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed',
            'downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed',
        ],
        'display': 'direct-logit $L_2$ → collateral',
    },
}

fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
axes = axes.flatten()

for ax, model_name in zip(axes, MODEL_ORDER):
    folder = MODEL_DIRS[model_name]
    merged_df = load_csv(
        folder,
        ['robust_phase2_phase3_merged.csv', '*phase2_phase3_merged*.csv'],
        required=True,
    )

    spec = REPRESENTATIVE_RELATIONSHIPS[model_name]
    pred_col = spec['predictor']
    target_col = choose_col(merged_df, spec['target_candidates'])

    sub = merged_df[[pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
    xvals = sub[pred_col].values
    yvals = sub[target_col].values
    rho, p = spearmanr(xvals, yvals)

    ax.scatter(xvals, yvals, s=14, alpha=0.65)

    if len(sub) > 2 and np.std(xvals) > 0:
        m, b = np.polyfit(xvals, yvals, 1)
        xs = np.linspace(np.min(xvals), np.max(xvals), 100)
        ax.plot(xs, m * xs + b, linewidth=1.2)

    ax.set_title(f'{model_name}\n{spec["display"]}', fontsize=10)
    ax.set_xlabel(pred_col.replace('_', ' '), fontsize=8)
    ax.set_ylabel(short_target_name(target_col), fontsize=8)
    ax.text(
        0.04, 0.94,
        rf'$\rho={rho:.3f}$' + '\n' + rf'$p={p:.1e}$',
        transform=ax.transAxes,
        va='top',
        fontsize=8,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.75')
    )

plt.tight_layout()

out_png = FIG_DIR / 'figure4_representative_univariate_with_llama.png'
out_pdf = FIG_DIR / 'figure4_representative_univariate_with_llama.pdf'
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
plt.show()

print('Saved:', out_png)
print('Saved:', out_pdf)


################################################################################
# Cell 5
################################################################################
# Cell 5 — Load per-feature Phase 4 data for boxplot Figure 5
# Fixed for selection_group_x / selection_group_y and Llama selected+fresh merge

import json
import numpy as np
import pandas as pd
from pathlib import Path

def find_first(folder, patterns):
    folder = Path(folder)
    for pat in patterns:
        matches = sorted(folder.glob(pat))
        if matches:
            return matches[0]
    return None

def normalize_group_name(x):
    x = str(x).lower()
    if "clean" in x:
        return "Predicted-clean"
    if "messy" in x:
        return "Predicted-messy"
    if "random" in x:
        return "Random"
    return x

def pick_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_per_feature_phase4(model_name, folder, preferred_selector=None):
    folder = Path(folder)
    print(f"\nLoading per-feature Phase 4 for {model_name}: {folder}")

    # Always merge selected_features + fresh_steering_eval.
    # This avoids bad selected_with_fresh_eval files that lost group labels.
    selected_file = find_first(folder, [
        "*phase4_selected_features*.csv",
        "*selected_features*.csv",
    ])

    fresh_file = find_first(folder, [
        "*phase4_fresh_steering_eval*.csv",
        "*fresh_steering_eval*.csv",
        "*fresh_eval*.csv",
    ])

    if selected_file is None or fresh_file is None:
        print("  WARNING: missing selected/fresh files.")
        print("  Files in folder:")
        for p in sorted(folder.glob("*")):
            print("   ", p.name)
        return pd.DataFrame()

    print("  selected:", selected_file.name)
    print("  fresh:", fresh_file.name)

    selected = pd.read_csv(selected_file)
    fresh = pd.read_csv(fresh_file)

    if "feature" not in selected.columns or "feature" not in fresh.columns:
        print("  WARNING: no feature column found.")
        return pd.DataFrame()

    df = selected.merge(fresh, on="feature", how="left")
    df["model"] = model_name

    # Selector column can appear under different names after merge
    selector_col = pick_column(df, [
        "selector",
        "predictor_set",
        "predictor_set_x",
        "predictor_set_y",
    ])

    if selector_col is None:
        df["selector"] = "unknown"
    else:
        df["selector"] = df[selector_col].astype(str)

    # Keep preferred selector if available
    if preferred_selector is not None:
        preferred = df[df["selector"] == preferred_selector].copy()
        if len(preferred) > 0:
            df = preferred
        else:
            print(f"  WARNING: preferred selector {preferred_selector} not found. Available:", sorted(df["selector"].unique()))

    # Group column can appear under different names after merge
    group_col = pick_column(df, [
        "selection_group",
        "selection_group_x",
        "selection_group_y",
        "group",
        "group_x",
        "group_y",
        "phase4_group",
        "phase4_group_x",
        "phase4_group_y",
    ])

    if group_col is None:
        print("  WARNING: no selection group column found.")
        print("  columns:", list(df.columns))
        return pd.DataFrame()

    df["group_display"] = df[group_col].map(normalize_group_name)

    # Normalize metric columns
    stability_col = pick_column(df, [
        "stability_to_mean_abs_cos",
        "stability_to_mean_abs_cos_y",
        "stability_to_mean_abs_cos_x",
        "stability_to_mean_abs_cos_fresh",
        "fresh_stability_abs",
    ])

    collateral_col = pick_column(df, [
        "downstream_count_0.05_per_effect_l2",
        "downstream_count_0.05_per_effect_l2_y",
        "downstream_count_0.05_per_effect_l2_x",
        "downstream_count_0_05_per_effect_l2",
        "downstream_count_0_05_per_effect_l2_y",
        "downstream_count_0_05_per_effect_l2_x",
        "collateral_per_effect",
    ])

    if stability_col is None:
        candidates = [c for c in df.columns if "stability" in c and "abs" in c]
        stability_col = candidates[0] if candidates else None

    if collateral_col is None:
        candidates = [
            c for c in df.columns
            if ("per_effect" in c or "per-effect" in c)
            and ("collateral" in c or "downstream_count" in c)
        ]
        collateral_col = candidates[0] if candidates else None

    if stability_col is None or collateral_col is None:
        print("  WARNING: missing metric columns.")
        print("  stability_col:", stability_col)
        print("  collateral_col:", collateral_col)
        print("  columns:", list(df.columns))
        return pd.DataFrame()

    df["stability"] = pd.to_numeric(df[stability_col], errors="coerce")
    df["collateral_per_effect"] = pd.to_numeric(df[collateral_col], errors="coerce")

    needed = ["model", "selector", "feature", "group_display", "stability", "collateral_per_effect"]
    out = df[needed].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["stability", "collateral_per_effect"])

    # Keep only the three intended groups
    out = out[out["group_display"].isin(["Predicted-clean", "Random", "Predicted-messy"])].copy()

    print("  loaded rows:", len(out))
    print("  selector:", out["selector"].unique())
    print("  groups:", out["group_display"].value_counts().to_dict())

    return out


MAIN_SELECTORS = {
    "GPT-2-small": "geometry_only",
    "Pythia-70M": "full_no_magnitude",
    "Gemma-2-2B": "full_no_magnitude",
    "Llama-3.1-8B": "full_no_magnitude",
}

phase4_box_parts = []

for model_name in MODEL_ORDER:
    df = load_per_feature_phase4(
        model_name,
        PHASE4_DIRS[model_name],
        preferred_selector=MAIN_SELECTORS.get(model_name)
    )
    if len(df) > 0:
        phase4_box_parts.append(df)

if len(phase4_box_parts) == 0:
    raise ValueError("No Phase 4 per-feature data loaded. Check folder paths and file names.")

phase4_box_df = pd.concat(phase4_box_parts, ignore_index=True)

print("\nFinal per-feature Phase 4 dataframe:", phase4_box_df.shape)
display(phase4_box_df.head())
display(phase4_box_df.groupby(["model", "selector", "group_display"]).size())


################################################################################
# Cell 6
################################################################################
# Patch — force-add Llama Phase 4 per-feature rows before Figure 5

import pandas as pd
import numpy as np
from pathlib import Path

llama_folder = PHASE4_DIRS["Llama-3.1-8B"]

llama_selected_path = llama_folder / "llama31_phase4_selected_features.csv"
llama_fresh_path = llama_folder / "llama31_phase4_fresh_steering_eval.csv"

print("Llama selected exists:", llama_selected_path.exists(), llama_selected_path)
print("Llama fresh exists:", llama_fresh_path.exists(), llama_fresh_path)

llama_selected = pd.read_csv(llama_selected_path)
llama_fresh = pd.read_csv(llama_fresh_path)

llama_df = llama_selected.merge(llama_fresh, on="feature", how="left")

# Keep the main selector used in the paper
llama_df = llama_df[llama_df["selector"] == "full_no_magnitude"].copy()

def normalize_group_name(x):
    x = str(x).lower()
    if "clean" in x:
        return "Predicted-clean"
    if "messy" in x:
        return "Predicted-messy"
    if "random" in x:
        return "Random"
    return x

llama_plot_df = pd.DataFrame({
    "model": "Llama-3.1-8B",
    "selector": "full_no_magnitude",
    "feature": llama_df["feature"].astype(int),
    "group_display": llama_df["selection_group"].map(normalize_group_name),
    "stability": pd.to_numeric(llama_df["stability_to_mean_abs_cos"], errors="coerce"),
    "collateral_per_effect": pd.to_numeric(llama_df["downstream_count_0.05_per_effect_l2"], errors="coerce"),
})

llama_plot_df = llama_plot_df.replace([np.inf, -np.inf], np.nan)
llama_plot_df = llama_plot_df.dropna(subset=["stability", "collateral_per_effect"])

print("Loaded Llama rows:", llama_plot_df.shape)
print(llama_plot_df.groupby("group_display").size())
print(llama_plot_df.groupby("group_display")[["stability", "collateral_per_effect"]].mean())

# Remove any old/empty Llama rows, then append correct Llama rows
phase4_box_df = phase4_box_df[phase4_box_df["model"] != "Llama-3.1-8B"].copy()
phase4_box_df = pd.concat([phase4_box_df, llama_plot_df], ignore_index=True)

print("\nUpdated phase4_box_df:")
display(phase4_box_df.groupby(["model", "group_display"]).size())


################################################################################
# Cell 7
################################################################################
# Patch — reconstruct Llama Phase 4 selected groups from phase123 merged file
# Run after Cell 5 and before Cell 6.

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

llama_folder = Path("/content/drive/MyDrive/SAE Prediction/llama31_8b_phase123_outputs")

llama_merged_path = llama_folder / "robust_phase2_phase3_merged.csv"
llama_fresh_path = llama_folder / "llama31_phase4_fresh_steering_eval.csv"

print("Llama merged exists:", llama_merged_path.exists(), llama_merged_path)
print("Llama fresh exists:", llama_fresh_path.exists(), llama_fresh_path)

merged = pd.read_csv(llama_merged_path)
fresh = pd.read_csv(llama_fresh_path)

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None found: {candidates}")

ABS_STAB_COL = find_col(merged, [
    "stability_to_mean_abs_cos__fixed_global_add__mixed",
])

COLLATERAL_COL = find_col(merged, [
    "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
    "downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed",
])

EFFECT_L2_COL = find_col(merged, [
    "effect_l2_mean__fixed_global_add__mixed",
])

EPS = 1e-8

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) + EPS)

merged = merged.copy()
merged["phase4_collateral_per_effect"] = merged[COLLATERAL_COL] / (merged[EFFECT_L2_COL] + EPS)
merged["phase4_clean_score"] = (
    zscore(merged[ABS_STAB_COL])
    - zscore(np.log1p(merged["phase4_collateral_per_effect"]))
)

FULL_NO_MAG_COLS = [
    "phase1_final_act_freq",
    "token_binary_entropy",
    "token_activation_entropy_norm",
    "token_act_kurtosis",

    "decoder_norm",
    "encoder_norm",
    "encoder_decoder_cos",
    "crowding_topk_mean_abs_cos",
    "crowding_topk_sum_abs_cos",
    "crowding_max_abs_cos",

    "coact_count_mean",
    "coact_entropy_norm",
    "coact_top20_mass",

    "direct_logit_l2",
    "direct_logit_linf",
    "direct_logit_entropy",
    "direct_logit_top10_mass_frac",
    "direct_logit_top100_mass_frac",
]

FULL_NO_MAG_COLS = [c for c in FULL_NO_MAG_COLS if c in merged.columns]
print("Using full_no_magnitude predictors:", len(FULL_NO_MAG_COLS), FULL_NO_MAG_COLS)

needed = ["feature", "phase4_clean_score", EFFECT_L2_COL] + FULL_NO_MAG_COLS
phase4_df = merged[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
phase4_df["feature"] = phase4_df["feature"].astype(int)

train_idx, pool_idx = train_test_split(
    np.arange(len(phase4_df)),
    train_size=0.70,
    random_state=1,
    shuffle=True,
)

train_df = phase4_df.iloc[train_idx].copy()
pool_df = phase4_df.iloc[pool_idx].copy()

clean_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
clean_model.fit(train_df[FULL_NO_MAG_COLS], train_df["phase4_clean_score"])

effect_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)
effect_model.fit(train_df[FULL_NO_MAG_COLS], train_df[EFFECT_L2_COL])

pred_df = pool_df.copy()
pred_df["selector"] = "full_no_magnitude"
pred_df["pred_clean_score"] = clean_model.predict(pred_df[FULL_NO_MAG_COLS])
pred_df["pred_effect_l2"] = effect_model.predict(pred_df[FULL_NO_MAG_COLS])

def select_groups(pred_df, group_size=25):
    df = pred_df.copy()

    lo, hi = df["pred_effect_l2"].quantile([0.10, 0.90])
    effect_pool = df[(df["pred_effect_l2"] >= lo) & (df["pred_effect_l2"] <= hi)].copy()

    if len(effect_pool) < group_size * 3:
        lo, hi = df["pred_effect_l2"].quantile([0.05, 0.95])
        effect_pool = df[(df["pred_effect_l2"] >= lo) & (df["pred_effect_l2"] <= hi)].copy()

    if len(effect_pool) < group_size * 3:
        effect_pool = df.copy()

    effect_pool = effect_pool.sort_values("pred_clean_score", ascending=False)

    clean = effect_pool.head(group_size).copy()
    clean["selection_group"] = "predicted_clean"

    messy = effect_pool.tail(group_size).copy()
    messy["selection_group"] = "predicted_messy"

    used = set(clean["feature"]) | set(messy["feature"])
    remaining = effect_pool[~effect_pool["feature"].isin(used)].copy()

    random_control = remaining.sample(
        n=group_size,
        random_state=1
    ).copy()
    random_control["selection_group"] = "random_control"

    return pd.concat([clean, random_control, messy], ignore_index=True)

llama_selected = select_groups(pred_df, group_size=25)

print("Reconstructed Llama selected groups:")
print(llama_selected["selection_group"].value_counts())

llama_df = llama_selected.merge(fresh, on="feature", how="left")

def normalize_group_name(x):
    x = str(x).lower()
    if "clean" in x:
        return "Predicted-clean"
    if "messy" in x:
        return "Predicted-messy"
    if "random" in x:
        return "Random"
    return x

llama_plot_df = pd.DataFrame({
    "model": "Llama-3.1-8B",
    "selector": "full_no_magnitude",
    "feature": llama_df["feature"].astype(int),
    "group_display": llama_df["selection_group"].map(normalize_group_name),
    "stability": pd.to_numeric(llama_df["stability_to_mean_abs_cos"], errors="coerce"),
    "collateral_per_effect": pd.to_numeric(llama_df["downstream_count_0.05_per_effect_l2"], errors="coerce"),
})

llama_plot_df = llama_plot_df.replace([np.inf, -np.inf], np.nan)
llama_plot_df = llama_plot_df.dropna(subset=["stability", "collateral_per_effect"])

print("\nLoaded reconstructed Llama rows:", llama_plot_df.shape)
print(llama_plot_df.groupby("group_display").size())
print(llama_plot_df.groupby("group_display")[["stability", "collateral_per_effect"]].mean())

# Remove any old/empty Llama rows, then append correct Llama rows
phase4_box_df = phase4_box_df[phase4_box_df["model"] != "Llama-3.1-8B"].copy()
phase4_box_df = pd.concat([phase4_box_df, llama_plot_df], ignore_index=True)

print("\nUpdated phase4_box_df:")
display(phase4_box_df.groupby(["model", "group_display"]).size())


################################################################################
# Cell 8
################################################################################
# Cell 6 — Figure 5: held-out screening boxplots including Llama

import matplotlib.pyplot as plt
import numpy as np

GROUP_ORDER = ["Predicted-clean", "Random", "Predicted-messy"]
MODEL_ORDER = ["GPT-2-small", "Pythia-70M", "Gemma-2-2B", "Llama-3.1-8B"]

fig, axes = plt.subplots(
    2,
    len(MODEL_ORDER),
    figsize=(12.5, 5.8),
    sharex=False
)

for col, model_name in enumerate(MODEL_ORDER):
    sub = phase4_box_df[phase4_box_df["model"] == model_name].copy()

    # Top row: stability
    ax = axes[0, col]
    data = [
        sub[sub["group_display"] == g]["stability"].dropna().values
        for g in GROUP_ORDER
    ]

    ax.boxplot(
        data,
        labels=["Clean", "Random", "Messy"],
        showfliers=True,
        widths=0.55
    )
    ax.set_title(model_name)
    ax.set_ylabel("Stability\n(abs. cosine)" if col == 0 else "")
    ax.tick_params(axis="x", rotation=25)

    # Bottom row: collateral per effect
    ax = axes[1, col]
    data = [
        sub[sub["group_display"] == g]["collateral_per_effect"].dropna().values
        for g in GROUP_ORDER
    ]

    ax.boxplot(
        data,
        labels=["Clean", "Random", "Messy"],
        showfliers=True,
        widths=0.55
    )
    ax.set_ylabel("Collateral\nper effect" if col == 0 else "")
    ax.tick_params(axis="x", rotation=25)

axes[0, 0].text(
    -0.35,
    1.08,
    "A",
    transform=axes[0, 0].transAxes,
    fontsize=14,
    fontweight="bold"
)

axes[1, 0].text(
    -0.35,
    1.08,
    "B",
    transform=axes[1, 0].transAxes,
    fontsize=14,
    fontweight="bold"
)

fig.suptitle(
    "Held-out screening separates cleaner features on fresh contexts",
    y=1.02,
    fontsize=14
)

plt.tight_layout()

out_png = FIG_DIR / "figure5_heldout_screening_boxplots_with_llama.png"
out_pdf = FIG_DIR / "figure5_heldout_screening_boxplots_with_llama.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print("Saved:", out_png)
print("Saved:", out_pdf)


################################################################################
# Cell 9
################################################################################
# Cell — Figure 3: predictive improvement including Llama, fixed legend

# Rebuild Figure 3 dataframe in case it is not already in memory
fig3_rows = []

for model_name, folder in MODEL_DIRS.items():
    baseline_df = load_csv(
        folder,
        ["robust_phase3_baseline_comparison.csv", "robust_phase3_baseline_comparison*.csv"],
        required=True
    )

    fig3_rows.append({
        "model": model_name,
        "mean_no_mag_minus_freq": baseline_df["full_no_magnitude_minus_freq"].mean(),
        "mean_no_mag_minus_actmag": baseline_df["full_no_magnitude_minus_activation_magnitude"].mean(),
    })

fig3_df = pd.DataFrame(fig3_rows)

fig3_df["model"] = pd.Categorical(
    fig3_df["model"],
    categories=MODEL_ORDER,
    ordered=True
)
fig3_df = fig3_df.sort_values("model")

print(fig3_df)

# Plot
fig, ax = plt.subplots(figsize=(10.5, 4.8))

x = np.arange(len(fig3_df))
width = 0.34

ax.bar(
    x - width / 2,
    fig3_df["mean_no_mag_minus_freq"],
    width,
    label="vs frequency-only"
)

ax.bar(
    x + width / 2,
    fig3_df["mean_no_mag_minus_actmag"],
    width,
    label="vs activation-magnitude-only"
)

ax.axhline(0, linewidth=0.8)

ax.set_title(
    "Predictive improvement of no-magnitude feature statistics",
    pad=34
)
ax.set_ylabel("Mean CV Spearman improvement")
ax.set_xlabel("Model / SAE setting")

ax.set_xticks(x)
ax.set_xticklabels(fig3_df["model"], rotation=15, ha="right")

# Legend outside plot area, above axes
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    frameon=False,
    fontsize=10
)

# Extra top room for legend
plt.tight_layout(rect=[0, 0, 1, 0.88])

out_png = FIG_DIR / "figure3_predictive_improvement_with_llama.png"
out_pdf = FIG_DIR / "figure3_predictive_improvement_with_llama.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print("Saved:", out_png)
print("Saved:", out_pdf)


################################################################################
# Cell 10
################################################################################
# Cell — Figure 4: representative univariate relationships in one horizontal row

REPRESENTATIVE_RELATIONSHIPS = {
    "GPT-2-small": {
        "predictor": "crowding_topk_mean_abs_cos",
        "target_candidates": [
            "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
            "downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed",
        ],
        "display": "crowding → collateral",
    },
    "Pythia-70M": {
        "predictor": "direct_logit_l2",
        "target_candidates": [
            "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
            "downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed",
        ],
        "display": "direct-logit $L_2$ → collateral",
    },
    "Gemma-2-2B": {
        "predictor": "encoder_norm",
        "target_candidates": [
            "stability_to_mean_signed_cos__fixed_global_add__mixed",
        ],
        "display": "encoder norm → stability",
    },
    "Llama-3.1-8B": {
        "predictor": "direct_logit_l2",
        "target_candidates": [
            "downstream_feat_count_abs_delta_gt_0.05__fixed_global_add__mixed",
            "downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed",
        ],
        "display": "direct-logit $L_2$ → collateral",
    },
}

fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.4))
axes = axes.flatten()

for ax, model_name in zip(axes, MODEL_ORDER):
    folder = MODEL_DIRS[model_name]

    merged_df = load_csv(
        folder,
        ["robust_phase2_phase3_merged.csv", "*phase2_phase3_merged*.csv"],
        required=True
    )

    spec = REPRESENTATIVE_RELATIONSHIPS[model_name]
    pred_col = spec["predictor"]
    target_col = choose_col(merged_df, spec["target_candidates"])

    sub = merged_df[[pred_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

    xvals = sub[pred_col].values
    yvals = sub[target_col].values

    rho, p = spearmanr(xvals, yvals)

    ax.scatter(xvals, yvals, s=12, alpha=0.65)

    # Linear trend for visualization only
    if len(sub) > 2:
        m, b = np.polyfit(xvals, yvals, 1)
        xs = np.linspace(np.min(xvals), np.max(xvals), 100)
        ax.plot(xs, m * xs + b, linewidth=1.2)

    ax.set_title(f"{model_name}\n{spec['display']}", fontsize=10)
    ax.set_xlabel(pred_col.replace("_", " "), fontsize=8)
    ax.set_ylabel(short_target_name(target_col), fontsize=8)

    ax.text(
        0.04,
        0.94,
        rf"$\rho={rho:.3f}$" + "\n" + rf"$p={p:.1e}$",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="0.75"
        )
    )

plt.tight_layout(w_pad=1.4)

out_png = FIG_DIR / "figure4_representative_univariate_with_llama_horizontal.png"
out_pdf = FIG_DIR / "figure4_representative_univariate_with_llama_horizontal.pdf"

plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()

print("Saved:", out_png)
print("Saved:", out_pdf)
