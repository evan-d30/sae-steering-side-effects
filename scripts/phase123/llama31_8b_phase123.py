#!/usr/bin/env python
# Converted from notebooks/phase123/llama31_8b_phase123.ipynb
# Original notebook removed from the repository; this script preserves the code cells for reproducibility.


################################################################################
# Cell 1
################################################################################
# Cell 1 — install dependencies
# NOTE: notebook-only command was: !pip -q install --upgrade "transformer-lens>=2.15.0" "sae-lens>=6.0.0" datasets accelerate bitsandbytes huggingface_hub einops tqdm scikit-learn scipy pandas pyarrow


################################################################################
# Cell 2
################################################################################
import os
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"


################################################################################
# Cell 3
################################################################################
# Hugging Face login cell
# Run this before loading gated models like Llama-3.1-8B.

from huggingface_hub import login
from getpass import getpass
import os

# Option 1: use environment variable if already set
hf_token = os.environ.get("HF_TOKEN", "")

# Option 2: securely paste token into notebook prompt
if not hf_token:
    hf_token = getpass("Paste your Hugging Face token: ")

login(token=hf_token, add_to_git_credential=False)

print("Hugging Face login complete.")


################################################################################
# Cell 4
################################################################################
# Put Hugging Face cache on persistent Vast.ai storage

import os
from pathlib import Path

HF_CACHE = Path("/workspace/hf_cache")
HF_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE)
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")

print("HF cache:", HF_CACHE)


################################################################################
# Cell 5
################################################################################
# Cell 2 — imports, Hugging Face login, and local Vast.ai configuration

from pathlib import Path
import os, json, math, random, gc, warnings, inspect
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score
from datasets import load_dataset
from huggingface_hub import login

from transformer_lens import HookedTransformer
from sae_lens import SAE

# Hugging Face login
# Option 1: set HF_TOKEN in the Vast.ai environment.
# Option 2: paste login(token="...") manually below.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print('No HF_TOKEN found. If model loading fails, run: login(token="YOUR_TOKEN")')

CONFIG = dict(
    MODEL_NAME='meta-llama/Llama-3.1-8B',
    MODEL_DTYPE='bfloat16',
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu',
    SEED=0,

    # Llama Scope residual SAE defaults. Edit if SAE discovery says release names differ.
    HOOK_NAME='blocks.16.hook_resid_post',
    DOWNSTREAM_HOOK_NAME='blocks.20.hook_resid_post',
    SAE_RELEASE='llama_scope_lxr_8x',
    SAE_ID='l16r_8x',
    DOWNSTREAM_SAE_RELEASE='llama_scope_lxr_8x',
    DOWNSTREAM_SAE_ID='l20r_8x',

    HF_DATASET_NAME='wikitext',
    HF_DATASET_CONFIG='wikitext-103-raw-v1',
    HF_DATASET_SPLIT='train[:2%]',
    MIN_DISTINCT_TEXTS=1000,
    MAX_DISTINCT_TEXTS=8000,
    CONTEXT_LEN=48,
    N_CONTEXTS=2048,
    N_FEATURES=300,
    CONTEXTS_PER_TYPE=16,

    BATCH_CONTEXTS=16,
    BATCH_STEER=16,

    ACTIVE_THRESH=1e-6,
    FIXED_GLOBAL_CLAMP_VALUE=1.0,
    PRIMARY_CLAMP_MODE='fixed_global_add',
    DOWNSTREAM_PANEL_SIZE=1024,
    DOWNSTREAM_DELTA_THRESH=0.05,

    N_SPLITS=5,
    RIDGE_ALPHA=1.0,
)

random.seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

# Local Vast.ai storage
# Use /workspace if available because it is usually persistent for the instance.
BASE = Path('/workspace/SAE_Prediction')

# Fallback if /workspace does not exist
if not Path('/workspace').exists():
    BASE = Path.cwd() / 'SAE_Prediction'

OUT_DIR = BASE / 'llama31_8b_phase123_outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('Output dir:', OUT_DIR)
print('Device:', CONFIG['DEVICE'])
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


################################################################################
# Cell 6
################################################################################
# Cell 3 — utility functions

def get_dtype(name: str):
    if name == 'bfloat16': return torch.bfloat16
    if name == 'float16': return torch.float16
    if name == 'float32': return torch.float32
    raise ValueError(name)

DTYPE = get_dtype(CONFIG['MODEL_DTYPE'])
DEVICE = CONFIG['DEVICE']

def sae_encode(acts: torch.Tensor, sae: SAE) -> torch.Tensor:
    # Robust SAE encoder wrapper across SAELens versions.
    with torch.no_grad():
        if hasattr(sae, 'encode'):
            return sae.encode(acts)
        if hasattr(sae, 'encode_standard'):
            return sae.encode_standard(acts)
        if hasattr(sae, 'forward'):
            out = sae(acts)
            if isinstance(out, tuple):
                for item in out:
                    if torch.is_tensor(item) and item.shape[:-1] == acts.shape[:-1]:
                        return item
            return out
    raise RuntimeError('Could not encode activations with this SAE object.')

def get_decoder_matrix(sae: SAE) -> torch.Tensor:
    for attr in ['W_dec', 'W_dec_normalized']:
        if hasattr(sae, attr):
            W = getattr(sae, attr)
            if torch.is_tensor(W):
                return W.detach()
    if hasattr(sae, 'state_dict'):
        for k, v in sae.state_dict().items():
            if 'W_dec' in k and torch.is_tensor(v):
                return v.detach()
    raise RuntimeError('Could not find decoder matrix on SAE.')

def get_encoder_matrix(sae: SAE) -> Optional[torch.Tensor]:
    if hasattr(sae, 'W_enc') and torch.is_tensor(sae.W_enc):
        return sae.W_enc.detach()
    if hasattr(sae, 'state_dict'):
        for k, v in sae.state_dict().items():
            if 'W_enc' in k and torch.is_tensor(v):
                return v.detach()
    return None

def ensure_dec_shape(W_dec: torch.Tensor, d_model: int) -> torch.Tensor:
    # Return W_dec as [n_features, d_model].
    if W_dec.shape[-1] == d_model:
        return W_dec
    if W_dec.shape[0] == d_model:
        return W_dec.T
    raise ValueError(f'Cannot infer decoder shape {tuple(W_dec.shape)} for d_model={d_model}')

def ensure_enc_shape(W_enc: Optional[torch.Tensor], d_model: int, n_features: int) -> Optional[torch.Tensor]:
    # Return W_enc as [n_features, d_model] if available.
    if W_enc is None:
        return None
    if W_enc.shape == (d_model, n_features):
        return W_enc.T
    if W_enc.shape == (n_features, d_model):
        return W_enc
    if W_enc.shape[-1] == d_model:
        return W_enc
    if W_enc.shape[0] == d_model:
        return W_enc.T
    print('Warning: cannot infer encoder shape:', W_enc.shape)
    return None

def clean_text(t: str) -> str:
    return ' '.join(str(t).split())

def safe_corr(fn, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return np.nan, np.nan
    r, p = fn(x[mask], y[mask])
    return float(r), float(p)

def cosine_rows_to_mean(X: torch.Tensor):
    mean = X.mean(dim=0, keepdim=True)
    cos = F.cosine_similarity(X, mean.expand_as(X), dim=-1, eps=1e-8)
    return cos.mean().item(), cos.abs().mean().item()


################################################################################
# Cell 7
################################################################################
# Cell 4 — discover available SAE releases and load Llama Scope SAEs

def print_saelens_directory_matches(patterns=('llama', 'llamascope', 'Llama3_1')):
    try:
        try:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        except Exception:
            from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
        directory = get_pretrained_saes_directory()
        print('SAELens pretrained directory entries containing:', patterns)
        n = 0
        for key, val in directory.items():
            text = str(key).lower() + ' ' + str(val).lower()
            if any(p.lower() in text for p in patterns):
                print('  ', key)
                n += 1
                if n >= 50:
                    print('  ... truncated')
                    break
        if n == 0:
            print('  No matching entries found in local SAELens directory.')
    except Exception as e:
        print('Could not inspect SAELens directory:', repr(e))

print_saelens_directory_matches()

SAE_CANDIDATES_PRIMARY = [
    (CONFIG['SAE_RELEASE'], CONFIG['SAE_ID']),
    ('llama_scope_lxr_8x', 'l16r_8x'),
    ('llama_scope_lxr_8x', 'l16r_8x'),
    ('llama_scope_lxr_8x', 'l16r_8x'),
]
SAE_CANDIDATES_DOWNSTREAM = [
    (CONFIG['DOWNSTREAM_SAE_RELEASE'], CONFIG['DOWNSTREAM_SAE_ID']),
    ('llama_scope_lxr_8x', 'l20r_8x'),
    ('llama_scope_lxr_8x', 'l20r_8x'),
    ('llama_scope_lxr_8x', 'l20r_8x'),
]

def load_sae_with_candidates(candidates, name='primary'):
    errors = []
    for release, sae_id in candidates:
        try:
            print(f'Trying {name} SAE: release={release!r}, sae_id={sae_id!r}')
            sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
            sae = sae.to(DEVICE)
            sae.eval()
            print(f'Loaded {name} SAE:', release, sae_id)
            return sae, cfg_dict, sparsity, release, sae_id
        except Exception as e:
            errors.append((release, sae_id, repr(e)))
            print('  failed:', repr(e)[:300])
    print('\nAll SAE candidates failed. Errors:')
    for e in errors:
        print(e)
    raise RuntimeError(f'Could not load {name} SAE. Inspect printed SAELens directory and update CONFIG release/id.')

sae, sae_cfg, sae_sparsity, SAE_RELEASE_ACTUAL, SAE_ID_ACTUAL = load_sae_with_candidates(SAE_CANDIDATES_PRIMARY, 'primary')
downstream_sae, downstream_sae_cfg, downstream_sae_sparsity, DOWNSTREAM_SAE_RELEASE_ACTUAL, DOWNSTREAM_SAE_ID_ACTUAL = load_sae_with_candidates(SAE_CANDIDATES_DOWNSTREAM, 'downstream')

print('Primary actual:', SAE_RELEASE_ACTUAL, SAE_ID_ACTUAL)
print('Downstream actual:', DOWNSTREAM_SAE_RELEASE_ACTUAL, DOWNSTREAM_SAE_ID_ACTUAL)


################################################################################
# Cell 8
################################################################################
# Cell 5 — load Llama-3.1-8B model with TransformerLens

print('Loading model:', CONFIG['MODEL_NAME'])
model = HookedTransformer.from_pretrained(
    CONFIG['MODEL_NAME'],
    device=DEVICE,
    dtype=DTYPE,
    default_prepend_bos=False,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

print('Loaded model.')
print('d_model:', model.cfg.d_model, 'n_layers:', model.cfg.n_layers, 'd_vocab:', model.cfg.d_vocab)
print('Primary hook:', CONFIG['HOOK_NAME'])
print('Downstream hook:', CONFIG['DOWNSTREAM_HOOK_NAME'])


################################################################################
# Cell 9
################################################################################
# Cell 6 — load Wikitext contexts and tokenize

ds = load_dataset(CONFIG['HF_DATASET_NAME'], CONFIG['HF_DATASET_CONFIG'], split=CONFIG['HF_DATASET_SPLIT'])
texts = []
seen = set()
for row in ds:
    t = clean_text(row.get('text', ''))
    if len(t) < 200:
        continue
    if t in seen:
        continue
    seen.add(t)
    texts.append(t)
    if len(texts) >= CONFIG['MAX_DISTINCT_TEXTS']:
        break

assert len(texts) >= CONFIG['MIN_DISTINCT_TEXTS'], f'Only found {len(texts)} usable texts.'
print('Distinct texts:', len(texts))

all_tokens = []
for t in tqdm(texts, desc='tokenizing'):
    toks = model.to_tokens(t, prepend_bos=False, truncate=True).squeeze(0)
    if toks.numel() >= CONFIG['CONTEXT_LEN']:
        all_tokens.append(toks[:CONFIG['CONTEXT_LEN']].cpu())
    if len(all_tokens) >= CONFIG['N_CONTEXTS']:
        break

assert len(all_tokens) >= CONFIG['N_CONTEXTS'], f'Only found {len(all_tokens)} token contexts.'
tokens = torch.stack(all_tokens[:CONFIG['N_CONTEXTS']], dim=0).long()
print('tokens:', tuple(tokens.shape))
torch.save(tokens, OUT_DIR / 'llama31_phase123_tokens.pt')


################################################################################
# Cell 10
################################################################################
# Cell 7 — Phase 1 clean pass: logits, primary SAE activations, downstream SAE activations
ACTIVE_THRESH = CONFIG['ACTIVE_THRESH']
HOOK_NAME = CONFIG['HOOK_NAME']
DOWNSTREAM_HOOK_NAME = CONFIG['DOWNSTREAM_HOOK_NAME']

clean_final_logits_chunks = []
final_primary_feat_chunks = []
downstream_final_feat_chunks = []
names_filter = [HOOK_NAME, DOWNSTREAM_HOOK_NAME]

for i in tqdm(range(0, tokens.shape[0], CONFIG['BATCH_CONTEXTS']), desc='clean pass'):
    batch = tokens[i:i+CONFIG['BATCH_CONTEXTS']].to(DEVICE)
    with torch.no_grad():
        logits, cache = model.run_with_cache(batch, names_filter=names_filter)
        clean_final_logits_chunks.append(logits[:, -1, :].detach().to('cpu', dtype=torch.float32))
        primary_feat = sae_encode(cache[HOOK_NAME], sae)[:, -1, :]
        final_primary_feat_chunks.append(primary_feat.detach().to('cpu', dtype=torch.float32))
        downstream_feat = sae_encode(cache[DOWNSTREAM_HOOK_NAME], downstream_sae)[:, -1, :]
        downstream_final_feat_chunks.append(downstream_feat.detach().to('cpu', dtype=torch.float32))
    del batch, logits, cache
    torch.cuda.empty_cache()

clean_final_logits = torch.cat(clean_final_logits_chunks, dim=0)
final_feat_primary = torch.cat(final_primary_feat_chunks, dim=0)
feat_downstream_clean = torch.cat(downstream_final_feat_chunks, dim=0)
print('clean_final_logits:', tuple(clean_final_logits.shape))
print('final_feat_primary:', tuple(final_feat_primary.shape))
print('feat_downstream_clean:', tuple(feat_downstream_clean.shape))
torch.save(clean_final_logits, OUT_DIR / 'llama31_clean_final_logits.pt')
torch.save(final_feat_primary, OUT_DIR / 'llama31_final_feat_primary.pt')
torch.save(feat_downstream_clean, OUT_DIR / 'llama31_feat_downstream_clean.pt')


################################################################################
# Cell 11
################################################################################
# Cell 8 — select primary SAE features and downstream collateral panel
acts = final_feat_primary
freq = (acts > ACTIVE_THRESH).float().mean(dim=0)
max_act = acts.max(dim=0).values
mask = (freq >= 0.005) & (freq <= 0.50) & (max_act > 0)
eligible = torch.where(mask)[0].cpu().numpy()
print('Eligible features:', len(eligible), 'of', acts.shape[1])
rng = np.random.default_rng(CONFIG['SEED'])
if len(eligible) < CONFIG['N_FEATURES']:
    print('Warning: fewer eligible features than requested. Using all eligible features.')
    selected_features = eligible.tolist()
else:
    eligible_freq = freq[eligible].cpu().numpy()
    order = eligible[np.argsort(eligible_freq)]
    bins = np.array_split(order, CONFIG['N_FEATURES'])
    selected_features = [int(rng.choice(b)) for b in bins if len(b) > 0][:CONFIG['N_FEATURES']]
selected_features = np.array(selected_features, dtype=int)
print('Selected features:', len(selected_features))

down_freq = (feat_downstream_clean > ACTIVE_THRESH).float().mean(dim=0)
panel_size = min(CONFIG['DOWNSTREAM_PANEL_SIZE'], feat_downstream_clean.shape[1])
downstream_panel = torch.topk(down_freq, k=panel_size).indices.cpu().numpy().astype(int)
print('Downstream panel size:', len(downstream_panel))
np.save(OUT_DIR / 'llama31_selected_features.npy', selected_features)
np.save(OUT_DIR / 'llama31_downstream_panel.npy', downstream_panel)


################################################################################
# Cell 12
################################################################################
# Cell 9 — Phase 1 predictors: geometry, activation, coactivation, direct-logit footprint
W_dec = ensure_dec_shape(get_decoder_matrix(sae).detach().to('cpu', dtype=torch.float32), model.cfg.d_model)
W_enc = ensure_enc_shape(get_encoder_matrix(sae), model.cfg.d_model, W_dec.shape[0])
if W_enc is not None:
    W_enc = W_enc.detach().to('cpu', dtype=torch.float32)
print('W_dec:', tuple(W_dec.shape), 'W_enc:', None if W_enc is None else tuple(W_enc.shape))

selected = torch.tensor(selected_features, dtype=torch.long)
rows = []
W_dec_norm = F.normalize(W_dec, dim=1)
B = (final_feat_primary > ACTIVE_THRESH).to(torch.float32)
W_U = model.W_U.detach().to('cpu', dtype=torch.float32)

for f in tqdm(selected_features, desc='predictors'):
    a = final_feat_primary[:, f].numpy().astype(float)
    active = a > ACTIVE_THRESH
    p = active.mean()
    row = {'feature': int(f)}
    row['phase1_final_act_freq'] = float(p)
    row['token_act_mean'] = float(np.mean(a))
    row['token_act_std'] = float(np.std(a))
    row['token_act_max'] = float(np.max(a))
    row['token_act_kurtosis'] = float(pd.Series(a).kurtosis()) if np.std(a) > 0 else np.nan
    row['token_binary_entropy'] = float(-(p*np.log(p+1e-12) + (1-p)*np.log(1-p+1e-12)))
    if a.sum() > 0:
        r = a / (a.sum() + 1e-12)
        row['token_activation_entropy_norm'] = float(-(r * np.log(r+1e-12)).sum() / np.log(len(r)))
    else:
        row['token_activation_entropy_norm'] = np.nan
    d = W_dec[f]
    row['decoder_norm'] = float(torch.linalg.norm(d).item())
    if W_enc is not None:
        e = W_enc[f]
        row['encoder_norm'] = float(torch.linalg.norm(e).item())
        row['encoder_decoder_cos'] = float(F.cosine_similarity(e, d, dim=0, eps=1e-8).item())
    else:
        row['encoder_norm'] = np.nan
        row['encoder_decoder_cos'] = np.nan
    sims = (W_dec_norm @ W_dec_norm[f]).abs()
    sims[f] = -1
    topk = torch.topk(sims, k=min(20, sims.numel()-1)).values
    row['crowding_topk_mean_abs_cos'] = float(topk.mean().item())
    row['crowding_topk_sum_abs_cos'] = float(topk.sum().item())
    row['crowding_max_abs_cos'] = float(topk.max().item())
    if active.sum() >= 2:
        B_active = B[torch.tensor(active)]
        co_rates = B_active.mean(dim=0)
        co_rates[f] = 0
        row['coact_count_mean'] = float(B_active.sum(dim=1).mean().item())
        probs = co_rates / (co_rates.sum() + 1e-12)
        row['coact_entropy_norm'] = float((-(probs * torch.log(probs + 1e-12)).sum() / math.log(len(probs))).item())
        row['coact_top20_mass'] = float(torch.topk(probs, k=min(20, len(probs))).values.sum().item())
    else:
        row['coact_count_mean'] = np.nan
        row['coact_entropy_norm'] = np.nan
        row['coact_top20_mass'] = np.nan
    rlogit = d @ W_U
    abs_r = rlogit.abs()
    mass = abs_r / (abs_r.sum() + 1e-12)
    row['direct_logit_l2'] = float(torch.linalg.norm(rlogit).item())
    row['direct_logit_linf'] = float(abs_r.max().item())
    row['direct_logit_entropy'] = float((-(mass * torch.log(mass + 1e-12)).sum()).item())
    row['direct_logit_top10_mass_frac'] = float(torch.topk(mass, k=10).values.sum().item())
    row['direct_logit_top100_mass_frac'] = float(torch.topk(mass, k=100).values.sum().item())
    rows.append(row)

predictor_df = pd.DataFrame(rows)
pred_path = OUT_DIR / 'llama31_phase1_predictors.csv'
predictor_df.to_csv(pred_path, index=False)
print('Saved:', pred_path)
display(predictor_df.head())


################################################################################
# Cell 13
################################################################################
# Cell 10 — Phase 2 helpers: context selection and steering evaluation
selected_features = np.load(OUT_DIR / 'llama31_selected_features.npy')
downstream_panel = np.load(OUT_DIR / 'llama31_downstream_panel.npy')
clean_logits_cpu = clean_final_logits
clean_downstream_panel_cpu = feat_downstream_clean[:, downstream_panel]

def get_context_indices_for_feature(f: int, contexts_per_type: int = None):
    if contexts_per_type is None:
        contexts_per_type = CONFIG['CONTEXTS_PER_TYPE']
    a = final_feat_primary[:, f].numpy()
    n = len(a)
    rng = np.random.default_rng(CONFIG['SEED'] + int(f))
    top_idx = np.argsort(-a)[:contexts_per_type]
    low_idx = np.argsort(a)[:contexts_per_type]
    rand_idx = rng.choice(np.arange(n), size=contexts_per_type, replace=False)
    mixed = []
    for arr in [top_idx, rand_idx, low_idx]:
        for v in arr:
            if int(v) not in mixed:
                mixed.append(int(v))
    return {'top': top_idx.astype(int), 'random': rand_idx.astype(int), 'low': low_idx.astype(int), 'mixed': np.array(mixed, dtype=int)}

def make_add_hook(d_vec: torch.Tensor, alpha: float):
    d_vec = d_vec.to(DEVICE, dtype=DTYPE)
    def hook_fn(resid, hook):
        resid[:, -1, :] = resid[:, -1, :] + alpha * d_vec
        return resid
    return hook_fn

def evaluate_feature_steering(f: int, context_idx: np.ndarray, alpha: float):
    d_vec = W_dec[int(f)].to(DEVICE, dtype=DTYPE)
    effect_chunks = []
    down_delta_chunks = []
    kl_vals = []
    for start in range(0, len(context_idx), CONFIG['BATCH_STEER']):
        idx = context_idx[start:start+CONFIG['BATCH_STEER']]
        batch = tokens[idx].to(DEVICE)
        clean_logits = clean_logits_cpu[idx].to(DEVICE)
        clean_down = clean_downstream_panel_cpu[idx].to(DEVICE)
        hook_fn = make_add_hook(d_vec, alpha)
        with torch.no_grad():
            with model.hooks(fwd_hooks=[(HOOK_NAME, hook_fn)]):
                steered_logits, steered_cache = model.run_with_cache(batch, names_filter=[DOWNSTREAM_HOOK_NAME])
            steered_final_logits = steered_logits[:, -1, :].to(torch.float32)
            effect = steered_final_logits - clean_logits.to(torch.float32)
            effect_chunks.append(effect.detach().cpu())
            steered_down = sae_encode(steered_cache[DOWNSTREAM_HOOK_NAME], downstream_sae)[:, -1, :]
            steered_down_panel = steered_down[:, downstream_panel].to(torch.float32)
            down_delta = steered_down_panel - clean_down.to(torch.float32)
            down_delta_chunks.append(down_delta.detach().cpu())
            logp = F.log_softmax(clean_logits.to(torch.float32), dim=-1)
            logq = F.log_softmax(steered_final_logits, dim=-1)
            p = torch.exp(logp)
            kl = (p * (logp - logq)).sum(dim=-1)
            kl_vals.extend(kl.detach().cpu().tolist())
        del batch, clean_logits, clean_down, steered_logits, steered_cache, steered_final_logits, effect
        torch.cuda.empty_cache()
    effects = torch.cat(effect_chunks, dim=0)
    down_deltas = torch.cat(down_delta_chunks, dim=0)
    effect_l2 = torch.linalg.norm(effects, dim=1)
    signed_stab, abs_stab = cosine_rows_to_mean(effects)
    downstream_count = (down_deltas.abs() > CONFIG['DOWNSTREAM_DELTA_THRESH']).float().sum(dim=1)
    downstream_l2 = torch.linalg.norm(down_deltas, dim=1)
    return dict(
        n_contexts=int(len(context_idx)),
        effect_l2_mean=float(effect_l2.mean().item()),
        effect_l2_std=float(effect_l2.std(unbiased=False).item()),
        effect_cv=float((effect_l2.std(unbiased=False) / (effect_l2.mean() + 1e-8)).item()),
        stability_to_mean_signed_cos=float(signed_stab),
        stability_to_mean_abs_cos=float(abs_stab),
        downstream_feat_count_abs_delta_gt_0_05=float(downstream_count.mean().item()),
        downstream_feat_effective_moved_mean=float(downstream_l2.mean().item()),
        downstream_count_0_05_per_effect_l2=float((downstream_count.mean() / (effect_l2.mean() + 1e-8)).item()),
        downstream_l2_per_effect_l2=float((downstream_l2.mean() / (effect_l2.mean() + 1e-8)).item()),
        kl_clean_to_steered_mean=float(np.mean(kl_vals)),
        kl_clean_to_steered_per_effect_l2=float(np.mean(kl_vals) / (float(effect_l2.mean().item()) + 1e-8)),
    )


################################################################################
# Cell 14
################################################################################
# Cell 11 — Phase 2: run additive steering labels for selected features
label_rows = []
alpha = CONFIG['FIXED_GLOBAL_CLAMP_VALUE']
mode = CONFIG['PRIMARY_CLAMP_MODE']
for f in tqdm(selected_features, desc='feature steering labels'):
    context_sets = get_context_indices_for_feature(int(f))
    labels = evaluate_feature_steering(int(f), context_sets['mixed'], alpha=alpha)
    row = {'feature': int(f)}
    for k, v in labels.items():
        row[f'{k}__{mode}__mixed'] = v
    row['fixed_global_clamp_value'] = alpha
    label_rows.append(row)
    if len(label_rows) % 25 == 0:
        pd.DataFrame(label_rows).to_csv(OUT_DIR / 'llama31_phase2_steering_labels_partial.csv', index=False)
label_df = pd.DataFrame(label_rows)
labels_path = OUT_DIR / 'llama31_phase2_steering_labels.csv'
label_df.to_csv(labels_path, index=False)
print('Saved:', labels_path)
display(label_df.head())


################################################################################
# Cell 15
################################################################################
# Cell 12 — merge Phase 1 predictors + Phase 2 labels
predictor_df = pd.read_csv(OUT_DIR / 'llama31_phase1_predictors.csv')
label_df = pd.read_csv(OUT_DIR / 'llama31_phase2_steering_labels.csv')
merged_df = predictor_df.merge(label_df, on='feature', how='inner')
merged_path = OUT_DIR / 'robust_phase2_phase3_merged.csv'
merged_df.to_csv(merged_path, index=False)
print('Merged:', merged_df.shape)
print('Saved:', merged_path)
display(merged_df.head())


################################################################################
# Cell 16
################################################################################
# Cell 13 — Phase 3 setup: predictor sets and target columns
work_df = pd.read_csv(OUT_DIR / 'robust_phase2_phase3_merged.csv')
PRIMARY_TARGETS = [
    'stability_to_mean_signed_cos__fixed_global_add__mixed',
    'stability_to_mean_abs_cos__fixed_global_add__mixed',
    'downstream_feat_count_abs_delta_gt_0_05__fixed_global_add__mixed',
    'downstream_count_0_05_per_effect_l2__fixed_global_add__mixed',
    'downstream_feat_effective_moved_mean__fixed_global_add__mixed',
    'downstream_l2_per_effect_l2__fixed_global_add__mixed',
    'effect_l2_mean__fixed_global_add__mixed',
    'effect_cv__fixed_global_add__mixed',
]
PRIMARY_TARGETS = [c for c in PRIMARY_TARGETS if c in work_df.columns]
print('Primary targets:', PRIMARY_TARGETS)
PREDICTOR_SETS = {
    'frequency_only': ['phase1_final_act_freq'],
    'activation_magnitude_only': ['token_act_mean', 'token_act_std', 'token_act_max', 'token_act_kurtosis'],
    'geometry_only': ['decoder_norm', 'encoder_norm', 'encoder_decoder_cos', 'crowding_topk_mean_abs_cos', 'crowding_max_abs_cos'],
    'direct_logit_only': ['direct_logit_l2', 'direct_logit_linf', 'direct_logit_entropy', 'direct_logit_top10_mass_frac', 'direct_logit_top100_mass_frac'],
    'coactivation_only': ['coact_count_mean', 'coact_entropy_norm', 'coact_top20_mass'],
}
PREDICTOR_SETS['full_no_magnitude'] = [
    'phase1_final_act_freq', 'token_binary_entropy', 'token_activation_entropy_norm',
    'decoder_norm', 'encoder_norm', 'encoder_decoder_cos', 'crowding_topk_mean_abs_cos', 'crowding_max_abs_cos',
    'coact_count_mean', 'coact_entropy_norm', 'coact_top20_mass',
    'direct_logit_l2', 'direct_logit_linf', 'direct_logit_entropy', 'direct_logit_top10_mass_frac', 'direct_logit_top100_mass_frac',
]
PREDICTOR_SETS['full_all'] = sorted(set(sum(PREDICTOR_SETS.values(), [])))
for k in list(PREDICTOR_SETS.keys()):
    PREDICTOR_SETS[k] = [c for c in PREDICTOR_SETS[k] if c in work_df.columns]
    print(k, len(PREDICTOR_SETS[k]), PREDICTOR_SETS[k])


################################################################################
# Cell 17
################################################################################
# Cell 14 — Phase 3A: univariate correlations
all_predictors = sorted(set(sum(PREDICTOR_SETS.values(), [])))
rows = []
for pred in all_predictors:
    for target in PRIMARY_TARGETS:
        sub = work_df[[pred, target]].replace([np.inf, -np.inf], np.nan).dropna()
        sp_r, sp_p = safe_corr(spearmanr, sub[pred], sub[target])
        pe_r, pe_p = safe_corr(pearsonr, sub[pred], sub[target])
        rows.append(dict(predictor=pred, target=target, target_is_primary=True, spearman_r=sp_r, spearman_p=sp_p, pearson_r=pe_r, pearson_p=pe_p, abs_spearman=abs(sp_r) if np.isfinite(sp_r) else np.nan, n=len(sub)))
corr_df = pd.DataFrame(rows).sort_values('abs_spearman', ascending=False)
corr_path = OUT_DIR / 'robust_phase3_univariate_correlations.csv'
corr_df.to_csv(corr_path, index=False)
print('Saved:', corr_path)
display(corr_df.head(20))


################################################################################
# Cell 18
################################################################################
# Cell 15 — Phase 3B: cross-validated ridge prediction

def cv_ridge_predict(df, predictors, target, n_splits=5, alpha=1.0, seed=0):
    needed = predictors + [target]
    sub = df[needed].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(sub) < max(20, n_splits * 4) or len(predictors) == 0:
        return dict(n=len(sub), cv_spearman=np.nan, cv_spearman_p=np.nan, cv_r2=np.nan)
    X = sub[predictors].values.astype(float)
    y = sub[target].values.astype(float)
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        model_r = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model_r.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model_r.predict(X[test_idx])
    rho, p = safe_corr(spearmanr, preds, y)
    r2 = r2_score(y[np.isfinite(preds)], preds[np.isfinite(preds)]) if np.isfinite(preds).sum() > 2 else np.nan
    return dict(n=len(sub), cv_spearman=rho, cv_spearman_p=p, cv_r2=float(r2))

cv_rows = []
for target in tqdm(PRIMARY_TARGETS, desc='CV targets'):
    for set_name, preds in PREDICTOR_SETS.items():
        res = cv_ridge_predict(work_df, preds, target, n_splits=CONFIG['N_SPLITS'], alpha=CONFIG['RIDGE_ALPHA'], seed=CONFIG['SEED'])
        cv_rows.append(dict(target=target, predictor_set=set_name, n_predictors=len(preds), **res))
cv_df = pd.DataFrame(cv_rows)
cv_path = OUT_DIR / 'robust_phase3_cv_regression_results.csv'
cv_df.to_csv(cv_path, index=False)
print('Saved:', cv_path)
display(cv_df.sort_values(['target','cv_spearman'], ascending=[True,False]).head(30))


################################################################################
# Cell 19
################################################################################
# Cell 16 — Phase 3C: baseline comparison table
pivot = cv_df.pivot(index='target', columns='predictor_set', values='cv_spearman').reset_index()
for col in ['frequency_only', 'activation_magnitude_only', 'geometry_only', 'full_no_magnitude', 'full_all']:
    if col not in pivot.columns:
        pivot[col] = np.nan
pivot['full_no_magnitude_minus_freq'] = pivot['full_no_magnitude'] - pivot['frequency_only']
pivot['full_no_magnitude_minus_activation_magnitude'] = pivot['full_no_magnitude'] - pivot['activation_magnitude_only']
pivot['geometry_only_minus_freq'] = pivot['geometry_only'] - pivot['frequency_only']
pivot['full_all_minus_freq'] = pivot['full_all'] - pivot['frequency_only']
baseline_path = OUT_DIR / 'robust_phase3_baseline_comparison.csv'
pivot.to_csv(baseline_path, index=False)
print('Saved:', baseline_path)
display(pivot)


################################################################################
# Cell 20
################################################################################
# Cell 17 — Phase 3D: residualized robustness for stability targets
RESID_TARGETS = [c for c in ['stability_to_mean_signed_cos__fixed_global_add__mixed', 'stability_to_mean_abs_cos__fixed_global_add__mixed'] if c in work_df.columns]
NUISANCE_COLS = [c for c in ['effect_l2_mean__fixed_global_add__mixed', 'fixed_global_clamp_value', 'token_act_mean'] if c in work_df.columns]
print('Residual targets:', RESID_TARGETS)
print('Nuisance cols:', NUISANCE_COLS)
resid_rows = []
resid_work = work_df.copy()
for target in RESID_TARGETS:
    needed = [target] + NUISANCE_COLS
    sub_idx = resid_work[needed].replace([np.inf, -np.inf], np.nan).dropna().index
    if len(sub_idx) < 20 or len(NUISANCE_COLS) == 0:
        continue
    Xn = resid_work.loc[sub_idx, NUISANCE_COLS].values.astype(float)
    y = resid_work.loc[sub_idx, target].values.astype(float)
    nuisance_model = make_pipeline(StandardScaler(), LinearRegression())
    nuisance_model.fit(Xn, y)
    y_hat = nuisance_model.predict(Xn)
    resid_col = target + '__residualized'
    resid_work[resid_col] = np.nan
    resid_work.loc[sub_idx, resid_col] = y - y_hat
    for set_name, preds in PREDICTOR_SETS.items():
        res = cv_ridge_predict(resid_work, preds, resid_col, n_splits=CONFIG['N_SPLITS'], alpha=CONFIG['RIDGE_ALPHA'], seed=CONFIG['SEED'])
        resid_rows.append(dict(original_target=target, target=resid_col, predictor_set=set_name, n_predictors=len(preds), **res))
resid_df = pd.DataFrame(resid_rows)
resid_path = OUT_DIR / 'robust_phase3_residualized_target_results.csv'
resid_df.to_csv(resid_path, index=False)
print('Saved:', resid_path)
display(resid_df.sort_values(['original_target','cv_spearman'], ascending=[True,False]).head(30))


################################################################################
# Cell 21
################################################################################
# Cell 18 — final summary verdict
primary_corr = corr_df[corr_df['target_is_primary']].copy()
best_primary_corr = primary_corr.iloc[0] if len(primary_corr) else None
if len(pivot):
    best_full_no_mag_minus_freq = pivot['full_no_magnitude_minus_freq'].max()
    best_full_no_mag_minus_actmag = pivot['full_no_magnitude_minus_activation_magnitude'].max()
    mean_full_no_mag_minus_freq = pivot['full_no_magnitude_minus_freq'].mean()
    mean_full_no_mag_minus_actmag = pivot['full_no_magnitude_minus_activation_magnitude'].mean()
else:
    best_full_no_mag_minus_freq = np.nan
    best_full_no_mag_minus_actmag = np.nan
    mean_full_no_mag_minus_freq = np.nan
    mean_full_no_mag_minus_actmag = np.nan
summary = {
    'model': CONFIG['MODEL_NAME'],
    'n_features': int(len(selected_features)),
    'n_contexts': int(tokens.shape[0]),
    'corpus_distinct_texts': int(len(texts)),
    'fixed_global_clamp_value': float(CONFIG['FIXED_GLOBAL_CLAMP_VALUE']),
    'primary_clamp_mode': CONFIG['PRIMARY_CLAMP_MODE'],
    'sae_release_actual': SAE_RELEASE_ACTUAL,
    'sae_id_actual': SAE_ID_ACTUAL,
    'downstream_sae_release_actual': DOWNSTREAM_SAE_RELEASE_ACTUAL,
    'downstream_sae_id_actual': DOWNSTREAM_SAE_ID_ACTUAL,
    'best_primary_corr': None if best_primary_corr is None else {'predictor': str(best_primary_corr['predictor']), 'target': str(best_primary_corr['target']), 'spearman_r': float(best_primary_corr['spearman_r']), 'spearman_p': float(best_primary_corr['spearman_p'])},
    'best_full_no_magnitude_minus_freq': None if np.isnan(best_full_no_mag_minus_freq) else float(best_full_no_mag_minus_freq),
    'best_full_no_magnitude_minus_activation_magnitude': None if np.isnan(best_full_no_mag_minus_actmag) else float(best_full_no_mag_minus_actmag),
    'mean_full_no_magnitude_minus_freq': None if np.isnan(mean_full_no_mag_minus_freq) else float(mean_full_no_mag_minus_freq),
    'mean_full_no_magnitude_minus_activation_magnitude': None if np.isnan(mean_full_no_mag_minus_actmag) else float(mean_full_no_mag_minus_actmag),
    'config': CONFIG,
}
summary_path = OUT_DIR / 'robust_phase123_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print('='*80)
print('LLAMA-3.1-8B PHASE 1/2/3 VERDICT')
print('='*80)
print(f'n features: {len(selected_features)}')
print(f'n contexts: {tokens.shape[0]}')
print(f'distinct corpus texts: {len(texts)}')
print(f'primary intervention: {CONFIG["PRIMARY_CLAMP_MODE"]} alpha={CONFIG["FIXED_GLOBAL_CLAMP_VALUE"]}')
print('-'*80)
if best_primary_corr is not None:
    print('Best primary-target univariate relationship:')
    print(best_primary_corr[['predictor','target','spearman_r','spearman_p']])
else:
    print('No primary-target correlation found.')
print('-'*80)
print(f'Best full_no_magnitude - frequency CV Spearman improvement: {best_full_no_mag_minus_freq:.3f}')
print(f'Best full_no_magnitude - activation_magnitude CV Spearman improvement: {best_full_no_mag_minus_actmag:.3f}')
print(f'Mean full_no_magnitude - frequency improvement across primary targets: {mean_full_no_mag_minus_freq:.3f}')
print(f'Mean full_no_magnitude - activation_magnitude improvement across primary targets: {mean_full_no_mag_minus_actmag:.3f}')
print('-'*80)
if (best_primary_corr is not None and abs(best_primary_corr['spearman_r']) >= 0.25 and best_full_no_mag_minus_freq >= 0.05 and best_full_no_mag_minus_actmag >= 0.00):
    print('VERDICT: LLAMA SIGNAL SURVIVED. Consider Phase 4 screening.')
elif (best_primary_corr is not None and abs(best_primary_corr['spearman_r']) >= 0.20 and best_full_no_mag_minus_freq >= 0.00):
    print('VERDICT: LLAMA PROMISING BUT BORDERLINE. Inspect targets before Phase 4.')
else:
    print('VERDICT: WEAK LLAMA RESULT AFTER ROBUST CONTROLS. Revisit SAE/hook/layer or labels.')
print('Saved summary:', summary_path)
