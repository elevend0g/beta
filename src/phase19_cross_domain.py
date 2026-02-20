"""
Phase 19: Cross-Domain Proprioception

Tests whether architectural proprioception is a general property of
thermodynamically-trained SSMs reasoning over structured sequential tasks,
or specific to the parity domain.

Primary task: Symbolic Sequence Sorting (symbols A–F, bubble-sort reasoning)

Sub-experiments:
  A) Independent training – C_sort, D_sort, E_sort
  B) Zero-shot transfer  – E_ssm parity backbone loaded into sorting model
  C) Few-shot adaptation – backbone from E_ssm + {100,500,1000} sorting examples

Output artifacts:
  results/phase19_{group}_model.pt
  results/phase19_{group}_metrics.json
  results/phase19_transfer_metrics.json
  results/phase19_comparison_table.json
  figures/fig_p19_1_gradient_comparison.png
  figures/fig_p19_2_transfer_curve.png
  figures/fig_p19_3_tier_analysis.png
  figures/fig_p19_4_trajectory_gallery.png
  figures/fig_p19_5_xcorr_comparison.png
"""

import os
import sys
import json
import math
import time
import copy
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats as sp_stats
from scipy.signal import correlate

sns.set_theme(style='whitegrid', font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from models import PNA_SSM, count_parameters
from losses import ThermodynamicLoss, CrossEntropyLoss
from train import get_device, get_cosine_schedule
from ssm_state_entropy_collapse import compute_state_entropy
from phase17_proprioception_repro import compute_all_metrics, aggregate_metrics


# ══════════════════════════════════════════════════════════════════════════════
# Sorting vocabulary
# ══════════════════════════════════════════════════════════════════════════════

SORT_VOCAB = {
    '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<HALT>': 3,
    'Input:': 4, 'Result:': 5,
    'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11,
    '>': 12, '<': 13, '=': 14,
    'swap': 15, 'keep': 16, ' ': 17,
}
SORT_ID_TO_TOKEN = {v: k for k, v in SORT_VOCAB.items()}
SORT_VOCAB_SIZE  = len(SORT_VOCAB)   # 18

SORT_ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F']

# ── Hyper-parameters ──────────────────────────────────────────────────────────
D_MODEL    = 512
D_STATE    = 16
N_LAYERS   = 6
MAX_SEQ    = 256
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 3e-4

FEW_SHOT_NS = [100, 500, 1000]

# Parity reference (Phase 17b)
PARITY_REF = {
    'C':     {'mean_r': -0.290, 'tau_drv': 2,  'label': 'SSM+CE'},
    'D':     {'mean_r': -0.725, 'tau_drv': -2, 'label': 'SSM+L_th'},
    'E_ssm': {'mean_r': -0.836, 'tau_drv': -2, 'label': 'SSM+halt'},
}

BLUE  = '#2980b9'
GREEN = '#27ae60'
RED   = '#e74c3c'

_MULTI_TOKENS = ('Input:', 'Result:', 'swap', 'keep')


# ══════════════════════════════════════════════════════════════════════════════
# Tokeniser
# ══════════════════════════════════════════════════════════════════════════════

def sort_tokenize(text: str) -> list:
    """Convert a sorting-task text string to a list of token IDs."""
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for tok in _MULTI_TOKENS:
            n = len(tok)
            if text[i:i + n] == tok:
                tokens.append(SORT_VOCAB[tok])
                i += n
                matched = True
                break
        if not matched:
            tokens.append(SORT_VOCAB.get(text[i], SORT_VOCAB[' ']))
            i += 1
    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# Sorting example generation (bubble sort)
# ══════════════════════════════════════════════════════════════════════════════

def generate_sorting_example(n_symbols: int, rng: random.Random):
    """
    Generate one bubble-sort example.
    Returns (text, sorted_list_of_symbols).

    Format:
      Input:D B F A C E D>B swap D<F keep F>A swap ... Result:A B C D E F
    """
    seq = [rng.choice(SORT_ALPHABET) for _ in range(n_symbols)]
    input_str = 'Input:' + ' '.join(seq)

    arr = list(seq)
    comparisons = []
    n_active = len(arr)
    for _ in range(len(arr) - 1):     # at most n-1 passes
        swapped = False
        for i in range(n_active - 1):  # reduce bound: last element already sorted
            a, b = arr[i], arr[i + 1]
            if a > b:
                comparisons.append(f"{a}>{b} swap")
                arr[i], arr[i + 1] = b, a
                swapped = True
            elif a < b:
                comparisons.append(f"{a}<{b} keep")
            else:
                comparisons.append(f"{a}={b} keep")
        n_active -= 1
        if not swapped:
            break

    result_str = 'Result:' + ' '.join(arr)
    if comparisons:
        text = f"{input_str} {' '.join(comparisons)} {result_str}"
    else:
        text = f"{input_str} {result_str}"

    return text, arr


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SortingDataset(Dataset):
    """Bubble-sort symbolic sequence dataset (Phase 19).

    Tiers by sequence length:
      T0: 3–4 symbols   T1: 5–6 symbols   T2: 7–8 symbols
    """

    def __init__(self, n_examples: int, sym_range=(3, 8),
                 max_seq_len: int = MAX_SEQ, seed: int = 42):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.examples: list = []
        rng = random.Random(seed)

        for _ in range(n_examples):
            n_sym = rng.randint(*sym_range)
            text, sorted_seq = generate_sorting_example(n_sym, rng)

            raw = ([SORT_VOCAB['<BOS>']]
                   + sort_tokenize(text)
                   + [SORT_VOCAB['<HALT>'], SORT_VOCAB['<EOS>']])

            # Find special-token positions
            result_pos = next(
                (i for i, t in enumerate(raw) if t == SORT_VOCAB['Result:']),
                len(raw) - 3)
            input_pos = next(
                (i for i, t in enumerate(raw) if t == SORT_VOCAB['Input:']),
                0)

            # Reasoning mask: everything between Input: and Result:
            mask = [0] * len(raw)
            for i in range(input_pos + 1, result_pos):
                mask[i] = 1

            tier = 0 if n_sym <= 4 else (1 if n_sym <= 6 else 2)

            self.examples.append({
                'tokens':         raw,
                'result_pos':     result_pos,
                'reasoning_mask': mask,
                'n_symbols':      n_sym,
                'sorted_seq':     sorted_seq,
                'tier':           tier,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex   = self.examples[idx]
        toks = ex['tokens']
        mask = ex['reasoning_mask']

        if len(toks) > self.max_seq_len:
            toks = toks[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
        else:
            pad  = self.max_seq_len - len(toks)
            toks = toks + [SORT_VOCAB['<PAD>']] * pad
            mask = mask + [0] * pad

        rp = min(ex['result_pos'], self.max_seq_len - 2)
        return {
            'input_ids':      torch.tensor(toks[:-1], dtype=torch.long),
            'targets':        torch.tensor(toks[1:],  dtype=torch.long),
            'reasoning_mask': torch.tensor(mask[:-1], dtype=torch.float),
            'result_pos':     torch.tensor(rp,                 dtype=torch.long),
            'tier':           torch.tensor(ex['tier'],         dtype=torch.long),
            'n_symbols':      torch.tensor(ex['n_symbols'],    dtype=torch.long),
        }


def create_sorting_datasets():
    train = SortingDataset(8000, seed=42)
    val   = SortingDataset(1000, seed=123)
    test  = SortingDataset(1000, seed=456)
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════════
# Model + loss helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_sorting_model(device):
    """Create a fresh PNA_SSM with sorting vocabulary."""
    return PNA_SSM(
        SORT_VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        d_state=D_STATE, max_seq_len=MAX_SEQ
    ).to(device)


def make_loss_fn(group_tag: str):
    """
    group_tag: 'C_sort' | 'D_sort' | 'E_sort'
    Returns a ThermodynamicLoss or CrossEntropyLoss (no AdaptiveGovernor).
    """
    pad = SORT_VOCAB['<PAD>']
    if group_tag == 'C_sort':
        return CrossEntropyLoss(pad_token_id=pad)
    elif group_tag == 'D_sort':
        return ThermodynamicLoss(alpha=0.05, beta=0.0, gamma=0.0,
                                 pad_token_id=pad)
    elif group_tag == 'E_sort':
        return ThermodynamicLoss(alpha=0.05, beta=0.10, gamma=0.0,
                                 pad_token_id=pad)
    else:
        raise ValueError(f"Unknown group_tag: {group_tag}")


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train_sorting_model(model, loss_fn, train_ds, val_ds, device,
                        epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                        label=''):
    """Train model with early stopping on val loss. Returns best model."""
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler   = get_cosine_schedule(optimizer, total_steps, warmup_steps=100)

    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    patience         = 10

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        for batch in train_loader:
            inp  = batch['input_ids'].to(device)
            tgt  = batch['targets'].to(device)
            rmsk = batch['reasoning_mask'].to(device)
            rpos = batch['result_pos'].to(device)

            out  = model(inp)
            ld   = loss_fn(logits=out['logits'], targets=tgt,
                           halt_confidence=out['halt_confidence'],
                           states_sequence=out.get('states_sequence'),
                           reasoning_mask=rmsk,
                           result_token_positions=rpos)
            optimizer.zero_grad()
            ld['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                inp  = batch['input_ids'].to(device)
                tgt  = batch['targets'].to(device)
                rmsk = batch['reasoning_mask'].to(device)
                rpos = batch['result_pos'].to(device)
                out  = model(inp)
                ld   = loss_fn(logits=out['logits'], targets=tgt,
                               halt_confidence=out['halt_confidence'],
                               states_sequence=out.get('states_sequence'),
                               reasoning_mask=rmsk,
                               result_token_positions=rpos)
                val_loss += ld['total'].item()
                preds = out['logits'].argmax(-1)
                for b in range(inp.size(0)):
                    rp = rpos[b].item()
                    if rp < tgt.size(1):
                        val_correct += int(preds[b, rp] == tgt[b, rp])
                        val_total   += 1
        val_loss /= max(1, len(val_loader))
        val_acc   = val_correct / max(1, val_total)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{label}] epoch {epoch:3d}  val_loss={val_loss:.4f}"
                  f"  val_acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [{label}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Accuracy evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_first_token_accuracy(model, dataset, device, batch_size=64):
    """Accuracy at result_pos (first sorted symbol). Used as proxy during eval."""
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    for batch in loader:
        inp  = batch['input_ids'].to(device)
        tgt  = batch['targets'].to(device)
        rpos = batch['result_pos']
        preds = model(inp)['logits'].argmax(-1)
        for b in range(inp.size(0)):
            rp = rpos[b].item()
            if rp < tgt.size(1):
                correct += int(preds[b, rp] == tgt[b, rp])
                total   += 1
    return correct / max(1, total)


@torch.no_grad()
def compute_full_accuracy(model, dataset, device, batch_size=64):
    """
    Full-sequence accuracy: ALL sorted symbols in Result: must be correct.
    Positions checked in targets: rp, rp+2, rp+4, … (symbols skip spaces).
    """
    model.eval()
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    for batch in loader:
        inp    = batch['input_ids'].to(device)
        tgt    = batch['targets'].to(device)
        rpos   = batch['result_pos']
        nsyms  = batch['n_symbols']
        preds  = model(inp)['logits'].argmax(-1)
        for b in range(inp.size(0)):
            rp   = rpos[b].item()
            nsym = nsyms[b].item()
            ok   = True
            for k in range(nsym):
                tp = rp + k * 2
                if tp >= tgt.size(1):
                    ok = False
                    break
                if preds[b, tp] != tgt[b, tp]:
                    ok = False
                    break
            correct += int(ok)
            total   += 1
    return correct / max(1, total)


# ══════════════════════════════════════════════════════════════════════════════
# Signal extraction
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_sorting_signals(model, dataset, device,
                             batch_size=64, min_rp=5):
    """
    Extract state_entropy and halt_confidence for every example.
    Returns list of dicts with keys: result_pos, state_entropy,
    halt_confidence, tier, n_symbols.
    """
    model.eval()
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    examples = []

    for batch in loader:
        inp  = batch['input_ids'].to(device)
        rpos = batch['result_pos']
        tier = batch['tier']
        nsym = batch['n_symbols']
        B    = inp.size(0)

        out  = model(inp)
        s_seq = out['states_sequence']   # [B, L, d_state]
        h_conf = out['halt_confidence']  # [B, L, 1]

        s_ent = compute_state_entropy(s_seq, method='energy')  # [B, L]

        for b in range(B):
            rp = rpos[b].item()
            if rp < min_rp:
                continue
            examples.append({
                'result_pos':      rp,
                'state_entropy':   s_ent[b, :rp + 1].cpu().numpy(),
                'halt_confidence': h_conf[b, :rp + 1, 0].cpu().numpy(),
                'tier':            tier[b].item(),
                'n_symbols':       nsym[b].item(),
            })

    return examples


# ══════════════════════════════════════════════════════════════════════════════
# Backbone transfer (Sub-experiments B & C)
# ══════════════════════════════════════════════════════════════════════════════

def load_parity_backbone(sorting_model, parity_ckpt_path, device):
    """
    Copy shared-dimension weights from a parity E_ssm checkpoint into a
    sorting model.  Copied: layers.*, norm.*, halt_head.*, pos_encoding.*
    Not copied: embedding.* (vocab-size dependent), token_head.*
    """
    parity_sd  = torch.load(parity_ckpt_path, map_location=device,
                            weights_only=True)
    sorting_sd = sorting_model.state_dict()

    skipped   = []
    copied    = []
    for key, pval in parity_sd.items():
        if key.startswith('embedding.') or key.startswith('token_head.'):
            skipped.append(key)
            continue
        if key in sorting_sd and sorting_sd[key].shape == pval.shape:
            sorting_sd[key] = pval.clone()
            copied.append(key)
        else:
            skipped.append(key)

    sorting_model.load_state_dict(sorting_sd)
    print(f"    Backbone transfer: {len(copied)} keys copied, "
          f"{len(skipped)} skipped")
    return sorting_model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate a single model: accuracy + proprioceptive metrics
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_sorting_model(model, test_ds, device, label=''):
    """
    Returns a metrics dict with:
      accuracy_first, accuracy_full,
      mean_r, tau_threshold, tau_drv, frac_negative, frac_significant,
      n_examples,
      tier_metrics: {0: {...}, 1: {...}, 2: {...}},
      agg: full aggregate dict,
      examples: raw signal list
    """
    model.eval()
    acc_first = compute_first_token_accuracy(model, test_ds, device)
    acc_full  = compute_full_accuracy(model, test_ds, device)
    print(f"  [{label}] accuracy: first={acc_first:.3f}  full={acc_full:.3f}")

    examples = extract_sorting_signals(model, test_ds, device)
    records, lag_range = compute_all_metrics(examples)
    agg = aggregate_metrics(records, lag_range)

    inst = agg['instantaneous']
    drv  = agg['deriv_xcorr']
    tlag = agg['threshold_lag']

    m = {
        'label':              label,
        'accuracy_first':     acc_first,
        'accuracy_full':      acc_full,
        'mean_r':             inst['mean_r'],
        'frac_negative':      inst['fraction_negative'],
        'frac_significant':   inst['fraction_significant'],
        'tau_threshold':      tlag['mean'],
        'tau_drv':            drv['peak_lag'],
        'n_examples':         inst['n_valid'],
        'agg':                agg,
        'examples':           examples,
        'records':            records,
    }

    # Per-tier metrics
    tier_metrics = {}
    for t in (0, 1, 2):
        t_ex = [e for e in examples if e['tier'] == t]
        if len(t_ex) < 5:
            tier_metrics[t] = None
            continue
        t_recs, t_lags = compute_all_metrics(t_ex)
        t_agg = aggregate_metrics(t_recs, t_lags)
        tier_metrics[t] = {
            'mean_r':    t_agg['instantaneous']['mean_r'],
            'tau_drv':   t_agg['deriv_xcorr']['peak_lag'],
            'n':         t_agg['instantaneous']['n_valid'],
        }
    m['tier_metrics'] = tier_metrics

    print(f"  [{label}] r={m['mean_r']:.3f}  "
          f"tau_thresh={m['tau_threshold']:.2f}  "
          f"tau_drv={m['tau_drv']}  "
          f"n={m['n_examples']}")

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Sub-experiment A: Independent training
# ══════════════════════════════════════════════════════════════════════════════

def run_subexp_a(train_ds, val_ds, test_ds, device, results_dir,
                 force_retrain=False,
                 epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train C_sort, D_sort, E_sort and measure proprioceptive metrics.
    Returns dict: group_tag → metrics.
    """
    groups    = ['C_sort', 'D_sort', 'E_sort']
    results   = {}

    for g in groups:
        ckpt_path    = os.path.join(results_dir, f"phase19_{g}_model.pt")
        metrics_path = os.path.join(results_dir, f"phase19_{g}_metrics.json")

        if not force_retrain and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            print(f"  [CACHE] {g}: r={m['mean_r']:.3f}  "
                  f"tau_drv={m['tau_drv']}")
            # Still need the full metrics for figures — recompute signals
            # if checkpoint exists but we need examples
            if os.path.exists(ckpt_path):
                model   = make_sorting_model(device)
                model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                                 weights_only=True))
                model.eval()
                m_full  = evaluate_sorting_model(model, test_ds, device,
                                                 label=g)
                m_full.update({k: v for k, v in m.items()
                               if k not in m_full})
                results[g] = m_full
            else:
                results[g] = m
            continue

        print(f"\n{'─'*50}")
        print(f"  Training {g} …")
        model   = make_sorting_model(device)
        loss_fn = make_loss_fn(g)
        model   = train_sorting_model(model, loss_fn, train_ds, val_ds,
                                      device, epochs=epochs,
                                      batch_size=batch_size, label=g)

        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved → {ckpt_path}")

        model.eval()
        m = evaluate_sorting_model(model, test_ds, device, label=g)

        # Save lightweight JSON (no raw signal arrays)
        m_json = {k: v for k, v in m.items()
                  if k not in ('agg', 'examples', 'records')}
        with open(metrics_path, 'w') as f:
            json.dump(m_json, f, indent=2, default=str)

        results[g] = m

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Sub-experiment B: Zero-shot transfer
# ══════════════════════════════════════════════════════════════════════════════

def run_subexp_b(test_ds, device, results_dir, parity_ckpt_path,
                 force_retrain=False):
    """
    Load E_ssm parity backbone into sorting model; evaluate without training.
    """
    metrics_path = os.path.join(results_dir, 'phase19_zeroshot_metrics.json')
    if not force_retrain and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"  [CACHE] zero-shot: r={m['mean_r']:.3f}  "
              f"tau_drv={m['tau_drv']}")
        return m

    print(f"\n{'─'*50}")
    print("  Sub-experiment B: zero-shot transfer …")

    if not os.path.exists(parity_ckpt_path):
        print(f"  SKIP: parity checkpoint not found: {parity_ckpt_path}")
        return None

    model = make_sorting_model(device)
    model = load_parity_backbone(model, parity_ckpt_path, device)
    model.eval()

    m = evaluate_sorting_model(model, test_ds, device, label='zero-shot')
    m_json = {k: v for k, v in m.items()
              if k not in ('agg', 'examples', 'records')}
    with open(metrics_path, 'w') as f:
        json.dump(m_json, f, indent=2, default=str)

    return m


# ══════════════════════════════════════════════════════════════════════════════
# Sub-experiment C: Few-shot adaptation
# ══════════════════════════════════════════════════════════════════════════════

def run_subexp_c(train_ds, val_ds, test_ds, device, results_dir,
                 parity_ckpt_path, force_retrain=False,
                 few_shot_ns=None, epochs=30):
    """
    Fine-tune E_ssm backbone on {100, 500, 1000} sorting examples.
    Returns list of dicts (one per n_examples value).
    """
    if few_shot_ns is None:
        few_shot_ns = FEW_SHOT_NS

    if not os.path.exists(parity_ckpt_path):
        print(f"  SKIP sub-exp C: parity checkpoint not found: {parity_ckpt_path}")
        return []

    results = []
    for n in few_shot_ns:
        tag          = f"fewshot{n}"
        metrics_path = os.path.join(results_dir,
                                    f"phase19_{tag}_metrics.json")

        if not force_retrain and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            print(f"  [CACHE] {tag}: r={m['mean_r']:.3f}  "
                  f"tau_drv={m['tau_drv']}")
            results.append(m)
            continue

        print(f"\n  Sub-exp C: fine-tuning on {n} sorting examples …")
        model   = make_sorting_model(device)
        model   = load_parity_backbone(model, parity_ckpt_path, device)
        loss_fn = make_loss_fn('E_sort')

        # Subset of training data
        fs_ds = Subset(train_ds, list(range(min(n, len(train_ds)))))

        model = train_sorting_model(
            model, loss_fn, fs_ds, val_ds, device,
            epochs=epochs, batch_size=min(32, n), lr=1e-4,
            label=tag)

        model.eval()
        m = evaluate_sorting_model(model, test_ds, device, label=tag)
        m['n_train'] = n

        m_json = {k: v for k, v in m.items()
                  if k not in ('agg', 'examples', 'records')}
        with open(metrics_path, 'w') as f:
            json.dump(m_json, f, indent=2, default=str)

        results.append(m)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: Gradient comparison (parity vs sorting)
# ══════════════════════════════════════════════════════════════════════════════

def plot_gradient_comparison(subexp_a, save_path):
    """
    Two-panel bar chart: C/D/E gradient for parity (left) and sorting (right).
    Each panel shows r and τ_drv.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    parity_groups  = ['C', 'D', 'E_ssm']
    sorting_groups = ['C_sort', 'D_sort', 'E_sort']
    sort_labels    = ['C_sort\nSSM+CE', 'D_sort\nSSM+L_th', 'E_sort\nSSM+halt']

    colors = ['#95a5a6', '#3498db', '#e67e22']
    x      = np.arange(3)
    w      = 0.35

    for ax, (groups, refs, labels, title) in zip(axes, [
        (parity_groups,
         [PARITY_REF[g]['mean_r']  for g in parity_groups],
         ['C\nSSM+CE', 'D\nSSM+L_th', 'E_ssm\nSSM+halt'],
         'Parity Domain (Phase 17b reference)'),
        (sorting_groups,
         [subexp_a.get(g, {}).get('mean_r', float('nan'))
          for g in sorting_groups],
         sort_labels,
         'Sorting Domain (Phase 19)')
    ]):
        tau_vals = ([PARITY_REF[g]['tau_drv'] for g in parity_groups]
                    if 'Parity' in title else
                    [subexp_a.get(g, {}).get('tau_drv', float('nan'))
                     for g in sorting_groups])

        bars1 = ax.bar(x - w / 2, refs, w,
                       color=colors, alpha=0.85, label='r (left axis)', zorder=3)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + w / 2, tau_vals, w,
                        color=colors, alpha=0.45, hatch='//',
                        label='τ_drv (right axis)', zorder=3)

        ax.axhline(0, color='gray', ls='--', lw=0.8)
        ax.axhline(-0.5, color='red', ls=':', lw=1, alpha=0.5,
                   label='r = −0.5 threshold')
        ax2.axhline(0, color='navy', ls=':', lw=0.8, alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Mean r (state entropy vs halt conf.)',
                      fontsize=10, color='#2c3e50')
        ax2.set_ylabel('τ_drv (derivative xcorr peak lag)',
                       fontsize=10, color='#5d6d7e')
        ax.set_ylim(-1.1, 0.3)
        ax2.set_ylim(-4, 5)
        ax.set_title(title, fontsize=11, fontweight='bold')

        # Annotate values
        for bar, val in zip(bars1, refs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val - 0.05, f'{val:.3f}',
                        ha='center', va='top', fontsize=8, fontweight='bold')

        # Legend proxy
        from matplotlib.patches import Patch
        h1 = [Patch(fc=c, alpha=0.85, label=lbl)
              for c, lbl in zip(colors, labels)]
        h2 = [Patch(fc='gray', alpha=0.45, hatch='//', label='τ_drv (hatched)')]
        ax.legend(handles=h1 + h2, fontsize=8, loc='lower right')

    fig.suptitle('Phase 19: Specificity Gradient — Parity vs Sorting\n'
                 'C→D→E progression: reactive → anticipatory coupling',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Transfer and adaptation curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_transfer_curve(zeroshot_m, fewshot_list, esort_m, save_path):
    """
    Line plot: r and τ_drv vs number of training examples (0→100→500→1000→full).
    """
    # Build data points
    points = []
    if zeroshot_m is not None:
        points.append({'n': 0, 'r': zeroshot_m.get('mean_r', np.nan),
                       'tau': zeroshot_m.get('tau_drv', np.nan),
                       'label': 'zero-shot'})
    for m in fewshot_list:
        points.append({'n': m.get('n_train', 0),
                       'r': m.get('mean_r', np.nan),
                       'tau': m.get('tau_drv', np.nan),
                       'label': str(m.get('n_train', ''))})
    if esort_m is not None:
        points.append({'n': 8000, 'r': esort_m.get('mean_r', np.nan),
                       'tau': esort_m.get('tau_drv', np.nan),
                       'label': 'E_sort\n(full)'})

    if len(points) < 2:
        print("  Not enough data for transfer curve, skipping.")
        return

    ns   = [p['n'] for p in points]
    rs   = [p['r'] for p in points]
    taus = [p['tau'] for p in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ns, rs, 'o-', color=BLUE, lw=2.5, ms=8, zorder=3)
    for p in points:
        if not np.isnan(p['r']):
            ax1.annotate(f"{p['r']:.3f}", (p['n'], p['r']),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8)
    ax1.axhline(-0.5, color=RED, ls=':', lw=1.5, alpha=0.7,
                label='r = −0.5 (E_sort success)')
    ax1.axhline(PARITY_REF['E_ssm']['mean_r'], color=GREEN, ls='--', lw=1.5,
                alpha=0.7, label=f"E_ssm parity r={PARITY_REF['E_ssm']['mean_r']}")
    ax1.set_xlabel('Training examples (sorting)', fontsize=11)
    ax1.set_ylabel('Mean r (state entropy vs halt conf.)', fontsize=11)
    ax1.set_title('Proprioceptive Coupling vs Training Size', fontweight='bold')
    ax1.set_xscale('symlog', linthresh=50)
    ax1.set_ylim(-1.1, 0.3)
    ax1.legend(fontsize=9)

    ax2.plot(ns, taus, 's-', color=GREEN, lw=2.5, ms=8, zorder=3)
    for p in points:
        if not np.isnan(p['tau']):
            ax2.annotate(str(int(p['tau'])) if isinstance(p['tau'], (int, float))
                         else '?',
                         (p['n'], p['tau']),
                         textcoords='offset points', xytext=(0, 8),
                         ha='center', fontsize=8)
    ax2.axhline(0,  color='gray', ls='--', lw=0.8, alpha=0.5)
    ax2.axhline(-1, color=RED, ls=':', lw=1.5, alpha=0.7,
                label='τ_drv = −1 (anticipatory)')
    ax2.axhline(PARITY_REF['E_ssm']['tau_drv'], color=BLUE, ls='--', lw=1.5,
                alpha=0.7, label=f"E_ssm parity τ_drv={PARITY_REF['E_ssm']['tau_drv']}")
    ax2.set_xlabel('Training examples (sorting)', fontsize=11)
    ax2.set_ylabel('τ_drv (derivative xcorr peak lag)', fontsize=11)
    ax2.set_title('Anticipatory Lead vs Training Size', fontweight='bold')
    ax2.set_xscale('symlog', linthresh=50)
    ax2.legend(fontsize=9)

    fig.suptitle('Phase 19: Zero-Shot → Few-Shot → Full Transfer Curve',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Tier analysis
# ══════════════════════════════════════════════════════════════════════════════

def plot_tier_analysis(subexp_a, save_path):
    """
    Grouped bar chart: r and τ_drv for each group × tier combination.
    """
    groups = ['C_sort', 'D_sort', 'E_sort']
    tiers  = [0, 1, 2]
    labels = ['T0 (3–4)', 'T1 (5–6)', 'T2 (7–8)']
    colors = ['#95a5a6', '#3498db', '#e67e22']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(tiers))
    w = 0.25

    for gi, (g, col) in enumerate(zip(groups, colors)):
        tm = subexp_a.get(g, {}).get('tier_metrics', {})
        r_vals   = [tm.get(t, {}).get('mean_r',  np.nan) if tm.get(t) else np.nan
                    for t in tiers]
        tau_vals = [tm.get(t, {}).get('tau_drv', np.nan) if tm.get(t) else np.nan
                    for t in tiers]

        offset = (gi - 1) * w
        ax1.bar(x + offset, r_vals,   w, color=col, alpha=0.85,
                label=g, zorder=3)
        ax2.bar(x + offset, tau_vals, w, color=col, alpha=0.85,
                label=g, zorder=3)

    ax1.axhline(0,    color='gray', ls='--', lw=0.8)
    ax1.axhline(-0.5, color=RED,   ls=':',  lw=1.2, alpha=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel('Mean r', fontsize=11)
    ax1.set_title('State Entropy–Halt Coupling by Tier', fontweight='bold')
    ax1.set_ylim(-1.1, 0.3)
    ax1.legend(fontsize=9)

    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.axhline(-1, color=RED,  ls=':',  lw=1.2, alpha=0.6)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('τ_drv (derivative xcorr peak lag)', fontsize=11)
    ax2.set_title('Anticipatory Lead by Tier', fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle('Phase 19 Tier Analysis: Coupling vs Reasoning Depth',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Trajectory gallery (E_sort)
# ══════════════════════════════════════════════════════════════════════════════

def _select_gallery(examples, records, n=9):
    """Select n examples with strongest anticipatory signal."""
    primary, fallback = [], []
    for ex, rec in zip(examples, records):
        r   = rec.get('r', float('nan'))
        lag = rec['threshold_lag']
        rp  = ex['result_pos']
        if rp < 8 or np.isnan(r):
            continue
        if r < -0.5:
            score = abs(r) * max(abs(lag), 0.1)
            if -4 <= lag <= -1:
                primary.append((score, ex, rec))
            else:
                fallback.append((score, ex, rec))

    primary.sort(key=lambda x: x[0], reverse=True)
    fallback.sort(key=lambda x: x[0], reverse=True)
    combined = primary + fallback
    return [(ex, rec) for _, ex, rec in combined[:n]]


def plot_trajectory_gallery(esort_m, save_path):
    """3×3 grid of E_sort individual examples."""
    if esort_m is None or 'examples' not in esort_m:
        print("  Gallery: no E_sort examples available, skipping.")
        return

    examples = esort_m['examples']
    records  = esort_m['records']
    pairs    = _select_gallery(examples, records, n=9)

    if not pairs:
        print("  Gallery: no suitable examples found, skipping.")
        return

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (ex, rec) in enumerate(pairs):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        se  = ex['state_entropy']
        hc  = ex['halt_confidence']
        L   = len(se)
        pos = np.arange(L)

        sc  = rec['sc_pos']
        hr  = rec['hr_pos']
        r   = rec['r']
        lag = rec['threshold_lag']

        se_scale = se.max() or 1.0
        se_n     = se / se_scale

        ax1.plot(pos, se_n, color=BLUE, lw=2.0, zorder=3)
        ax2.plot(pos, hc,   color=RED,  lw=2.0, ls=':', zorder=3)

        if hr < L:
            ax2.axvline(hr, color=RED,  ls='-',  lw=1.8, alpha=0.75)
        if sc < L:
            ax1.axvline(sc, color=BLUE, ls='--', lw=1.8, alpha=0.75)

        if hr < L and sc < L and hr != sc:
            y_bkt = 1.08
            ax1.annotate('', xy=(sc, y_bkt), xytext=(hr, y_bkt),
                         arrowprops=dict(arrowstyle='<->',
                                         color='#2c3e50', lw=1.3,
                                         mutation_scale=12))
            ax1.text((hr + sc) / 2, y_bkt + 0.05, f'τ = {lag:+d}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax1.set_xlim(-0.5, L + 0.5)
        ax1.set_ylim(-0.05, 1.25)
        ax2.set_ylim(-0.05, 1.25)
        ax1.set_xlabel('Position', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        ax1.set_title(f'r={r:.3f},  lag={lag:+d}',
                      fontsize=9, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=BLUE)
        ax2.tick_params(axis='y', labelcolor=RED)
        if idx % ncols == 0:
            ax1.set_ylabel('State Entropy (norm.)', fontsize=8, color=BLUE)
        if idx % ncols == ncols - 1:
            ax2.set_ylabel('Halt Conf.', fontsize=8, color=RED)

    legend_handles = [
        mlines.Line2D([], [], color=BLUE, lw=2.0,
                      label='State Entropy (norm.)'),
        mlines.Line2D([], [], color=RED,  lw=2.0, ls=':',
                      label='Halt Confidence'),
        mlines.Line2D([], [], color=BLUE, lw=1.8, ls='--', alpha=0.75,
                      label='State collapse (SE = 50%)'),
        mlines.Line2D([], [], color=RED,  lw=1.8, ls='-',  alpha=0.75,
                      label='Halt rise (HC > 0.5)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               fontsize=8.5, bbox_to_anchor=(0.5, 0.0), framealpha=0.92)

    for idx in range(len(pairs), nrows * ncols):
        axes[idx].set_visible(False)

    fig.suptitle(
        'Phase 19: E_sort Trajectory Gallery (Sorting Domain)\n'
        'τ = halt rise − state collapse  (negative = anticipatory)',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5: Derivative xcorr comparison (parity vs sorting)
# ══════════════════════════════════════════════════════════════════════════════

def plot_xcorr_comparison(esort_m, parity_xcorr_path, save_path):
    """
    Side-by-side derivative xcorr: E_ssm parity (left) vs E_sort sorting (right).
    """
    # Load parity xcorr from Phase 17b results
    parity_drv = None
    if os.path.exists(parity_xcorr_path):
        with open(parity_xcorr_path) as f:
            p17 = json.load(f)
        if 'E_ssm' in p17:
            parity_drv = p17['E_ssm'].get('deriv_xcorr')

    if parity_drv is None or esort_m is None or 'agg' not in esort_m:
        print("  XCorr comparison: missing data, skipping.")
        return

    sort_drv = esort_m['agg'].get('deriv_xcorr')
    if sort_drv is None:
        print("  XCorr comparison: no sorting xcorr data, skipping.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        (parity_drv, BLUE,  'E_ssm (Parity)',
         f"τ_drv = {parity_drv['peak_lag']:+d}  "
         f"({parity_drv['peak_val']:.3f})"),
        (sort_drv,   GREEN, 'E_sort (Sorting)',
         f"τ_drv = {sort_drv['peak_lag']:+d}  "
         f"({sort_drv['peak_val']:.3f})"),
    ]

    for ax, (drv, color, domain_label, peak_label) in zip(axes, panels):
        lags = np.array(drv['lags'])
        mean = np.array(drv['mean'])
        std  = np.array(drv['std'])
        peak = drv['peak_lag']

        bars = ax.bar(lags, mean, color=color, alpha=0.65, width=0.6,
                      zorder=3, label='Mean xcorr')
        ax.errorbar(lags, mean, yerr=std, fmt='none',
                    color='#2c3e50', capsize=4, lw=1.5, zorder=4)

        peak_idx = np.where(lags == peak)[0]
        if len(peak_idx):
            bars[peak_idx[0]].set_alpha(0.95)
            bars[peak_idx[0]].set_edgecolor('#1a7a44')
            bars[peak_idx[0]].set_linewidth(2)

        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
        ax.axvline(0, color='gray', ls=':', lw=1, alpha=0.4)
        ax.axvline(peak, color='#1a7a44', ls='--', lw=1.8, alpha=0.85,
                   label=peak_label)

        ax.set_xlabel('Lag τ (steps)\n(positive = dHC/dt leads −dSE/dt)',
                      fontsize=11)
        ax.set_ylabel('Normalised derivative xcorr', fontsize=11)
        ax.set_title(f'Derivative Cross-Correlation\n{domain_label}',
                     fontsize=11, fontweight='bold')
        ax.set_xticks(lags)
        ax.legend(fontsize=10)

    fig.suptitle(
        'Phase 19: Derivative XCorr — Parity vs Sorting\n'
        'τ > 0: dHC/dt leads −dSE/dt (anticipatory)',
        fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(subexp_a, zeroshot_m, fewshot_list):
    """Build and print the full comparison table."""
    table = {'parity_reference': PARITY_REF, 'sorting': {}, 'transfer': {}}

    for g in ['C_sort', 'D_sort', 'E_sort']:
        m = subexp_a.get(g)
        if m is None:
            continue
        table['sorting'][g] = {
            'accuracy_full':   m.get('accuracy_full',  None),
            'accuracy_first':  m.get('accuracy_first', None),
            'mean_r':          m.get('mean_r',         None),
            'tau_threshold':   m.get('tau_threshold',  None),
            'tau_drv':         m.get('tau_drv',        None),
            'frac_negative':   m.get('frac_negative',  None),
            'frac_significant': m.get('frac_significant', None),
            'n_examples':      m.get('n_examples',     None),
            'tier_metrics':    m.get('tier_metrics',   {}),
        }

    if zeroshot_m:
        table['transfer']['zero_shot'] = {
            'n_train': 0,
            'mean_r':  zeroshot_m.get('mean_r'),
            'tau_drv': zeroshot_m.get('tau_drv'),
        }
    for m in fewshot_list:
        n = m.get('n_train', 0)
        table['transfer'][f'few_shot_{n}'] = {
            'n_train': n,
            'mean_r':  m.get('mean_r'),
            'tau_drv': m.get('tau_drv'),
        }

    return table


def print_summary(table):
    """Print a formatted comparison table to stdout."""
    print()
    print('═' * 70)
    print('  PHASE 19 COMPARISON TABLE')
    print('═' * 70)
    print()
    print('  ── Parity Reference (Phase 17b) ──────────────────────────────')
    print(f"  {'Group':<12s}  {'r':>8s}  {'τ_drv':>7s}  {'Label':<18s}")
    print('  ' + '─' * 52)
    for g, ref in PARITY_REF.items():
        print(f"  {g:<12s}  {ref['mean_r']:>8.3f}  {ref['tau_drv']:>7d}  "
              f"{ref['label']:<18s}")

    print()
    print('  ── Sorting (Phase 19, Sub-exp A) ─────────────────────────────')
    print(f"  {'Group':<12s}  {'acc_full':>8s}  {'r':>8s}  "
          f"{'τ_threshold':>12s}  {'τ_drv':>7s}  {'n':>6s}")
    print('  ' + '─' * 64)
    for g, m in table.get('sorting', {}).items():
        r   = m.get('mean_r');       r_s   = f"{r:.3f}"   if r   is not None else '   N/A'
        af  = m.get('accuracy_full'); af_s  = f"{af:.3f}"  if af  is not None else '   N/A'
        tt  = m.get('tau_threshold'); tt_s  = f"{tt:.2f}"  if tt  is not None else '  N/A'
        td  = m.get('tau_drv');       td_s  = str(td)       if td  is not None else ' N/A'
        n   = m.get('n_examples');    n_s   = str(n)        if n   is not None else ' N/A'
        print(f"  {g:<12s}  {af_s:>8s}  {r_s:>8s}  "
              f"{tt_s:>12s}  {td_s:>7s}  {n_s:>6s}")

    # Specificity gradient test
    print()
    sort_m = table.get('sorting', {})
    c_r  = sort_m.get('C_sort', {}).get('mean_r',  float('nan'))
    d_r  = sort_m.get('D_sort', {}).get('mean_r',  float('nan'))
    e_r  = sort_m.get('E_sort', {}).get('mean_r',  float('nan'))
    c_td = sort_m.get('C_sort', {}).get('tau_drv', float('nan'))
    d_td = sort_m.get('D_sort', {}).get('tau_drv', float('nan'))
    e_td = sort_m.get('E_sort', {}).get('tau_drv', float('nan'))

    gradient_ok_r  = (not np.isnan(c_r) and not np.isnan(e_r) and c_r > e_r)
    e_anticipatory = (not np.isnan(e_r) and e_r < -0.5 and
                      not np.isnan(e_td) and e_td <= -1)

    print('  ── Specificity Gradient Test ─────────────────────────────────')
    print(f"  C_sort → E_sort r gradient: {c_r:.3f} → {e_r:.3f}  "
          f"{'PASS' if gradient_ok_r else 'FAIL'}")
    print(f"  E_sort anticipatory (r<-0.5, τ_drv≤-1): "
          f"{'PASS' if e_anticipatory else 'FAIL'}")

    if e_anticipatory and gradient_ok_r:
        print('  >>> OUTCOME 1 or 2: Full / Partial generalisation <<<')
    elif e_anticipatory:
        print('  >>> OUTCOME 2: Partial generalisation <<<')
    else:
        print('  >>> OUTCOME 3 or 4: Domain-specific or not transferable <<<')

    # Transfer summary
    tr = table.get('transfer', {})
    if tr:
        print()
        print('  ── Transfer Curve (Sub-exp B/C) ──────────────────────────────')
        print(f"  {'n_train':>8s}  {'r':>8s}  {'τ_drv':>7s}")
        print('  ' + '─' * 30)
        for key in sorted(tr.keys(), key=lambda k: tr[k].get('n_train', 0)):
            m  = tr[key]
            r  = m.get('mean_r', float('nan'))
            td = m.get('tau_drv', float('nan'))
            print(f"  {m.get('n_train',0):>8d}  "
                  f"{r:>8.3f}  {str(td):>7s}")

    print()
    print('═' * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Phase 19: Cross-Domain Proprioception')
    parser.add_argument('--results-dir',   default='results')
    parser.add_argument('--figures-dir',   default='figures')
    parser.add_argument('--device',        default=None)
    parser.add_argument('--epochs',        type=int,   default=EPOCHS)
    parser.add_argument('--batch-size',    type=int,   default=BATCH_SIZE)
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--skip-subexp-a', action='store_true')
    parser.add_argument('--skip-subexp-b', action='store_true')
    parser.add_argument('--skip-subexp-c', action='store_true')
    parser.add_argument('--skip-figures',  action='store_true')
    parser.add_argument('--parity-ckpt',
                        default='results/group_E_ssm_model.pt',
                        help='Path to trained parity E_ssm checkpoint')
    parser.add_argument('--phase17b-results',
                        default='results/phase17b_results.json',
                        help='Phase 17b results for parity xcorr reference')
    args = parser.parse_args()

    device = (torch.device(args.device) if args.device
              else get_device())
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print('=' * 70)
    print('Phase 19: Cross-Domain Proprioception')
    print('=' * 70)
    print(f'Device:        {device}')
    print(f'Sorting vocab: {SORT_VOCAB_SIZE} tokens')
    print(f'Model:         PNA_SSM d_model={D_MODEL} d_state={D_STATE}'
          f' n_layers={N_LAYERS}')
    print(f'Epochs:        {args.epochs}')

    # ── Datasets ─────────────────────────────────────────────────────────────
    print('\nCreating sorting datasets …')
    train_ds, val_ds, test_ds = create_sorting_datasets()
    print(f'  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')

    # Quick sanity check on a single example
    ex0 = train_ds.examples[0]
    print(f'  Example 0: n_sym={ex0["n_symbols"]}  '
          f'tier={ex0["tier"]}  '
          f'result_pos={ex0["result_pos"]}  '
          f'seq_len={len(ex0["tokens"])}')

    # ── Sub-experiment A ──────────────────────────────────────────────────────
    subexp_a = {}
    if not args.skip_subexp_a:
        print(f'\n{"="*70}')
        print('SUB-EXPERIMENT A: Independent Training')
        print(f'{"="*70}')
        subexp_a = run_subexp_a(
            train_ds, val_ds, test_ds, device,
            args.results_dir,
            force_retrain=args.force_retrain,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    # ── Sub-experiment B ──────────────────────────────────────────────────────
    zeroshot_m = None
    if not args.skip_subexp_b:
        print(f'\n{"="*70}')
        print('SUB-EXPERIMENT B: Zero-Shot Transfer')
        print(f'{"="*70}')
        zeroshot_m = run_subexp_b(
            test_ds, device, args.results_dir,
            parity_ckpt_path=args.parity_ckpt,
            force_retrain=args.force_retrain,
        )

    # ── Sub-experiment C ──────────────────────────────────────────────────────
    fewshot_list = []
    if not args.skip_subexp_c:
        print(f'\n{"="*70}')
        print('SUB-EXPERIMENT C: Few-Shot Adaptation')
        print(f'{"="*70}')
        fewshot_list = run_subexp_c(
            train_ds, val_ds, test_ds, device,
            args.results_dir,
            parity_ckpt_path=args.parity_ckpt,
            force_retrain=args.force_retrain,
            epochs=30,
        )

    # ── Comparison table ─────────────────────────────────────────────────────
    table = build_comparison_table(subexp_a, zeroshot_m, fewshot_list)
    print_summary(table)

    table_path = os.path.join(args.results_dir, 'phase19_comparison_table.json')
    with open(table_path, 'w') as f:
        json.dump(table, f, indent=2, default=str)
    print(f'\n  Saved {table_path}')

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.skip_figures:
        print(f'\n{"="*70}')
        print('FIGURES')
        print(f'{"="*70}')

        esort_m = subexp_a.get('E_sort')

        plot_gradient_comparison(
            subexp_a,
            os.path.join(args.figures_dir, 'fig_p19_1_gradient_comparison.png'))

        plot_transfer_curve(
            zeroshot_m, fewshot_list, esort_m,
            os.path.join(args.figures_dir, 'fig_p19_2_transfer_curve.png'))

        plot_tier_analysis(
            subexp_a,
            os.path.join(args.figures_dir, 'fig_p19_3_tier_analysis.png'))

        plot_trajectory_gallery(
            esort_m,
            os.path.join(args.figures_dir, 'fig_p19_4_trajectory_gallery.png'))

        plot_xcorr_comparison(
            esort_m,
            args.phase17b_results,
            os.path.join(args.figures_dir, 'fig_p19_5_xcorr_comparison.png'))

    print('\nPhase 19 complete.')


if __name__ == '__main__':
    main()
