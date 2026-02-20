"""
Phase 18b: Focused Grid — Transition from Simultaneous to Anticipatory Coupling

Motivation: Phase 18 revealed a sharp dissociation at (α=0.01, β=0.05):
  r = -0.789 (strong anti-correlation) but τ_drv ≈ 0 (simultaneous, NOT anticipatory)
  Compare E_ssm (α=0.05, β=0.10): r=-0.836, τ_drv≈-2 (both correlated AND anticipatory)

The transition from simultaneous to anticipatory coupling occurs somewhere in
  α ∈ [0.01, 0.05] × β ∈ [0.05, 0.10]

This script:
  1. Trains a 4×4 focused grid:
       α ∈ {0.01, 0.02, 0.03, 0.05} × β ∈ {0.05, 0.07, 0.10, 0.15}
  2. Reuses Phase 18 checkpoints where available (4 cache hits)
  3. Adds Measure C: Gradient Disparity
       gradient_disparity = argmax(dHC/dt) − argmax(−dSE/dt)
       Negative = halt rise peak precedes state collapse peak (anticipatory)
       Zero     = simultaneous (halt and state gradients peak together)
  4. Generates 5 figures including the dissociation scatter and transition boundary

Figures
-------
  figures/fig_p18b_1_r_heatmap.png
  figures/fig_p18b_2_tau_heatmap.png
  figures/fig_p18b_3_disparity_heatmap.png
  figures/fig_p18b_4_dissociation_scatter.png
  figures/fig_p18b_5_transition_boundary.png

Outputs
-------
  results/phase18b_grid_{key}_model.pt          (only for new models)
  results/phase18b_grid_{key}_metrics.json      (all 16 models)
  results/phase18b_response_surface.json
"""

import os
import sys
import copy
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE, VOCAB
from models import PNA_SSM, count_parameters
from losses import ThermodynamicLoss, CrossEntropyLoss
from train import (
    get_device, get_cosine_schedule,
    train_one_epoch, evaluate, compute_halt_f1,
)
from entropy_halt_correlation import (
    AnswerProbe, train_probe, evaluate_probe_accuracy,
    get_parity_indices,
)
from phase17_proprioception_repro import (
    extract_signals, compute_all_metrics, aggregate_metrics,
)

# ── Grid specification ────────────────────────────────────────
ALPHA_VALUES = [0.01, 0.02, 0.03, 0.05]
BETA_VALUES  = [0.05, 0.07, 0.10, 0.15]

# Phase 18 checkpoint directory (for cache hits)
PHASE18_CKPT_TEMPLATE = "results/phase18_grid_{key}_model.pt"

# Reference points
E_SSM_R   = -0.836
E_SSM_TAU_DRV = -2.032   # from Phase 17
D_R       = -0.725
D_TAU_DRV = -2.0         # approximate

# Training hyper-parameters (identical to Phase 18)
EPOCHS     = 50
BATCH_SIZE = 32
LR         = 3e-4
PATIENCE   = 10
WARMUP     = 100
PROBE_EPOCHS = 10
PROBE_LR     = 1e-3

# Architecture (same as Phase 18 grid)
D_MODEL   = 512
D_STATE   = 16
N_LAYERS  = 6
MAX_SEQ   = 256

ACCURACY_THRESHOLD = 0.95


# ============================================================
# Measure C: Gradient Disparity
# ============================================================

def compute_gradient_disparity(examples, min_len=5):
    """
    Gradient Disparity (Measure C): per-example difference in peak positions
    of fastest halt-confidence rise versus fastest state-entropy collapse.

    gradient_disparity = argmax(dHC/dt) − argmax(−dSE/dt)
      < 0 → halt rise peak PRECEDES state collapse peak  (anticipatory)
      = 0 → simultaneous
      > 0 → state collapse precedes halt rise peak        (lagging)

    Also computes:
      frac_anticipatory  — fraction of examples with disparity < 0
      collapse_strength  — mean peak magnitude of entropy collapse rate
      rise_strength      — mean peak magnitude of halt rise rate
      magnitude_ratio    — mean(rise_strength / collapse_strength)
    """
    records = []
    for ex in examples:
        se = ex['state_entropy']
        hc = ex['halt_confidence']
        if len(se) < min_len:
            continue

        dse = np.diff(se)   # se[t+1] - se[t]  (negative = collapsing)
        dhc = np.diff(hc)   # hc[t+1] - hc[t]  (positive = rising)

        if len(dse) == 0:
            continue

        # Fastest collapse: largest decrease in SE = argmax(-dSE)
        collapse_pos = int(np.argmax(-dse))
        # Fastest halt rise: argmax(dHC)
        rise_pos = int(np.argmax(dhc))

        records.append({
            'disparity':        rise_pos - collapse_pos,
            'collapse_pos':     collapse_pos,
            'rise_pos':         rise_pos,
            'collapse_strength': float(-dse[collapse_pos]),
            'rise_strength':    float(dhc[rise_pos]),
        })

    if not records:
        return None

    disp    = np.array([r['disparity']        for r in records])
    c_str   = np.array([r['collapse_strength'] for r in records])
    r_str   = np.array([r['rise_strength']     for r in records])

    return {
        'mean':               float(np.mean(disp)),
        'median':             float(np.median(disp)),
        'std':                float(np.std(disp)),
        'frac_anticipatory':  float(np.mean(disp < 0)),
        'frac_simultaneous':  float(np.mean(disp == 0)),
        'mean_collapse_strength': float(np.mean(c_str)),
        'mean_rise_strength':     float(np.mean(r_str)),
        'mean_magnitude_ratio':   float(np.mean(r_str / (c_str + 1e-8))),
        'n':                  len(records),
    }


# ============================================================
# Model / loss helpers
# ============================================================

def _key(alpha, beta):
    return f"a{alpha:.2f}_b{beta:.2f}".replace('.', 'p')


def make_model(device):
    m = PNA_SSM(VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
                d_state=D_STATE, max_seq_len=MAX_SEQ)
    return m.to(device)


def make_loss_fn(alpha, beta):
    if alpha == 0.0 and beta == 0.0:
        return CrossEntropyLoss(pad_token_id=VOCAB['<PAD>'])
    return ThermodynamicLoss(alpha=alpha, beta=beta, gamma=0.0,
                             pad_token_id=VOCAB['<PAD>'])


def phase18_ckpt(alpha, beta):
    """Return path to Phase 18 checkpoint if it exists, else None."""
    k = _key(alpha, beta)
    p = PHASE18_CKPT_TEMPLATE.format(key=k)
    return p if os.path.exists(p) else None


# ============================================================
# Training
# ============================================================

def train_model(model, loss_fn, train_ds, val_ds, device,
                epochs=EPOCHS, batch_size=BATCH_SIZE, label=""):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR,
                                    betas=(0.9, 0.999), weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler   = get_cosine_schedule(optimizer, total_steps, warmup_steps=WARMUP)

    best_val    = float('inf')
    best_state  = None
    patience_ct = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, governor=None,
            optimizer=optimizer, scheduler=scheduler, device=device, group='D',
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        vl = val_metrics['total_loss']
        improved = ""
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            patience_ct = 0
            improved = " *"
        else:
            patience_ct += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] epoch {epoch:3d} | "
                  f"tr={train_metrics['total_loss']:.4f} "
                  f"val={vl:.4f} acc={val_metrics['accuracy']:.3f} "
                  f"({elapsed:.1f}s){improved}")

        if patience_ct >= PATIENCE:
            print(f"    [{label}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# Per-model evaluation (Phase 17 + Measure C)
# ============================================================

def evaluate_model(model, loss_fn, test_ds, train_ds,
                   parity_test, parity_train, device,
                   batch_size=BATCH_SIZE):
    """
    Full evaluation: task accuracy, halt F1, probe accuracy,
    Phase 17 metrics (r, τ_threshold, τ_derivative), Gradient Disparity (C).
    """
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    accuracy = test_metrics['accuracy']

    halt_f1 = compute_halt_f1(model, test_loader, device)

    # Probe
    probe = AnswerProbe(D_MODEL)
    probe = train_probe(model, probe, train_ds, parity_train, device,
                        probe_source='d_model', epochs=PROBE_EPOCHS,
                        batch_size=batch_size, lr=PROBE_LR)
    probe_acc = evaluate_probe_accuracy(model, probe, test_ds, parity_test, device,
                                        probe_source='d_model')

    # Phase 17 signals
    examples = extract_signals(model, probe, test_ds, parity_test, device,
                               batch_size=batch_size)
    records, lag_range = compute_all_metrics(examples)
    agg = aggregate_metrics(records, lag_range)

    inst = agg['instantaneous']
    thr  = agg['threshold_lag']

    drv_lags = [r['deriv_peak_lag'] for r in records if r['deriv_peak_lag'] is not None]
    tau_drv_mean   = float(np.mean(drv_lags))   if drv_lags else float('nan')
    tau_drv_median = float(np.median(drv_lags)) if drv_lags else float('nan')

    # Measure C: Gradient Disparity
    gd = compute_gradient_disparity(examples)

    return {
        'accuracy':              float(accuracy),
        'halt_f1':               float(halt_f1),
        'probe_acc':             float(probe_acc),
        'mean_r':                inst['mean_r'],
        'median_r':              inst['median_r'],
        'std_r':                 inst['std_r'],
        'frac_significant':      inst['fraction_significant'],
        'tau_threshold_mean':    thr['mean'],
        'tau_threshold_median':  thr['median'],
        'tau_derivative_mean':   tau_drv_mean,
        'tau_derivative_median': tau_drv_median,
        'gradient_disparity':    gd,
        'n_examples':            agg['n_examples'],
    }


# ============================================================
# Main sweep
# ============================================================

def run_sweep(train_ds, val_ds, test_ds, parity_train, parity_test,
              device, results_dir, force_retrain=False,
              epochs=EPOCHS, batch_size=BATCH_SIZE):
    surface = {}
    total = len(ALPHA_VALUES) * len(BETA_VALUES)
    done  = 0

    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            done += 1
            key          = _key(alpha, beta)
            ckpt_18b     = os.path.join(results_dir, f"phase18b_grid_{key}_model.pt")
            metrics_18b  = os.path.join(results_dir, f"phase18b_grid_{key}_metrics.json")

            print(f"\n[{done}/{total}] α={alpha}  β={beta}  ({key})")

            # Load cached Phase 18b metrics
            if not force_retrain and os.path.exists(metrics_18b):
                with open(metrics_18b) as f:
                    m = json.load(f)
                gd_mean = m.get('gradient_disparity', {})
                gd_mean = gd_mean.get('mean', float('nan')) if gd_mean else float('nan')
                print(f"  [CACHE] acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
                      f"τ_drv={m['tau_derivative_mean']:.3f}  gd={gd_mean:.3f}")
                surface[key] = {'alpha': alpha, 'beta': beta, **m}
                continue

            # Determine checkpoint source
            p18_ckpt = phase18_ckpt(alpha, beta)
            model = make_model(device)
            loss_fn = make_loss_fn(alpha, beta)

            if p18_ckpt and not force_retrain:
                print(f"  [PHASE18 CKPT] loading {p18_ckpt}")
                model.load_state_dict(
                    torch.load(p18_ckpt, map_location=device, weights_only=True)
                )
                model.eval()
            elif os.path.exists(ckpt_18b) and not force_retrain:
                print(f"  [18b CKPT] loading {ckpt_18b}")
                model.load_state_dict(
                    torch.load(ckpt_18b, map_location=device, weights_only=True)
                )
                model.eval()
            else:
                print(f"  Training (α={alpha}, β={beta})...")
                model = train_model(model, loss_fn, train_ds, val_ds, device,
                                    epochs=epochs, batch_size=batch_size,
                                    label=f"α={alpha} β={beta}")
                torch.save(model.state_dict(), ckpt_18b)
                print(f"  Saved → {ckpt_18b}")

            # Evaluate
            model.eval()
            m = evaluate_model(model, loss_fn, test_ds, train_ds,
                               parity_test, parity_train, device,
                               batch_size=batch_size)
            m['alpha'] = alpha
            m['beta']  = beta

            gd = m.get('gradient_disparity') or {}
            gd_mean = gd.get('mean', float('nan')) if gd else float('nan')
            print(f"  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
                  f"τ_drv={m['tau_derivative_mean']:.3f}  "
                  f"gd={gd_mean:.3f}  "
                  f"gd_frac_ant={gd.get('frac_anticipatory', float('nan')):.3f}")

            with open(metrics_18b, 'w') as f:
                json.dump(m, f, indent=2, default=str)

            surface[key] = {'alpha': alpha, 'beta': beta, **m}

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return surface


# ============================================================
# Figures
# ============================================================

def _grid_arrays(surface):
    """Extract 2-D arrays indexed [ai, bi] for plotting."""
    A, B = ALPHA_VALUES, BETA_VALUES
    r_arr   = np.full((len(A), len(B)), np.nan)
    tau_arr = np.full((len(A), len(B)), np.nan)
    gd_arr  = np.full((len(A), len(B)), np.nan)
    fa_arr  = np.full((len(A), len(B)), np.nan)  # frac_anticipatory
    acc_arr = np.full((len(A), len(B)), np.nan)
    ok_arr  = np.zeros((len(A), len(B)), dtype=bool)

    for ai, alpha in enumerate(A):
        for bi, beta in enumerate(B):
            k = _key(alpha, beta)
            if k not in surface:
                continue
            m = surface[k]
            acc_arr[ai, bi] = m['accuracy']
            ok_arr[ai, bi]  = m['accuracy'] >= ACCURACY_THRESHOLD
            r_arr[ai, bi]   = m['mean_r']
            tau_arr[ai, bi] = m['tau_derivative_mean']
            gd = m.get('gradient_disparity') or {}
            if gd:
                gd_arr[ai, bi] = gd.get('mean', np.nan)
                fa_arr[ai, bi] = gd.get('frac_anticipatory', np.nan)

    return r_arr, tau_arr, gd_arr, fa_arr, acc_arr, ok_arr


def _heatmap(ax, data, ok_arr, alpha_labels, beta_labels, title, cmap,
             vmin=None, vmax=None, fmt='{:.3f}', ref_ai=None, ref_bi=None):
    """Generic 4×4 heatmap with failure hatching and reference star."""
    vmin = vmin if vmin is not None else np.nanmin(data)
    vmax = vmax if vmax is not None else np.nanmax(data)

    im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   origin='lower')

    for ai in range(len(ALPHA_VALUES)):
        for bi in range(len(BETA_VALUES)):
            if not ok_arr[ai, bi]:
                ax.add_patch(plt.Rectangle(
                    (bi - 0.5, ai - 0.5), 1, 1,
                    fill=True, facecolor='#e0e0e0', edgecolor='gray',
                    hatch='///', linewidth=0.5, zorder=2,
                ))
            val = data[ai, bi]
            if not np.isnan(val):
                text = fmt.format(val)
                # Choose text color for readability
                norm_val = (val - vmin) / max(vmax - vmin, 1e-9)
                fc = 'white' if norm_val < 0.35 or norm_val > 0.75 else 'black'
                ax.text(bi, ai, text, ha='center', va='center',
                        fontsize=9, color=fc, zorder=3, fontweight='bold')

    if ref_ai is not None and ref_bi is not None:
        ax.plot(ref_bi, ref_ai, 'y*', markersize=14, zorder=4)

    ax.set_xticks(range(len(BETA_VALUES)))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels(alpha_labels)
    ax.set_xlabel('β (halt loss weight)', fontsize=11)
    ax.set_ylabel('α (energy penalty)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')

    return im


def _ref_indices():
    """Return (ai, bi) for the E_ssm reference point (α=0.05, β=0.10)."""
    try:
        ai = ALPHA_VALUES.index(0.05)
        bi = BETA_VALUES.index(0.10)
        return ai, bi
    except ValueError:
        return None, None


def fig1_r_heatmap(surface, save_path):
    r_arr, _, _, _, acc_arr, ok_arr = _grid_arrays(surface)
    fig, ax = plt.subplots(figsize=(7, 6))
    ai_ref, bi_ref = _ref_indices()
    im = _heatmap(ax, r_arr, ok_arr,
                  [str(a) for a in ALPHA_VALUES],
                  [str(b) for b in BETA_VALUES],
                  'Focused Grid: mean Pearson r(α, β)\n(★ = E_ssm reference α=0.05, β=0.10)',
                  cmap='RdBu', vmin=-1.0, vmax=0.5,
                  ref_ai=ai_ref, ref_bi=bi_ref)
    plt.colorbar(im, ax=ax, label='mean r')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig2_tau_heatmap(surface, save_path):
    _, tau_arr, _, _, acc_arr, ok_arr = _grid_arrays(surface)
    fig, ax = plt.subplots(figsize=(7, 6))
    ai_ref, bi_ref = _ref_indices()

    vmin = min(float(np.nanmin(tau_arr)), -3.0) if not np.all(np.isnan(tau_arr)) else -3.0
    vmax = max(float(np.nanmax(tau_arr)),  1.0) if not np.all(np.isnan(tau_arr)) else  1.0

    im = _heatmap(ax, tau_arr, ok_arr,
                  [str(a) for a in ALPHA_VALUES],
                  [str(b) for b in BETA_VALUES],
                  'Focused Grid: mean τ_derivative(α, β)\n'
                  '(negative = halt gradient leads state gradient)',
                  cmap='coolwarm_r', vmin=vmin, vmax=vmax,
                  fmt='{:.2f}',
                  ref_ai=ai_ref, ref_bi=bi_ref)
    plt.colorbar(im, ax=ax, label='τ_derivative (steps)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig3_disparity_heatmap(surface, save_path):
    _, _, gd_arr, fa_arr, acc_arr, ok_arr = _grid_arrays(surface)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    ai_ref, bi_ref = _ref_indices()
    alpha_labels = [str(a) for a in ALPHA_VALUES]
    beta_labels  = [str(b) for b in BETA_VALUES]

    vmin_gd = min(float(np.nanmin(gd_arr)), -3.0) if not np.all(np.isnan(gd_arr)) else -3.0
    vmax_gd = max(float(np.nanmax(gd_arr)),  1.0) if not np.all(np.isnan(gd_arr)) else  1.0

    im1 = _heatmap(ax1, gd_arr, ok_arr, alpha_labels, beta_labels,
                   'Gradient Disparity (C) — mean\n'
                   'argmax(dHC/dt) − argmax(−dSE/dt)',
                   cmap='coolwarm_r', vmin=vmin_gd, vmax=vmax_gd,
                   fmt='{:.2f}',
                   ref_ai=ai_ref, ref_bi=bi_ref)
    plt.colorbar(im1, ax=ax1, label='gradient disparity (steps)')

    im2 = _heatmap(ax2, fa_arr, ok_arr, alpha_labels, beta_labels,
                   'Fraction Anticipatory (gd < 0)\n'
                   'per-example halt-leads-state fraction',
                   cmap='RdYlGn', vmin=0.0, vmax=1.0,
                   fmt='{:.2f}',
                   ref_ai=ai_ref, ref_bi=bi_ref)
    plt.colorbar(im2, ax=ax2, label='fraction anticipatory')

    fig.suptitle('Measure C: Gradient Disparity — Focused 4×4 Grid',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig4_dissociation_scatter(surface, save_path):
    """
    Scatter: r vs τ_drv for all 16 models, colored by gradient disparity.
    This is the key diagnostic plot showing the simultaneous vs anticipatory regimes.
    """
    r_vals, tau_vals, gd_vals, labels = [], [], [], []

    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            k = _key(alpha, beta)
            if k not in surface:
                continue
            m = surface[k]
            if m['accuracy'] < ACCURACY_THRESHOLD:
                continue
            r_vals.append(m['mean_r'])
            tau_vals.append(m['tau_derivative_mean'])
            gd = m.get('gradient_disparity') or {}
            gd_vals.append(gd.get('mean', np.nan) if gd else float('nan'))
            labels.append(f"α={alpha}\nβ={beta}")

    if not r_vals:
        print("  No data for dissociation scatter (no models passed accuracy filter)")
        return

    r_arr   = np.array(r_vals)
    tau_arr = np.array(tau_vals)
    gd_arr  = np.array(gd_vals)

    fig, ax = plt.subplots(figsize=(9, 7))

    vmin_c = np.nanmin(gd_arr) if not np.all(np.isnan(gd_arr)) else -3
    vmax_c = np.nanmax(gd_arr) if not np.all(np.isnan(gd_arr)) else 3
    norm = mcolors.TwoSlopeNorm(vmin=vmin_c, vcenter=0.0, vmax=vmax_c)

    sc = ax.scatter(r_arr, tau_arr, c=gd_arr, cmap='coolwarm_r', norm=norm,
                    s=120, zorder=3, edgecolors='k', linewidths=0.6)
    plt.colorbar(sc, ax=ax, label='Gradient Disparity (C) — mean steps')

    # Label each point
    for r, tau, lab in zip(r_arr, tau_arr, labels):
        ax.annotate(lab, (r, tau), textcoords='offset points',
                    xytext=(6, 4), fontsize=7.5, color='#333333')

    # Reference points
    ax.scatter([E_SSM_R], [E_SSM_TAU_DRV], marker='*', s=300,
               color='gold', edgecolors='k', linewidths=0.8, zorder=5,
               label=f'E_ssm ref  r={E_SSM_R:.3f}  τ_drv={E_SSM_TAU_DRV:.2f}')

    # Quadrant lines
    ax.axhline(0, color='gray', linestyle=':', linewidth=1.0)
    ax.axvline(0, color='gray', linestyle=':', linewidth=1.0)

    # Quadrant labels
    ax.text(0.02, 0.98, 'Correlated\n& anticipatory', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='darkgreen',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f5e9', alpha=0.8))
    ax.text(0.02, 0.50, 'Correlated\n& simultaneous', transform=ax.transAxes,
            ha='left', va='center', fontsize=9, color='steelblue',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#e3f2fd', alpha=0.8))
    ax.text(0.75, 0.98, 'Uncorrelated\n& anticipatory', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('mean Pearson r (state entropy ↔ halt confidence)', fontsize=12)
    ax.set_ylabel('mean τ_derivative (steps, negative = halt leads)', fontsize=12)
    ax.set_title('Dissociation Scatter: r vs τ_derivative\n'
                 'Colored by Gradient Disparity (C)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(-1.1, 0.6)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig5_transition_boundary(surface, save_path):
    """
    τ_drv and gradient_disparity as parallel lines across α at each β slice,
    showing where the transition from simultaneous (≈0) to anticipatory (<0) occurs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = plt.cm.plasma(np.linspace(0.15, 0.85, len(BETA_VALUES)))

    for bi, (beta, c) in enumerate(zip(BETA_VALUES, palette)):
        tau_col = []
        gd_col  = []
        alphas_col = []
        for alpha in ALPHA_VALUES:
            k = _key(alpha, beta)
            if k not in surface:
                tau_col.append(np.nan); gd_col.append(np.nan)
            else:
                m = surface[k]
                if m['accuracy'] >= ACCURACY_THRESHOLD:
                    tau_col.append(m['tau_derivative_mean'])
                    gd = m.get('gradient_disparity') or {}
                    gd_col.append(gd.get('mean', np.nan) if gd else float('nan'))
                else:
                    tau_col.append(np.nan); gd_col.append(np.nan)
            alphas_col.append(alpha)

        label = f'β={beta}'
        axes[0].plot(alphas_col, tau_col, 'o-', color=c, linewidth=2,
                     markersize=7, label=label)
        axes[1].plot(alphas_col, gd_col, 's-', color=c, linewidth=2,
                     markersize=7, label=label)

    for ax, title, ylabel in [
        (axes[0],
         'τ_derivative vs α (per β)\nTransition boundary tracking',
         'τ_derivative (steps)'),
        (axes[1],
         'Gradient Disparity (C) vs α (per β)\nShould match τ_derivative',
         'Gradient Disparity (steps)'),
    ]:
        ax.axhline(0, color='gray', linestyle=':', linewidth=1)
        ax.axhline(E_SSM_TAU_DRV, color='gold', linestyle='--', linewidth=1.5,
                   label=f'E_ssm ref={E_SSM_TAU_DRV:.2f}')
        ax.set_xlabel('α (energy penalty)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='lower left')

    fig.suptitle('Transition from Simultaneous to Anticipatory Coupling\n'
                 'Phase 18b: α ∈ {0.01..0.05} × β ∈ {0.05..0.15}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Analysis
# ============================================================

def analyse_and_print(surface):
    print("\n" + "=" * 70)
    print("PHASE 18b ANALYSIS")
    print("=" * 70)

    # Print full table
    print(f"\n{'key':<22s}  {'acc':>5s}  {'r':>7s}  "
          f"{'τ_thr':>7s}  {'τ_drv':>7s}  {'gd_mean':>8s}  {'gd_frac_ant':>11s}")
    print("-" * 85)
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            k = _key(alpha, beta)
            if k not in surface:
                continue
            m = surface[k]
            gd = m.get('gradient_disparity') or {}
            gd_mean = gd.get('mean', float('nan')) if gd else float('nan')
            gd_fant = gd.get('frac_anticipatory', float('nan')) if gd else float('nan')
            flag = '!' if m['accuracy'] < ACCURACY_THRESHOLD else ' '
            print(f"{flag} {k:<22s}  {m['accuracy']:>5.3f}  {m['mean_r']:>7.4f}  "
                  f"{m['tau_threshold_mean']:>7.3f}  {m['tau_derivative_mean']:>7.3f}  "
                  f"{gd_mean:>8.3f}  {gd_fant:>11.3f}")

    # Dissociation diagnosis
    print("\n--- Dissociation Diagnosis ---")
    print("  Points with strong r (< -0.5) but weak τ_drv (> -0.5)  [simultaneous]:")
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            k = _key(alpha, beta)
            if k not in surface:
                continue
            m = surface[k]
            if (m['accuracy'] >= ACCURACY_THRESHOLD and
                    m['mean_r'] < -0.5 and m['tau_derivative_mean'] > -0.5):
                gd = m.get('gradient_disparity') or {}
                gd_mean = gd.get('mean', float('nan')) if gd else float('nan')
                print(f"    {k}  r={m['mean_r']:.4f}  τ_drv={m['tau_derivative_mean']:.3f}  "
                      f"gd={gd_mean:.3f}")

    print("  Points with both r < -0.5 and τ_drv < -1.0  [anticipatory]:")
    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            k = _key(alpha, beta)
            if k not in surface:
                continue
            m = surface[k]
            if (m['accuracy'] >= ACCURACY_THRESHOLD and
                    m['mean_r'] < -0.5 and m['tau_derivative_mean'] < -1.0):
                gd = m.get('gradient_disparity') or {}
                gd_mean = gd.get('mean', float('nan')) if gd else float('nan')
                print(f"    {k}  r={m['mean_r']:.4f}  τ_drv={m['tau_derivative_mean']:.3f}  "
                      f"gd={gd_mean:.3f}")

    # τ_drv vs gradient_disparity tracking
    print("\n--- Measure C vs τ_derivative tracking ---")
    pairs = []
    for k, m in surface.items():
        if m['accuracy'] < ACCURACY_THRESHOLD:
            continue
        gd = m.get('gradient_disparity') or {}
        gd_mean = gd.get('mean', float('nan')) if gd else float('nan')
        tau = m['tau_derivative_mean']
        if not (np.isnan(gd_mean) or np.isnan(tau)):
            pairs.append((tau, gd_mean))
    if len(pairs) >= 3:
        from scipy.stats import pearsonr as _pearsonr
        taus_arr = np.array([p[0] for p in pairs])
        gds_arr  = np.array([p[1] for p in pairs])
        r_track, _ = _pearsonr(taus_arr, gds_arr)
        print(f"  Pearson r(τ_drv, gradient_disparity) = {r_track:.4f}  "
              f"({'tracking ✓' if r_track > 0.7 else 'partially tracking' if r_track > 0.3 else 'dissociated ✗'})")

    print(f"\n  E_ssm reference: r={E_SSM_R:.3f}  τ_drv={E_SSM_TAU_DRV:.3f}")
    beat = [(k, m) for k, m in surface.items()
            if m['accuracy'] >= ACCURACY_THRESHOLD and m['mean_r'] < E_SSM_R]
    if beat:
        print(f"  Models beating E_ssm r: {len(beat)}")
        for k, m in sorted(beat, key=lambda x: x[1]['mean_r']):
            print(f"    {k}  r={m['mean_r']:.4f}  τ_drv={m['tau_derivative_mean']:.3f}")
    else:
        print("  No model in this grid beats E_ssm r=-0.836")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 18b: Focused grid with gradient disparity tracking")
    parser.add_argument('--results-dir',   default='results')
    parser.add_argument('--figures-dir',   default='figures')
    parser.add_argument('--device',        default=None)
    parser.add_argument('--epochs',        type=int, default=EPOCHS)
    parser.add_argument('--batch-size',    type=int, default=BATCH_SIZE)
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--skip-figures',  action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("Phase 18b: Focused Grid — Simultaneous vs Anticipatory Coupling")
    print(f"Device:  {device}")
    print(f"Grid:    {len(ALPHA_VALUES)}×{len(BETA_VALUES)} = "
          f"{len(ALPHA_VALUES)*len(BETA_VALUES)} models")
    print(f"α values: {ALPHA_VALUES}")
    print(f"β values: {BETA_VALUES}")

    # Datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64,
    )
    parity_train = get_parity_indices(train_ds)
    parity_test  = get_parity_indices(test_ds)
    print(f"  Train parity: {len(parity_train)}/{len(train_ds)}")
    print(f"  Test parity:  {len(parity_test)}/{len(test_ds)}")

    # Run sweep
    surface = run_sweep(
        train_ds, val_ds, test_ds, parity_train, parity_test,
        device, args.results_dir,
        force_retrain=args.force_retrain,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # Aggregate JSON
    out_path = os.path.join(args.results_dir, 'phase18b_response_surface.json')
    with open(out_path, 'w') as f:
        json.dump({
            'grid': surface,
            'config': {
                'alpha_values': ALPHA_VALUES,
                'beta_values':  BETA_VALUES,
                'accuracy_threshold': ACCURACY_THRESHOLD,
                'e_ssm_reference_r':       E_SSM_R,
                'e_ssm_reference_tau_drv': E_SSM_TAU_DRV,
            },
        }, f, indent=2, default=str)
    print(f"\nSaved aggregate → {out_path}")

    analyse_and_print(surface)

    if not args.skip_figures:
        print(f"\n{'='*70}")
        print("FIGURES")
        print(f"{'='*70}")
        fig1_r_heatmap(surface,
                       os.path.join(args.figures_dir, 'fig_p18b_1_r_heatmap.png'))
        fig2_tau_heatmap(surface,
                         os.path.join(args.figures_dir, 'fig_p18b_2_tau_heatmap.png'))
        fig3_disparity_heatmap(surface,
                               os.path.join(args.figures_dir,
                                            'fig_p18b_3_disparity_heatmap.png'))
        fig4_dissociation_scatter(surface,
                                  os.path.join(args.figures_dir,
                                               'fig_p18b_4_dissociation_scatter.png'))
        fig5_transition_boundary(surface,
                                 os.path.join(args.figures_dir,
                                              'fig_p18b_5_transition_boundary.png'))

    print("\nPhase 18b complete.")


if __name__ == '__main__':
    main()
