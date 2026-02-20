"""
Phase 18: Thermodynamic Control of Anticipatory State-Halt Coupling

Primary:   6×5 = 30-model grid sweep
             α ∈ {0.0, 0.01, 0.05, 0.10, 0.20, 0.50}  (energy penalty)
             β ∈ {0.0, 0.05, 0.10, 0.20, 0.40}          (halt loss weight)
           γ = 0.0 fixed;  no AdaptiveGovernor

Secondary: d_state sweep ∈ {8, 16, 32, 64, 128} at α=0.05, β=0.10
           d_model adjusted to maintain ≈5M total params

Analysis:  Phase 17 protocol per model; accuracy filter ≥95%.
           Response surface r(α,β), τ(α,β); 6 figures.

Figures
-------
  figures/fig_p18_1_r_surface.png
  figures/fig_p18_2_tau_surface.png
  figures/fig_p18_3_alpha_main.png
  figures/fig_p18_4_beta_main.png
  figures/fig_p18_5_dstate.png
  figures/fig_p18_6_interaction.png

Outputs
-------
  results/phase18_grid_a{a}_b{b}_model.pt
  results/phase18_grid_a{a}_b{b}_metrics.json
  results/phase18_dstate{d}_model.pt
  results/phase18_dstate{d}_metrics.json
  results/phase18_response_surface.json
"""

import os
import sys
import copy
import json
import math
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    HiddenStateCapture, AnswerProbe,
    train_probe, evaluate_probe_accuracy,
    get_parity_indices,
)
from phase17_proprioception_repro import (
    extract_signals, compute_all_metrics, aggregate_metrics,
)

# ── Reference values from Phase 17b ──────────────────────────
E_SSM_REFERENCE_R   = -0.836
E_SSM_REFERENCE_TAU = -2.032

# ── Grid specification ────────────────────────────────────────
ALPHA_VALUES = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
BETA_VALUES  = [0.0, 0.05, 0.10, 0.20, 0.40]

# d_state sweep at fixed (α=0.05, β=0.10)
DSTATE_VALUES    = [8, 16, 32, 64, 128]
DSTATE_ALPHA     = 0.05
DSTATE_BETA      = 0.10
DSTATE_TARGET_N  = 5_000_000   # ≈5M parameter target

# Training hyper-parameters (identical to Phase 9 / train_group)
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 3e-4
PATIENCE    = 10
WARMUP      = 100
PROBE_EPOCHS = 10
PROBE_LR     = 1e-3

# Fixed architecture for grid (same as Groups C/D/E_ssm)
GRID_D_MODEL = 512
GRID_D_STATE = 16
GRID_N_LAYERS = 6
MAX_SEQ_LEN   = 256

ACCURACY_THRESHOLD = 0.95


# ============================================================
# Model helpers
# ============================================================

def _count_ssm_params(d_model, d_state, n_layers=GRID_N_LAYERS,
                       vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN):
    """Estimate parameter count for PNA_SSM without instantiating on GPU."""
    m = PNA_SSM(vocab_size, d_model=d_model, n_layers=n_layers,
                d_state=d_state, max_seq_len=max_seq_len)
    return count_parameters(m)


def _find_d_model(d_state, target=DSTATE_TARGET_N,
                  lo=64, hi=1024, n_layers=GRID_N_LAYERS):
    """Binary search for d_model that achieves ≈target parameters."""
    while lo < hi - 1:
        mid = (lo + hi) // 2
        n = _count_ssm_params(mid, d_state, n_layers)
        if n < target:
            lo = mid
        else:
            hi = mid
    # Pick whichever is closer
    n_lo = _count_ssm_params(lo, d_state, n_layers)
    n_hi = _count_ssm_params(hi, d_state, n_layers)
    return lo if abs(n_lo - target) <= abs(n_hi - target) else hi


def make_grid_model(device):
    """Create the standard grid model (d_model=512, d_state=16, 6 layers)."""
    m = PNA_SSM(VOCAB_SIZE, d_model=GRID_D_MODEL, n_layers=GRID_N_LAYERS,
                d_state=GRID_D_STATE, max_seq_len=MAX_SEQ_LEN)
    return m.to(device)


def make_dstate_model(d_state, device):
    """Create a d_state-sweep model with d_model adjusted for ≈5M params."""
    d_model = _find_d_model(d_state)
    m = PNA_SSM(VOCAB_SIZE, d_model=d_model, n_layers=GRID_N_LAYERS,
                d_state=d_state, max_seq_len=MAX_SEQ_LEN)
    return m.to(device), d_model


def make_loss_fn(alpha, beta, gamma=0.0):
    """
    Create fixed-coefficient loss function. gamma=0.0 for all Phase 18 models.
    Uses CrossEntropyLoss only at the (0,0) corner, ThermodynamicLoss elsewhere.
    Both handle halt_confidence=None gracefully when needed, but we pass it always.
    ThermodynamicLoss with alpha=beta=gamma=0.0 is equivalent to CE-only; using
    CrossEntropyLoss there avoids the halt_target construction overhead.
    """
    if alpha == 0.0 and beta == 0.0:
        return CrossEntropyLoss(pad_token_id=VOCAB['<PAD>'])
    return ThermodynamicLoss(
        alpha=alpha, beta=beta, gamma=gamma,
        pad_token_id=VOCAB['<PAD>'],
    )


# ============================================================
# Training loop (no AdaptiveGovernor)
# ============================================================

def train_model(model, loss_fn, train_ds, val_ds, device,
                epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE,
                label="model"):
    """Train with fixed loss weights; returns best val-loss model state dict."""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr,
                                     betas=(0.9, 0.999), weight_decay=0.01)
    total_steps  = epochs * len(train_loader)
    scheduler    = get_cosine_schedule(optimizer, total_steps, warmup_steps=WARMUP)

    best_val_loss   = float('inf')
    best_state      = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # governor=None → no adaptive coefficient updates
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, governor=None,
            optimizer=optimizer, scheduler=scheduler, device=device, group='D',
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        val_loss = val_metrics['total_loss']
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] epoch {epoch:3d} | "
                  f"tr={train_metrics['total_loss']:.4f} "
                  f"val={val_loss:.4f} acc={val_metrics['accuracy']:.3f} "
                  f"({elapsed:.1f}s){improved}")

        if patience_counter >= patience:
            print(f"    [{label}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# Per-model evaluation  (Phase 17 protocol)
# ============================================================

def evaluate_model(model, loss_fn, test_ds, train_ds, parity_idx_test,
                   parity_idx_train, device, batch_size=BATCH_SIZE):
    """
    Run the full Phase 17 evaluation protocol on a single trained model.

    Returns a dict with keys:
      accuracy, halt_f1, probe_acc,
      mean_r, median_r, std_r, frac_significant,
      tau_threshold_mean, tau_threshold_median,
      tau_derivative_mean, tau_derivative_median,
      n_examples
    """
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Task accuracy
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    accuracy = test_metrics['accuracy']

    # Halt F1
    halt_f1 = compute_halt_f1(model, test_loader, device)

    # Probe — train on train set, evaluate on test set
    d_model = model.d_model
    probe = AnswerProbe(d_model)
    probe = train_probe(
        model, probe, train_ds, parity_idx_train, device,
        probe_source='d_model', epochs=PROBE_EPOCHS,
        batch_size=batch_size, lr=PROBE_LR,
    )
    probe_acc = evaluate_probe_accuracy(
        model, probe, test_ds, parity_idx_test, device, probe_source='d_model'
    )

    # Phase 17 signal extraction + metrics
    examples = extract_signals(model, probe, test_ds, parity_idx_test, device,
                               batch_size=batch_size)
    records, lag_range = compute_all_metrics(examples)
    agg = aggregate_metrics(records, lag_range)

    inst = agg['instantaneous']
    thr  = agg['threshold_lag']
    drv  = agg['deriv_xcorr']

    # Derive per-example deriv_peak_lag for mean/median
    drv_lags = [r['deriv_peak_lag'] for r in records if r['deriv_peak_lag'] is not None]
    tau_drv_mean   = float(np.mean(drv_lags))   if drv_lags else float('nan')
    tau_drv_median = float(np.median(drv_lags)) if drv_lags else float('nan')

    return {
        'accuracy':           float(accuracy),
        'halt_f1':            float(halt_f1),
        'probe_acc':          float(probe_acc),
        'mean_r':             inst['mean_r'],
        'median_r':           inst['median_r'],
        'std_r':              inst['std_r'],
        'frac_significant':   inst['fraction_significant'],
        'tau_threshold_mean': thr['mean'],
        'tau_threshold_median': thr['median'],
        'tau_derivative_mean':  tau_drv_mean,
        'tau_derivative_median': tau_drv_median,
        'n_examples':         agg['n_examples'],
    }


# ============================================================
# Grid sweep
# ============================================================

def run_grid(train_ds, val_ds, test_ds, parity_train, parity_test,
             device, results_dir, force_retrain=False,
             epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train and evaluate all 30 (α, β) models. Returns surface dict."""
    surface = {}

    total = len(ALPHA_VALUES) * len(BETA_VALUES)
    done  = 0

    for alpha in ALPHA_VALUES:
        for beta in BETA_VALUES:
            done += 1
            a_str = f"{alpha:.2f}".replace('.', 'p')
            b_str = f"{beta:.2f}".replace('.', 'p')
            key   = f"a{a_str}_b{b_str}"
            ckpt_path    = os.path.join(results_dir, f"phase18_grid_{key}_model.pt")
            metrics_path = os.path.join(results_dir, f"phase18_grid_{key}_metrics.json")

            print(f"\n[{done}/{total}] Grid α={alpha}, β={beta}  ({key})")

            # Load cached metrics if available
            if not force_retrain and os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    m = json.load(f)
                print(f"  Loaded cached metrics  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}")
                surface[key] = {'alpha': alpha, 'beta': beta, **m}
                continue

            model = make_grid_model(device)
            loss_fn = make_loss_fn(alpha, beta)

            label = f"α={alpha} β={beta}"
            model = train_model(model, loss_fn, train_ds, val_ds, device,
                                epochs=epochs, batch_size=batch_size, label=label)

            # Save checkpoint
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

            # Evaluate
            model.eval()
            m = evaluate_model(model, loss_fn, test_ds, train_ds,
                               parity_test, parity_train, device)
            m['alpha'] = alpha
            m['beta']  = beta
            print(f"  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
                  f"τ_thr={m['tau_threshold_mean']:.3f}  τ_drv={m['tau_derivative_mean']:.3f}")

            # Save metrics
            with open(metrics_path, 'w') as f:
                json.dump(m, f, indent=2, default=str)

            surface[key] = {'alpha': alpha, 'beta': beta, **m}

            del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    return surface


# ============================================================
# d_state sweep
# ============================================================

def run_dstate_sweep(train_ds, val_ds, test_ds, parity_train, parity_test,
                     device, results_dir, force_retrain=False,
                     epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train and evaluate 5 d_state variants at α=0.05, β=0.10."""
    results = {}

    for d_state in DSTATE_VALUES:
        key          = f"dstate{d_state}"
        ckpt_path    = os.path.join(results_dir, f"phase18_{key}_model.pt")
        metrics_path = os.path.join(results_dir, f"phase18_{key}_metrics.json")

        print(f"\n[d_state sweep] d_state={d_state}")

        if not force_retrain and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            print(f"  Loaded cached metrics  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}")
            results[d_state] = m
            continue

        model, d_model_used = make_dstate_model(d_state, device)
        n_params = count_parameters(model)
        print(f"  d_model={d_model_used}  d_state={d_state}  params={n_params:,}")

        loss_fn = make_loss_fn(DSTATE_ALPHA, DSTATE_BETA)
        label   = f"dstate={d_state}"

        model = train_model(model, loss_fn, train_ds, val_ds, device,
                            epochs=epochs, batch_size=batch_size, label=label)

        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

        model.eval()
        m = evaluate_model(model, loss_fn, test_ds, train_ds,
                           parity_test, parity_train, device)
        m['d_state']  = d_state
        m['d_model']  = d_model_used
        m['n_params'] = n_params
        print(f"  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
              f"τ_drv={m['tau_derivative_mean']:.3f}")

        with open(metrics_path, 'w') as f:
            json.dump(m, f, indent=2, default=str)

        results[d_state] = m

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ============================================================
# Figures
# ============================================================

def _build_grid_arrays(surface):
    """Build 2-D arrays indexed [alpha_idx, beta_idx] from surface dict."""
    A = ALPHA_VALUES
    B = BETA_VALUES

    r_arr    = np.full((len(A), len(B)), np.nan)
    tau_arr  = np.full((len(A), len(B)), np.nan)
    acc_arr  = np.full((len(A), len(B)), np.nan)
    ok_arr   = np.zeros((len(A), len(B)), dtype=bool)

    for ai, alpha in enumerate(A):
        for bi, beta in enumerate(B):
            a_str = f"{alpha:.2f}".replace('.', 'p')
            b_str = f"{beta:.2f}".replace('.', 'p')
            key   = f"a{a_str}_b{b_str}"
            if key not in surface:
                continue
            m = surface[key]
            acc_arr[ai, bi] = m['accuracy']
            ok_arr[ai, bi]  = m['accuracy'] >= ACCURACY_THRESHOLD
            if ok_arr[ai, bi]:
                r_arr[ai, bi]   = m['mean_r']
                tau_arr[ai, bi] = m['tau_threshold_mean']

    return r_arr, tau_arr, acc_arr, ok_arr


def fig1_r_surface(surface, save_path):
    """r(α, β) heatmap with low-accuracy cells hatched out."""
    r_arr, _, acc_arr, ok_arr = _build_grid_arrays(surface)

    fig, ax = plt.subplots(figsize=(9, 6))
    alpha_labels = [str(a) for a in ALPHA_VALUES]
    beta_labels  = [str(b) for b in BETA_VALUES]

    vmin = -1.0
    vmax = 0.0
    cmap = 'RdBu'

    im = ax.imshow(r_arr, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   origin='lower')

    # Hatch failed cells
    for ai in range(len(ALPHA_VALUES)):
        for bi in range(len(BETA_VALUES)):
            if not ok_arr[ai, bi]:
                ax.add_patch(plt.Rectangle(
                    (bi - 0.5, ai - 0.5), 1, 1,
                    fill=True, facecolor='lightgray', edgecolor='gray',
                    hatch='///', linewidth=0.5, zorder=2,
                ))
            else:
                # Annotate with value
                val = r_arr[ai, bi]
                if not np.isnan(val):
                    ax.text(bi, ai, f'{val:.3f}', ha='center', va='center',
                            fontsize=7.5, zorder=3)

    ax.set_xticks(range(len(BETA_VALUES)))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels(alpha_labels)
    ax.set_xlabel('β (halt loss weight)', fontsize=12)
    ax.set_ylabel('α (energy penalty)', fontsize=12)
    ax.set_title('Response Surface: mean Pearson r(α, β)\n'
                 '(gray hatch = accuracy < 95%)', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='mean r')

    # Mark E_ssm reference
    try:
        ai_ref = ALPHA_VALUES.index(0.05)
        bi_ref = BETA_VALUES.index(0.10)
        ax.plot(bi_ref, ai_ref, 'k*', markersize=12, zorder=4, label='E_ssm ref')
        ax.legend(fontsize=9, loc='upper left')
    except ValueError:
        pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig2_tau_surface(surface, save_path):
    """τ_threshold(α, β) heatmap."""
    _, tau_arr, acc_arr, ok_arr = _build_grid_arrays(surface)

    fig, ax = plt.subplots(figsize=(9, 6))
    alpha_labels = [str(a) for a in ALPHA_VALUES]
    beta_labels  = [str(b) for b in BETA_VALUES]

    vmin = min(np.nanmin(tau_arr), -3.0) if not np.all(np.isnan(tau_arr)) else -3.0
    vmax = max(np.nanmax(tau_arr),  2.0) if not np.all(np.isnan(tau_arr)) else  2.0
    cmap = 'coolwarm_r'

    im = ax.imshow(tau_arr, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap,
                   origin='lower')

    for ai in range(len(ALPHA_VALUES)):
        for bi in range(len(BETA_VALUES)):
            if not ok_arr[ai, bi]:
                ax.add_patch(plt.Rectangle(
                    (bi - 0.5, ai - 0.5), 1, 1,
                    fill=True, facecolor='lightgray', edgecolor='gray',
                    hatch='///', linewidth=0.5, zorder=2,
                ))
            else:
                val = tau_arr[ai, bi]
                if not np.isnan(val):
                    ax.text(bi, ai, f'{val:.2f}', ha='center', va='center',
                            fontsize=7.5, zorder=3)

    ax.set_xticks(range(len(BETA_VALUES)))
    ax.set_xticklabels(beta_labels)
    ax.set_yticks(range(len(ALPHA_VALUES)))
    ax.set_yticklabels(alpha_labels)
    ax.set_xlabel('β (halt loss weight)', fontsize=12)
    ax.set_ylabel('α (energy penalty)', fontsize=12)
    ax.set_title('Response Surface: mean τ_threshold(α, β)\n'
                 '(negative = halt fires before state collapse)', fontsize=13, fontweight='bold')

    plt.colorbar(im, ax=ax, label='τ threshold (steps)')

    try:
        ai_ref = ALPHA_VALUES.index(0.05)
        bi_ref = BETA_VALUES.index(0.10)
        ax.plot(bi_ref, ai_ref, 'k*', markersize=12, zorder=4, label='E_ssm ref')
        ax.legend(fontsize=9, loc='upper left')
    except ValueError:
        pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig3_alpha_main(surface, save_path):
    """α main-effect curve: r and τ vs α at β=0 (induction only)."""
    beta_fixed = 0.0
    rs, taus, accs = [], [], []
    valid_alphas = []

    for alpha in ALPHA_VALUES:
        a_str = f"{alpha:.2f}".replace('.', 'p')
        b_str = f"{beta_fixed:.2f}".replace('.', 'p')
        key = f"a{a_str}_b{b_str}"
        if key not in surface:
            continue
        m = surface[key]
        accs.append(m['accuracy'])
        if m['accuracy'] >= ACCURACY_THRESHOLD:
            valid_alphas.append(alpha)
            rs.append(m['mean_r'])
            taus.append(m['tau_threshold_mean'])
        else:
            valid_alphas.append(alpha)
            rs.append(np.nan)
            taus.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    color_r   = '#2980b9'
    color_tau = '#e74c3c'

    ax1.plot(valid_alphas, rs, 'o-', color=color_r, linewidth=2, markersize=8)
    ax1.axhline(E_SSM_REFERENCE_R, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  r={E_SSM_REFERENCE_R:.3f}')
    ax1.set_ylabel('mean Pearson r', fontsize=12)
    ax1.set_ylim(-1.0, 0.2)
    ax1.legend(fontsize=9)
    ax1.set_title('α Main Effect (β=0): Thermodynamic Induction Alone', fontweight='bold')

    # Mark failed points
    for i, (a, r_val, acc) in enumerate(zip(valid_alphas, rs, accs)):
        if acc < ACCURACY_THRESHOLD:
            ax1.annotate(f'acc={acc:.2f}', xy=(a, -0.05), fontsize=7.5,
                         ha='center', color='gray', style='italic')

    ax2.plot(valid_alphas, taus, 's-', color=color_tau, linewidth=2, markersize=8)
    ax2.axhline(E_SSM_REFERENCE_TAU, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  τ={E_SSM_REFERENCE_TAU:.2f}')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('mean τ_threshold (steps)', fontsize=12)
    ax2.set_xlabel('α (energy penalty weight)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_title('τ Threshold vs α (β=0)', fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig4_beta_main(surface, save_path):
    """β main-effect curve: r and τ vs β at α=0 (halt supervision only)."""
    alpha_fixed = 0.0
    rs, taus, accs = [], [], []
    valid_betas = []

    for beta in BETA_VALUES:
        a_str = f"{alpha_fixed:.2f}".replace('.', 'p')
        b_str = f"{beta:.2f}".replace('.', 'p')
        key = f"a{a_str}_b{b_str}"
        if key not in surface:
            continue
        m = surface[key]
        accs.append(m['accuracy'])
        valid_betas.append(beta)
        if m['accuracy'] >= ACCURACY_THRESHOLD:
            rs.append(m['mean_r'])
            taus.append(m['tau_threshold_mean'])
        else:
            rs.append(np.nan)
            taus.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    color_r   = '#2980b9'
    color_tau = '#e74c3c'

    ax1.plot(valid_betas, rs, 'o-', color=color_r, linewidth=2, markersize=8)
    ax1.axhline(E_SSM_REFERENCE_R, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  r={E_SSM_REFERENCE_R:.3f}')
    ax1.set_ylabel('mean Pearson r', fontsize=12)
    ax1.set_ylim(-1.0, 0.2)
    ax1.legend(fontsize=9)
    ax1.set_title('β Main Effect (α=0): Halt Supervision Alone', fontweight='bold')

    for i, (b, r_val, acc) in enumerate(zip(valid_betas, rs, accs)):
        if acc < ACCURACY_THRESHOLD:
            ax1.annotate(f'acc={acc:.2f}', xy=(b, -0.05), fontsize=7.5,
                         ha='center', color='gray', style='italic')

    ax2.plot(valid_betas, taus, 's-', color=color_tau, linewidth=2, markersize=8)
    ax2.axhline(E_SSM_REFERENCE_TAU, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  τ={E_SSM_REFERENCE_TAU:.2f}')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('mean τ_threshold (steps)', fontsize=12)
    ax2.set_xlabel('β (halt loss weight)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.set_title('τ Threshold vs β (α=0)', fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig5_dstate(dstate_results, save_path):
    """r and τ_derivative vs d_state at fixed α=0.05, β=0.10."""
    d_states_sorted = sorted(dstate_results.keys())
    rs   = [dstate_results[d]['mean_r']           for d in d_states_sorted]
    taus = [dstate_results[d]['tau_derivative_mean'] for d in d_states_sorted]
    accs = [dstate_results[d]['accuracy']          for d in d_states_sorted]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    color_r   = '#2980b9'
    color_tau = '#e74c3c'

    x = np.log2(np.array(d_states_sorted, dtype=float))
    x_ticks = [np.log2(d) for d in d_states_sorted]
    x_labels = [str(d) for d in d_states_sorted]

    # Filter by accuracy
    rs_plot   = [r if a >= ACCURACY_THRESHOLD else np.nan for r, a in zip(rs, accs)]
    taus_plot = [t if a >= ACCURACY_THRESHOLD else np.nan for t, a in zip(taus, accs)]

    ax1.plot(x, rs_plot, 'o-', color=color_r, linewidth=2, markersize=8)
    ax1.axhline(E_SSM_REFERENCE_R, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  r={E_SSM_REFERENCE_R:.3f}')
    ax1.set_ylabel('mean Pearson r', fontsize=12)
    ax1.set_ylim(-1.0, 0.2)
    ax1.legend(fontsize=9)
    ax1.set_title(f'd_state Sweep (α={DSTATE_ALPHA}, β={DSTATE_BETA}): '
                  'Geometric Capacity Effect', fontweight='bold')

    ax2.plot(x, taus_plot, 's-', color=color_tau, linewidth=2, markersize=8)
    ax2.axhline(E_SSM_REFERENCE_TAU, color='orange', linestyle='--', linewidth=1.5,
                label=f'E_ssm ref  τ_drv={E_SSM_REFERENCE_TAU:.2f}')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.set_ylabel('mean τ_derivative (steps)', fontsize=12)
    ax2.set_xlabel('d_state', fontsize=12)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)
    ax2.legend(fontsize=9)
    ax2.set_title('τ_derivative vs d_state', fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


def fig6_interaction(surface, save_path):
    """
    Interaction contrast: for each β, plot r vs α.
    Also show expected-additive baseline to visualise sub/super-additivity.
    """
    fig, (ax_r, ax_tau) = plt.subplots(1, 2, figsize=(14, 6))

    palette = plt.cm.viridis(np.linspace(0.15, 0.85, len(BETA_VALUES)))

    for bi, (beta, c) in enumerate(zip(BETA_VALUES, palette)):
        rs_col, taus_col = [], []
        alphas_col = []
        for alpha in ALPHA_VALUES:
            a_str = f"{alpha:.2f}".replace('.', 'p')
            b_str = f"{beta:.2f}".replace('.', 'p')
            key = f"a{a_str}_b{b_str}"
            if key not in surface:
                rs_col.append(np.nan)
                taus_col.append(np.nan)
            else:
                m = surface[key]
                if m['accuracy'] >= ACCURACY_THRESHOLD:
                    rs_col.append(m['mean_r'])
                    taus_col.append(m['tau_threshold_mean'])
                else:
                    rs_col.append(np.nan)
                    taus_col.append(np.nan)
            alphas_col.append(alpha)

        label = f'β={beta}'
        ax_r.plot(alphas_col, rs_col, 'o-', color=c, linewidth=1.8,
                  markersize=6, label=label)
        ax_tau.plot(alphas_col, taus_col, 's-', color=c, linewidth=1.8,
                    markersize=6, label=label)

    ax_r.axhline(E_SSM_REFERENCE_R, color='orange', linestyle='--', linewidth=1.5,
                 label=f'E_ssm ref')
    ax_r.set_xlabel('α (energy penalty)', fontsize=12)
    ax_r.set_ylabel('mean Pearson r', fontsize=12)
    ax_r.set_ylim(-1.0, 0.2)
    ax_r.set_title('Interaction: r(α) for each β\n(family of curves)', fontweight='bold')
    ax_r.legend(fontsize=8, loc='lower left')

    ax_tau.axhline(E_SSM_REFERENCE_TAU, color='orange', linestyle='--', linewidth=1.5,
                   label=f'E_ssm ref')
    ax_tau.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax_tau.set_xlabel('α (energy penalty)', fontsize=12)
    ax_tau.set_ylabel('mean τ_threshold (steps)', fontsize=12)
    ax_tau.set_title('Interaction: τ(α) for each β', fontweight='bold')
    ax_tau.legend(fontsize=8, loc='lower left')

    fig.suptitle('α × β Interaction Contrast (α main-effect per β slice)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Analysis / reporting
# ============================================================

def analyse_and_print(surface, dstate_results):
    """Print Phase 18 success/failure criteria assessment."""
    print("\n" + "=" * 70)
    print("PHASE 18 ANALYSIS")
    print("=" * 70)

    # ── Accuracy filter ───────────────────────────────────────────────────────
    failed = [(k, v) for k, v in surface.items()
              if v['accuracy'] < ACCURACY_THRESHOLD]
    passed = [(k, v) for k, v in surface.items()
              if v['accuracy'] >= ACCURACY_THRESHOLD]
    print(f"\nAccuracy filter ({ACCURACY_THRESHOLD*100:.0f}%):"
          f"  {len(passed)}/30 passed  |  {len(failed)} failed")
    if failed:
        for k, v in sorted(failed, key=lambda x: x[1]['accuracy']):
            print(f"  FAILED  {k}  acc={v['accuracy']:.3f}")

    # ── Success criterion 1: β=0 column monotone ─────────────────────────────
    print("\n--- Criterion 1: β=0 column monotone in r ---")
    beta0 = []
    for alpha in ALPHA_VALUES:
        a_str = f"{alpha:.2f}".replace('.', 'p')
        key = f"a{a_str}_b0p00"
        if key in surface and surface[key]['accuracy'] >= ACCURACY_THRESHOLD:
            beta0.append((alpha, surface[key]['mean_r']))
    if len(beta0) >= 2:
        rs_beta0 = [r for _, r in beta0]
        monotone = all(rs_beta0[i] <= rs_beta0[i + 1] or
                       abs(rs_beta0[i] - rs_beta0[i + 1]) < 0.01
                       for i in range(len(rs_beta0) - 1))
        print(f"  β=0 r values: {[(a, f'{r:.4f}') for a, r in beta0]}")
        print(f"  Monotone (more negative with α): {'YES' if not monotone else 'check direction'}")
        # Decreasing r (more negative) with α = increasing coupling
        monotone_coupling = all(rs_beta0[i] >= rs_beta0[i + 1]
                                for i in range(len(rs_beta0) - 1))
        print(f"  Coupling monotone (r decreasing with α): {'YES ✓' if monotone_coupling else 'NO ✗'}")
    else:
        print("  Insufficient data.")

    # ── Success criterion 2: α=0 column weaker than diagonal ─────────────────
    print("\n--- Criterion 2: α=0 column weaker than diagonal ---")
    alpha0_rs = []
    for beta in BETA_VALUES:
        b_str = f"{beta:.2f}".replace('.', 'p')
        key = f"a0p00_b{b_str}"
        if key in surface and surface[key]['accuracy'] >= ACCURACY_THRESHOLD:
            alpha0_rs.append(surface[key]['mean_r'])
    diag_rs = []
    for alpha, beta in zip(ALPHA_VALUES, BETA_VALUES):
        a_str = f"{alpha:.2f}".replace('.', 'p')
        b_str = f"{beta:.2f}".replace('.', 'p')
        key = f"a{a_str}_b{b_str}"
        if key in surface and surface[key]['accuracy'] >= ACCURACY_THRESHOLD:
            diag_rs.append(surface[key]['mean_r'])
    if alpha0_rs and diag_rs:
        mean_alpha0 = np.mean(alpha0_rs)
        mean_diag   = np.mean(diag_rs)
        print(f"  Mean r, α=0 column: {mean_alpha0:.4f}")
        print(f"  Mean r, diagonal:   {mean_diag:.4f}")
        print(f"  α=0 weaker: {'YES ✓' if mean_alpha0 > mean_diag else 'NO ✗'}")

    # ── Success criterion 3: τ_derivative tracks τ_threshold ─────────────────
    print("\n--- Criterion 3: τ_derivative tracks τ_threshold ---")
    thr_list, drv_list = [], []
    for m in surface.values():
        if m['accuracy'] >= ACCURACY_THRESHOLD:
            thr_list.append(m['tau_threshold_mean'])
            drv_list.append(m['tau_derivative_mean'])
    if len(thr_list) >= 3:
        from scipy.stats import pearsonr as _pearsonr
        thr_arr = np.array(thr_list)
        drv_arr = np.array(drv_list)
        valid = ~(np.isnan(thr_arr) | np.isnan(drv_arr))
        if valid.sum() >= 3:
            r_track, _ = _pearsonr(thr_arr[valid], drv_arr[valid])
            print(f"  Pearson r(τ_thr, τ_drv) = {r_track:.4f}  "
                  f"({'tracking ✓' if r_track > 0.5 else 'dissociated ✗'})")

    # ── Success criterion 4: beat E_ssm reference ────────────────────────────
    print("\n--- Criterion 4: any model beats E_ssm r=-0.836 ---")
    beat = [(k, v) for k, v in surface.items()
            if v['accuracy'] >= ACCURACY_THRESHOLD and v['mean_r'] < E_SSM_REFERENCE_R]
    if beat:
        print(f"  YES ✓  {len(beat)} model(s):")
        for k, v in sorted(beat, key=lambda x: x[1]['mean_r']):
            print(f"    {k}  r={v['mean_r']:.4f}  acc={v['accuracy']:.3f}")
    else:
        print("  NO ✗  No model exceeded the E_ssm reference.")

    # ── d_state sweep summary ─────────────────────────────────────────────────
    if dstate_results:
        print("\n--- d_state Sweep ---")
        print(f"  {'d_state':>8s}  {'d_model':>8s}  {'params':>10s}  "
              f"{'acc':>6s}  {'r':>8s}  {'τ_drv':>8s}")
        for d in sorted(dstate_results.keys()):
            m = dstate_results[d]
            n = m.get('n_params', 0)
            print(f"  {d:>8d}  {m.get('d_model', '?'):>8}  {n:>10,}  "
                  f"{m['accuracy']:>6.3f}  {m['mean_r']:>8.4f}  "
                  f"{m['tau_derivative_mean']:>8.3f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 18: Thermodynamic Control")
    parser.add_argument('--results-dir',  default='results')
    parser.add_argument('--figures-dir',  default='figures')
    parser.add_argument('--device',       default=None)
    parser.add_argument('--epochs',       type=int, default=EPOCHS)
    parser.add_argument('--batch-size',   type=int, default=BATCH_SIZE)
    parser.add_argument('--force-retrain', action='store_true',
                        help='Ignore cached checkpoints and retrain everything')
    parser.add_argument('--skip-grid',    action='store_true')
    parser.add_argument('--skip-dstate',  action='store_true')
    parser.add_argument('--skip-figures', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    epochs     = args.epochs
    batch_size = args.batch_size

    print(f"Phase 18: Thermodynamic Control of Anticipatory Coupling")
    print(f"Device: {device}")
    print(f"Grid: {len(ALPHA_VALUES)}×{len(BETA_VALUES)} = "
          f"{len(ALPHA_VALUES)*len(BETA_VALUES)} models")
    print(f"d_state sweep: {DSTATE_VALUES}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64,
    )
    parity_train = get_parity_indices(train_ds)
    parity_test  = get_parity_indices(test_ds)
    print(f"  Train parity: {len(parity_train)}/{len(train_ds)}")
    print(f"  Test parity:  {len(parity_test)}/{len(test_ds)}")

    # ── Primary: Grid sweep ───────────────────────────────────────────────────
    surface = {}
    if not args.skip_grid:
        print(f"\n{'='*70}")
        print("PRIMARY EXPERIMENT: 2D HYPERPARAMETER GRID")
        print(f"{'='*70}")
        surface = run_grid(
            train_ds, val_ds, test_ds, parity_train, parity_test,
            device, args.results_dir, force_retrain=args.force_retrain,
            epochs=epochs, batch_size=batch_size,
        )

    # ── Secondary: d_state sweep ──────────────────────────────────────────────
    dstate_results = {}
    if not args.skip_dstate:
        print(f"\n{'='*70}")
        print("SECONDARY EXPERIMENT: d_state SWEEP")
        print(f"{'='*70}")
        dstate_results = run_dstate_sweep(
            train_ds, val_ds, test_ds, parity_train, parity_test,
            device, args.results_dir, force_retrain=args.force_retrain,
            epochs=epochs, batch_size=batch_size,
        )

    # ── Save aggregate JSON ───────────────────────────────────────────────────
    aggregate = {
        'grid':   surface,
        'dstate': dstate_results,
        'config': {
            'alpha_values':  ALPHA_VALUES,
            'beta_values':   BETA_VALUES,
            'dstate_values': DSTATE_VALUES,
            'accuracy_threshold': ACCURACY_THRESHOLD,
            'e_ssm_reference_r':   E_SSM_REFERENCE_R,
            'e_ssm_reference_tau': E_SSM_REFERENCE_TAU,
        },
    }
    surface_path = os.path.join(args.results_dir, 'phase18_response_surface.json')
    with open(surface_path, 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"\nSaved aggregate → {surface_path}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    if surface or dstate_results:
        analyse_and_print(surface, dstate_results)

    # ── Figures ───────────────────────────────────────────────────────────────
    if not args.skip_figures and surface:
        print(f"\n{'='*70}")
        print("GENERATING FIGURES")
        print(f"{'='*70}")

        fig1_r_surface(surface,
                       os.path.join(args.figures_dir, 'fig_p18_1_r_surface.png'))
        fig2_tau_surface(surface,
                         os.path.join(args.figures_dir, 'fig_p18_2_tau_surface.png'))
        fig3_alpha_main(surface,
                        os.path.join(args.figures_dir, 'fig_p18_3_alpha_main.png'))
        fig4_beta_main(surface,
                       os.path.join(args.figures_dir, 'fig_p18_4_beta_main.png'))
        if dstate_results:
            fig5_dstate(dstate_results,
                        os.path.join(args.figures_dir, 'fig_p18_5_dstate.png'))
        fig6_interaction(surface,
                         os.path.join(args.figures_dir, 'fig_p18_6_interaction.png'))

    print("\nPhase 18 complete.")


if __name__ == '__main__':
    main()
