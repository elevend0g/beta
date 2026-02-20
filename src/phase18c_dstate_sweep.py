"""
Phase 18c: d_state Dimensionality Sweep
Fixed α=0.05, β=0.10, γ=0.0 — identical loss to E_ssm.

Question: Is proprioceptive coupling geometrically bottlenecked by d_state,
or does the recurrent state at d_state=16 already have sufficient capacity?

d_state ∈ {8, 16, 32, 64, 128}
d_model adjusted per model to maintain ≈5M total parameters.

Metrics per model (full Phase 17 + Measure C):
  r            — mean instantaneous Pearson r(state_entropy, halt_confidence)
  τ_derivative — mean derivative cross-correlation peak lag (negative = anticipatory)
  τ_threshold  — mean threshold lag (halt-rise minus state-collapse crossing)
  gradient_disparity — mean argmax(dHC/dt) − argmax(−dSE/dt)  (Measure C)
  frac_anticipatory  — fraction of examples where halt gradient leads state gradient
  accuracy     — task accuracy on parity test set

Figures
-------
  figures/fig_p18c_dstate_sweep.png   — 3-panel: r, τ_drv, gradient_disparity vs d_state

Outputs
-------
  results/phase18c_dstate{d}_model.pt
  results/phase18c_dstate{d}_metrics.json
  results/phase18c_dstate_sweep.json
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
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE, VOCAB
from models import PNA_SSM, count_parameters
from losses import ThermodynamicLoss
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

# ── Fixed hyperparameters ─────────────────────────────────────
FIXED_ALPHA = 0.05
FIXED_BETA  = 0.10
FIXED_GAMMA = 0.0

DSTATE_VALUES   = [8, 16, 32, 64, 128]
TARGET_PARAMS   = 5_000_000   # ≈5M

# Training (identical to Phase 9 / train_group)
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 3e-4
PATIENCE    = 10
WARMUP      = 100
PROBE_EPOCHS = 10
PROBE_LR     = 1e-3

N_LAYERS    = 6
MAX_SEQ     = 256

# Reference points
E_SSM_R       = -0.836
E_SSM_TAU_DRV = -2.032


# ============================================================
# d_model search
# ============================================================

def _param_count(d_model, d_state):
    m = PNA_SSM(VOCAB_SIZE, d_model=d_model, n_layers=N_LAYERS,
                d_state=d_state, max_seq_len=MAX_SEQ)
    return count_parameters(m)


def find_d_model(d_state, target=TARGET_PARAMS, lo=64, hi=1024):
    """Binary-search d_model so that parameter count ≈ target."""
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if _param_count(mid, d_state) < target:
            lo = mid
        else:
            hi = mid
    n_lo = _param_count(lo, d_state)
    n_hi = _param_count(hi, d_state)
    return lo if abs(n_lo - target) <= abs(n_hi - target) else hi


# ============================================================
# Measure C: Gradient Disparity
# ============================================================

def compute_gradient_disparity(examples, min_len=5):
    """
    gradient_disparity = argmax(dHC/dt) − argmax(−dSE/dt)
      < 0  halt rise peak precedes state collapse peak  (anticipatory)
      = 0  simultaneous
      > 0  state collapses first, halt follows           (lagging)
    """
    records = []
    for ex in examples:
        se = ex['state_entropy']
        hc = ex['halt_confidence']
        if len(se) < min_len:
            continue
        dse = np.diff(se)
        dhc = np.diff(hc)
        if len(dse) == 0:
            continue
        collapse_pos = int(np.argmax(-dse))
        rise_pos     = int(np.argmax(dhc))
        records.append({
            'disparity':         rise_pos - collapse_pos,
            'collapse_strength': float(-dse[collapse_pos]),
            'rise_strength':     float(dhc[rise_pos]),
        })

    if not records:
        return None

    disp  = np.array([r['disparity']         for r in records])
    c_str = np.array([r['collapse_strength']  for r in records])
    r_str = np.array([r['rise_strength']      for r in records])

    return {
        'mean':               float(np.mean(disp)),
        'median':             float(np.median(disp)),
        'std':                float(np.std(disp)),
        'frac_anticipatory':  float(np.mean(disp < 0)),
        'mean_collapse_strength': float(np.mean(c_str)),
        'mean_rise_strength':     float(np.mean(r_str)),
        'n': len(records),
    }


# ============================================================
# Training
# ============================================================

def train_model(model, loss_fn, train_ds, val_ds, device,
                epochs=EPOCHS, batch_size=BATCH_SIZE, label=""):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR,
                                    betas=(0.9, 0.999), weight_decay=0.01)
    scheduler   = get_cosine_schedule(optimizer, epochs * len(train_loader),
                                      warmup_steps=WARMUP)

    best_val   = float('inf')
    best_state = None
    pat_count  = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, loss_fn, governor=None,
                             optimizer=optimizer, scheduler=scheduler,
                             device=device, group='D')
        val = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        vl = val['total_loss']
        imp = ""
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(model.state_dict())
            pat_count = 0
            imp = " *"
        else:
            pat_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    [{label}] epoch {epoch:3d} | "
                  f"tr={tr['total_loss']:.4f} val={vl:.4f} "
                  f"acc={val['accuracy']:.3f} ({elapsed:.1f}s){imp}")

        if pat_count >= PATIENCE:
            print(f"    [{label}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(model, loss_fn, test_ds, train_ds,
                   parity_test, parity_train, device,
                   batch_size=BATCH_SIZE):
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    test_metrics = evaluate(model, test_loader, loss_fn, device)
    halt_f1      = compute_halt_f1(model, test_loader, device)

    probe = AnswerProbe(model.d_model)
    probe = train_probe(model, probe, train_ds, parity_train, device,
                        probe_source='d_model', epochs=PROBE_EPOCHS,
                        batch_size=batch_size, lr=PROBE_LR)
    probe_acc = evaluate_probe_accuracy(model, probe, test_ds, parity_test,
                                        device, probe_source='d_model')

    examples = extract_signals(model, probe, test_ds, parity_test,
                               device, batch_size=batch_size)
    records, lag_range = compute_all_metrics(examples)
    agg  = aggregate_metrics(records, lag_range)
    inst = agg['instantaneous']
    thr  = agg['threshold_lag']

    drv_lags = [r['deriv_peak_lag'] for r in records if r['deriv_peak_lag'] is not None]
    tau_drv_mean   = float(np.mean(drv_lags))   if drv_lags else float('nan')
    tau_drv_median = float(np.median(drv_lags)) if drv_lags else float('nan')

    gd = compute_gradient_disparity(examples)

    return {
        'accuracy':              float(test_metrics['accuracy']),
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
# Sweep
# ============================================================

def run_sweep(train_ds, val_ds, test_ds, parity_train, parity_test,
              device, results_dir,
              force_retrain=False, epochs=EPOCHS, batch_size=BATCH_SIZE):
    results = {}
    loss_fn = ThermodynamicLoss(alpha=FIXED_ALPHA, beta=FIXED_BETA,
                                gamma=FIXED_GAMMA, pad_token_id=VOCAB['<PAD>'])

    for d_state in DSTATE_VALUES:
        key          = f"dstate{d_state}"
        ckpt_path    = os.path.join(results_dir, f"phase18c_{key}_model.pt")
        metrics_path = os.path.join(results_dir, f"phase18c_{key}_metrics.json")

        d_model = find_d_model(d_state)
        n_params = _param_count(d_model, d_state)

        print(f"\n[d_state={d_state}]  d_model={d_model}  params={n_params:,}")

        if not force_retrain and os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            gd = m.get('gradient_disparity') or {}
            gd_m = gd.get('mean', float('nan')) if gd else float('nan')
            print(f"  [CACHE] acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
                  f"τ_drv={m['tau_derivative_mean']:.3f}  gd={gd_m:.3f}")
            results[d_state] = m
            continue

        model = PNA_SSM(VOCAB_SIZE, d_model=d_model, n_layers=N_LAYERS,
                        d_state=d_state, max_seq_len=MAX_SEQ).to(device)

        model = train_model(model, loss_fn, train_ds, val_ds, device,
                            epochs=epochs, batch_size=batch_size,
                            label=f"d_state={d_state}")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved → {ckpt_path}")

        model.eval()
        m = evaluate_model(model, loss_fn, test_ds, train_ds,
                           parity_test, parity_train, device,
                           batch_size=batch_size)
        m['d_state']  = d_state
        m['d_model']  = d_model
        m['n_params'] = n_params

        gd = m.get('gradient_disparity') or {}
        gd_m = gd.get('mean', float('nan')) if gd else float('nan')
        print(f"  acc={m['accuracy']:.3f}  r={m['mean_r']:.4f}  "
              f"τ_drv={m['tau_derivative_mean']:.3f}  gd={gd_m:.3f}  "
              f"gd_frac_ant={gd.get('frac_anticipatory', float('nan')):.3f}")

        with open(metrics_path, 'w') as f:
            json.dump(m, f, indent=2, default=str)

        results[d_state] = m

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


# ============================================================
# Figure
# ============================================================

def make_figure(results, save_path):
    """
    3-panel figure:
      Top:    mean r vs d_state
      Middle: τ_derivative vs d_state
      Bottom: gradient_disparity (C) vs d_state

    All three should move together if proprioception is geometrically
    bottlenecked by d_state capacity.  Flat curves = loss function is
    the binding constraint, not state dimensionality.
    """
    d_states = sorted(results.keys())
    d_model_labels = [f"d={results[d]['d_model']}" for d in d_states]
    n_params_k = [results[d]['n_params'] / 1e6 for d in d_states]

    rs    = [results[d]['mean_r']               for d in d_states]
    taus  = [results[d]['tau_derivative_mean']  for d in d_states]
    accs  = [results[d]['accuracy']             for d in d_states]

    gd_means = []
    gd_fants = []
    for d in d_states:
        gd = results[d].get('gradient_disparity') or {}
        gd_means.append(gd.get('mean', float('nan'))             if gd else float('nan'))
        gd_fants.append(gd.get('frac_anticipatory', float('nan')) if gd else float('nan'))

    # x-axis: log2(d_state) with d_state labels
    x = np.log2(np.array(d_states, dtype=float))

    # Mask low-accuracy models
    ACC_THRESH = 0.95
    rs_plot    = [r if a >= ACC_THRESH else np.nan for r, a in zip(rs, accs)]
    taus_plot  = [t if a >= ACC_THRESH else np.nan for t, a in zip(taus, accs)]
    gd_plot    = [g if a >= ACC_THRESH else np.nan for g, a in zip(gd_means, accs)]
    gdf_plot   = [f if a >= ACC_THRESH else np.nan for f, a in zip(gd_fants, accs)]

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

    BLUE  = '#2980b9'
    RED   = '#e74c3c'
    GREEN = '#27ae60'
    REF   = '#f39c12'

    def _panel(ax, y, color, ylabel, ref_val, ref_label, title, ylim=None):
        ax.plot(x, y, 'o-', color=color, linewidth=2.5, markersize=9, zorder=3)
        # Flag failed models
        for xi, (yi, ai) in enumerate(zip(y, accs)):
            if ai < ACC_THRESH:
                ax.plot(x[xi], yi if not np.isnan(yi) else 0,
                        'x', color='gray', markersize=10, zorder=4)
        ax.axhline(ref_val, color=REF, linestyle='--', linewidth=1.8,
                   label=ref_label, zorder=2)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1.0, zorder=1)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in d_states])

    _panel(axes[0], rs_plot, BLUE,
           'mean Pearson r',
           E_SSM_R, f'E_ssm (d_state=16)  r={E_SSM_R:.3f}',
           'Instantaneous Correlation r vs d_state',
           ylim=(-1.05, 0.3))

    _panel(axes[1], taus_plot, RED,
           'τ_derivative (steps)',
           E_SSM_TAU_DRV, f'E_ssm (d_state=16)  τ={E_SSM_TAU_DRV:.2f}',
           'Anticipatory Lead τ_derivative vs d_state\n'
           '(negative = halt gradient peaks before state collapse)')

    _panel(axes[2], gd_plot, GREEN,
           'gradient disparity (steps)',
           E_SSM_TAU_DRV, f'E_ssm τ_drv ref={E_SSM_TAU_DRV:.2f}',
           'Gradient Disparity (C) vs d_state\n'
           'argmax(dHC/dt) − argmax(−dSE/dt)')

    # Add fraction-anticipatory as a twin axis on the bottom panel
    ax_twin = axes[2].twinx()
    ax_twin.plot(x, gdf_plot, 's--', color='#8e44ad', linewidth=1.5,
                 markersize=6, alpha=0.8, label='frac anticipatory')
    ax_twin.set_ylabel('fraction anticipatory', color='#8e44ad', fontsize=10)
    ax_twin.tick_params(axis='y', labelcolor='#8e44ad')
    ax_twin.set_ylim(0, 1.05)
    ax_twin.legend(fontsize=8, loc='upper right')

    axes[2].set_xlabel('d_state (recurrent state dimension)', fontsize=12)

    # Secondary x-axis showing d_model
    ax_top = axes[0].twiny()
    ax_top.set_xlim(axes[0].get_xlim())
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(d_model_labels, fontsize=8)
    ax_top.set_xlabel('d_model (adjusted for ≈5M params)', fontsize=9)

    fig.suptitle(
        f'Phase 18c: d_state Sweep\n'
        f'Fixed α={FIXED_ALPHA}, β={FIXED_BETA}, γ={FIXED_GAMMA}  '
        f'(E_ssm loss configuration)',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Analysis
# ============================================================

def analyse_and_print(results):
    print("\n" + "=" * 70)
    print("PHASE 18c ANALYSIS — d_state Sweep")
    print("=" * 70)

    print(f"\n  {'d_state':>8s}  {'d_model':>8s}  {'params':>9s}  "
          f"{'acc':>6s}  {'r':>8s}  {'τ_drv':>7s}  {'gd_mean':>8s}  "
          f"{'gd_fant':>8s}")
    print("-" * 80)

    for d in sorted(results.keys()):
        m  = results[d]
        gd = m.get('gradient_disparity') or {}
        gd_m = gd.get('mean', float('nan'))             if gd else float('nan')
        gd_f = gd.get('frac_anticipatory', float('nan')) if gd else float('nan')
        flag = '!' if m['accuracy'] < 0.95 else ' '
        print(f"{flag} {d:>8d}  {m.get('d_model', '?'):>8}  "
              f"{m.get('n_params', 0):>9,}  "
              f"{m['accuracy']:>6.3f}  {m['mean_r']:>8.4f}  "
              f"{m['tau_derivative_mean']:>7.3f}  "
              f"{gd_m:>8.3f}  {gd_f:>8.3f}")

    print(f"\n  E_ssm reference (d_state=16): r={E_SSM_R:.3f}  τ_drv={E_SSM_TAU_DRV:.3f}")

    # Saturation test
    valid = [(d, results[d]) for d in sorted(results.keys())
             if results[d]['accuracy'] >= 0.95]
    if len(valid) >= 3:
        rs   = [m['mean_r']             for _, m in valid]
        taus = [m['tau_derivative_mean'] for _, m in valid]

        # Monotone test: does r decrease (more negative) with d_state?
        r_mono = all(rs[i] >= rs[i+1] for i in range(len(rs)-1))
        t_mono = all(taus[i] >= taus[i+1] for i in range(len(taus)-1))

        print(f"\n  Monotone improvement in r with d_state:     "
              f"{'YES ✓' if r_mono else 'NO ✗  (check for saturation)'}")
        print(f"  Monotone improvement in τ_drv with d_state: "
              f"{'YES ✓' if t_mono else 'NO ✗  (check for saturation)'}")

        # Beat E_ssm?
        beat_r = [d for d, m in valid if m['mean_r'] < E_SSM_R]
        if beat_r:
            print(f"\n  d_state values beating E_ssm r={E_SSM_R:.3f}: {beat_r}")
            for d, m in valid:
                if m['mean_r'] < E_SSM_R:
                    print(f"    d_state={d}  r={m['mean_r']:.4f}  "
                          f"τ_drv={m['tau_derivative_mean']:.3f}")
        else:
            print(f"\n  No d_state beats E_ssm r={E_SSM_R:.3f}  "
                  f"→ state dimensionality is NOT the bottleneck at this loss setting")

    # Bottleneck conclusion
    print("\n  Interpretation:")
    if len(valid) >= 3:
        r_range = max(abs(rs[-1] - rs[0]), 1e-9)
        if r_range < 0.05:
            print("    Flat r curve → d_state is NOT the primary lever; "
                  "loss function balance dominates.")
        elif r_range > 0.15:
            print("    Significant r range → d_state amplifies proprioceptive coupling; "
                  "geometric capacity matters.")
        else:
            print("    Moderate r range → partial geometric bottleneck; "
                  "both d_state and loss function contribute.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 18c: d_state sweep at fixed α=0.05, β=0.10")
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

    print("Phase 18c: d_state Dimensionality Sweep")
    print(f"  Fixed: α={FIXED_ALPHA}  β={FIXED_BETA}  γ={FIXED_GAMMA}")
    print(f"  d_state values: {DSTATE_VALUES}")
    print(f"  Target params:  ≈{TARGET_PARAMS/1e6:.1f}M  (per model)")
    print(f"  Device: {device}")

    # Show d_model plan upfront
    print("\n  Parameter plan:")
    print(f"  {'d_state':>8s}  {'d_model':>8s}  {'params':>10s}")
    for d in DSTATE_VALUES:
        dm = find_d_model(d)
        n  = _param_count(dm, d)
        print(f"  {d:>8d}  {dm:>8d}  {n:>10,}")

    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64,
    )
    parity_train = get_parity_indices(train_ds)
    parity_test  = get_parity_indices(test_ds)
    print(f"  Train parity: {len(parity_train)}/{len(train_ds)}")
    print(f"  Test parity:  {len(parity_test)}/{len(test_ds)}")

    results = run_sweep(
        train_ds, val_ds, test_ds, parity_train, parity_test,
        device, args.results_dir,
        force_retrain=args.force_retrain,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    # Save aggregate
    out_path = os.path.join(args.results_dir, 'phase18c_dstate_sweep.json')
    with open(out_path, 'w') as f:
        json.dump({
            'sweep':  results,
            'config': {
                'alpha':        FIXED_ALPHA,
                'beta':         FIXED_BETA,
                'gamma':        FIXED_GAMMA,
                'dstate_values': DSTATE_VALUES,
                'target_params': TARGET_PARAMS,
                'e_ssm_reference_r':       E_SSM_R,
                'e_ssm_reference_tau_drv': E_SSM_TAU_DRV,
            },
        }, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")

    analyse_and_print(results)

    if not args.skip_figures:
        make_figure(
            results,
            os.path.join(args.figures_dir, 'fig_p18c_dstate_sweep.png'),
        )

    print("\nPhase 18c complete.")


if __name__ == '__main__':
    main()
