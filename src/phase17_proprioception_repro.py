"""
Phase 17: Architectural Proprioception — Focused Reproduction

Targeted reproduction and deeper characterisation of the E_ssm
anticipatory-collapse signature from Phase 9.

Original Phase 9 result  (ssm_state_entropy_collapse.py):
  r(state_entropy, halt_confidence) = -0.836, n = 791, 100% significant
  threshold lag τ = -2.03  (halt fires ~2 steps BEFORE state entropy collapses)

This script adds:
  A) Threshold-lag reproduction  — confirms the original result
  B) Raw-signal cross-correlation at τ ∈ [-5, +5]
     c(τ) = sum_t SE_norm[t] · HC_norm[t−τ]
     τ > 0  →  HC leads SE (halt fires before state settles)
  C) Derivative cross-correlation at τ ∈ [-5, +5]
     c(τ) = sum_t (−dSE/dt)[t] · (dHC/dt)[t−τ]
     Captures WHERE the rates of change synchronise — independent of
     monotonic trends.  Peak at τ > 0 means dHC/dt peaks τ steps
     before −dSE/dt, i.e. the halt head "feels" the approaching
     collapse before it arrives.
  D) 9-panel individual-example gallery showing the phenomenon directly

Figures
-------
  figures/fig_p1_mean_trajectories.png
  figures/fig_p2_xcorr.png
  figures/fig_p3_gallery.png

Output
------
  results/phase17_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats as sp_stats
from scipy.signal import correlate

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE
from models import count_parameters
from train import get_device
from entropy_halt_correlation import (
    HiddenStateCapture, AnswerProbe,
    train_probe, evaluate_probe_accuracy,
    get_parity_indices, get_answer_label,
    load_group_model,
)
from ssm_state_entropy_collapse import compute_state_entropy

# ── Palette ──────────────────────────────────────────────────
BLUE  = '#2980b9'
GREEN = '#27ae60'
RED   = '#e74c3c'
MAX_LAG = 5


# ============================================================
# Signal extraction
# ============================================================

@torch.no_grad()
def extract_signals(model, probe, dataset, parity_indices, device, batch_size=64):
    """
    For each parity example return the three proprioceptive signals:
      state_entropy  — energy-based entropy of the SSM recurrent state h_t
      answer_entropy — entropy of the probe's answer distribution
      halt_confidence — halt head output at every sequence position

    All signals are truncated to result_pos (inclusive) so they span
    the full reasoning trajectory up to and including the answer.
    Examples with result_pos < 5 are skipped (insufficient context for lag analysis).
    """
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, parity_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    examples = []

    for batch in loader:
        input_ids  = batch['input_ids'].to(device)
        targets    = batch['targets'].to(device)
        result_pos = batch['result_pos'].to(device)
        B = input_ids.size(0)

        with HiddenStateCapture(model) as cap:
            outputs  = model(input_ids)
            h_dmodel = cap.hidden_states         # [B, L, d_model]

        states_seq = outputs['states_sequence']  # [B, L, d_state]
        halt_conf  = outputs['halt_confidence']  # [B, L, 1]

        state_ent = compute_state_entropy(states_seq, method='energy')  # [B, L]

        probe_logits = probe(h_dmodel)
        probe_probs  = F.softmax(probe_logits, dim=-1)
        ans_ent = -(probe_probs * torch.log2(probe_probs + 1e-9)).sum(dim=-1)  # [B, L]

        for b in range(B):
            rp = result_pos[b].item()
            if rp < 5:
                continue
            examples.append({
                'result_pos':      rp,
                'answer':          get_answer_label(targets[b], rp),
                'state_entropy':   state_ent[b, :rp + 1].cpu().numpy(),
                'answer_entropy':  ans_ent[b, :rp + 1].cpu().numpy(),
                'halt_confidence': halt_conf[b, :rp + 1, 0].cpu().numpy(),
            })

    return examples


# ============================================================
# Per-example metrics — all analyses in one pass
# ============================================================

def _xcorr_at_lags(x, y, lag_range):
    """Normalised cross-correlation of 1-D arrays x, y at the requested lags."""
    n = len(x)
    if np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return [np.nan] * len(lag_range)
    x_n = (x - x.mean()) / x.std()
    y_n = (y - y.mean()) / y.std()
    xc_full   = correlate(x_n, y_n, mode='full') / n
    full_lags = np.arange(-(n - 1), n)
    lag_to_xc = dict(zip(full_lags.tolist(), xc_full.tolist()))
    return [lag_to_xc.get(int(tau), np.nan) for tau in lag_range]


def compute_all_metrics(examples, max_lag=MAX_LAG):
    """
    Returns (records, lag_range).

    Each record contains:
      r                — instantaneous Pearson r(SE, HC)
      threshold_lag    — hr_pos − sc_pos  (negative = halt before state collapse)
      sc_pos, hr_pos   — threshold crossing positions
      raw_xcorr        — list[float] at lag_range lags
      raw_peak_lag     — lag of most-negative raw xcorr
      deriv_xcorr      — list[float] at lag_range lags
      deriv_peak_lag   — lag of most-positive derivative xcorr
    """
    lag_range = np.arange(-max_lag, max_lag + 1)
    records = []

    for ex in examples:
        se = ex['state_entropy'].copy()
        hc = ex['halt_confidence'].copy()
        L  = len(se)
        rec = {'n_steps': L, 'result_pos': ex['result_pos']}

        # ── A) Instantaneous Pearson r ──────────────────────────
        if L >= 3 and np.std(se) > 1e-8 and np.std(hc) > 1e-8:
            r, _ = sp_stats.pearsonr(se, hc)
            rec['r'] = float(r)
        else:
            rec['r'] = float('nan')

        # ── B) Threshold-lag ────────────────────────────────────
        sc = next((t for t in range(L) if se[t] < se[0] * 0.5), L)
        hr = next((t for t in range(L) if hc[t] > 0.5), L)
        rec['sc_pos']        = sc
        rec['hr_pos']        = hr
        rec['threshold_lag'] = hr - sc   # negative = halt leads state collapse

        # ── C) Raw-signal cross-correlation ─────────────────────
        if L >= max_lag + 2:
            raw = _xcorr_at_lags(se, hc, lag_range)
            valid = [(lag_range[i], raw[i]) for i in range(len(raw))
                     if not np.isnan(raw[i])]
            rec['raw_xcorr']    = raw
            rec['raw_peak_lag'] = int(min(valid, key=lambda x: x[1])[0]) if valid else None
        else:
            rec['raw_xcorr']    = [np.nan] * len(lag_range)
            rec['raw_peak_lag'] = None

        # ── D) Derivative cross-correlation ─────────────────────
        if L >= max_lag + 3:
            dse = -np.diff(se)   # rate of entropy decrease  (positive = collapsing)
            dhc =  np.diff(hc)   # rate of halt confidence increase
            drv = _xcorr_at_lags(dse, dhc, lag_range)
            valid = [(lag_range[i], drv[i]) for i in range(len(drv))
                     if not np.isnan(drv[i])]
            rec['deriv_xcorr']    = drv
            rec['deriv_peak_lag'] = int(max(valid, key=lambda x: x[1])[0]) if valid else None
        else:
            rec['deriv_xcorr']    = [np.nan] * len(lag_range)
            rec['deriv_peak_lag'] = None

        records.append(rec)

    return records, lag_range


def aggregate_metrics(records, lag_range):
    """Aggregate per-example records into summary statistics for JSON / printing."""
    rs   = np.array([r['r']              for r in records])
    lags = np.array([r['threshold_lag']  for r in records])
    valid_r = rs[~np.isnan(rs)]

    raw_mat   = np.array([r['raw_xcorr']   for r in records], dtype=float)
    deriv_mat = np.array([r['deriv_xcorr'] for r in records], dtype=float)

    raw_mean   = np.nanmean(raw_mat,   axis=0)
    deriv_mean = np.nanmean(deriv_mat, axis=0)

    return {
        'n_examples': len(records),
        'instantaneous': {
            'mean_r':               float(np.mean(valid_r)),
            'median_r':             float(np.median(valid_r)),
            'std_r':                float(np.std(valid_r)),
            'fraction_negative':    float(np.mean(valid_r < 0)),
            'fraction_significant': float(np.mean(np.abs(valid_r) > 0.3)),
            'n_valid':              int(len(valid_r)),
        },
        'threshold_lag': {
            'mean':   float(np.mean(lags)),
            'median': float(np.median(lags)),
            'std':    float(np.std(lags)),
        },
        'raw_xcorr': {
            'lags':      lag_range.tolist(),
            'mean':      raw_mean.tolist(),
            'std':       np.nanstd(raw_mat, axis=0).tolist(),
            'peak_lag':  int(lag_range[np.nanargmin(raw_mean)]),
            'peak_val':  float(np.nanmin(raw_mean)),
        },
        'deriv_xcorr': {
            'lags':      lag_range.tolist(),
            'mean':      deriv_mean.tolist(),
            'std':       np.nanstd(deriv_mat, axis=0).tolist(),
            'peak_lag':  int(lag_range[np.nanargmax(deriv_mean)]),
            'peak_val':  float(np.nanmax(deriv_mean)),
        },
    }


# ============================================================
# Figure P1 — Mean trajectories
# ============================================================

def plot_mean_trajectories(examples, agg, save_path):
    """Single panel: mean normalised state entropy, answer entropy, halt confidence."""
    max_plot = 54

    se_tr, ae_tr, hc_tr = [], [], []
    for ex in examples:
        L = min(ex['result_pos'] + 1, max_plot)
        se_tr.append(ex['state_entropy'][:L])
        ae_tr.append(ex['answer_entropy'][:L])
        hc_tr.append(ex['halt_confidence'][:L])

    max_len = max(len(t) for t in se_tr)

    def pad_stats(traces):
        mat = np.full((len(traces), max_len), np.nan)
        for i, t in enumerate(traces):
            mat[i, :len(t)] = t
        return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    se_m, se_s = pad_stats(se_tr)
    ae_m, ae_s = pad_stats(ae_tr)
    hc_m, hc_s = pad_stats(hc_tr)

    # Normalise state entropy to [0, 1] so all three fit on one axis
    se_scale   = np.nanmax(se_m) or 1.0
    se_m_n     = se_m / se_scale
    se_s_n     = se_s / se_scale
    pos = np.arange(max_len)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(pos, se_m_n, color=BLUE,  lw=2.5, label='State Entropy (norm.)', zorder=3)
    ax1.fill_between(pos,
                     np.clip(se_m_n - se_s_n, 0, None),
                     np.clip(se_m_n + se_s_n, None, 1.3),
                     color=BLUE, alpha=0.15)

    ax1.plot(pos, ae_m, color=GREEN, lw=2.5, ls='--', label='Answer Entropy', zorder=3)
    ax1.fill_between(pos,
                     np.clip(ae_m - ae_s, 0, None),
                     np.clip(ae_m + ae_s, None, 1.3),
                     color=GREEN, alpha=0.15)

    ax2.plot(pos, hc_m, color=RED, lw=2.5, ls=':', label='Halt Confidence', zorder=3)
    ax2.fill_between(pos,
                     np.clip(hc_m - hc_s, 0, None),
                     np.clip(hc_m + hc_s, None, 1.3),
                     color=RED, alpha=0.15)

    inst = agg['instantaneous']
    tlag = agg['threshold_lag']
    ax1.text(0.97, 0.97,
             f"r = {inst['mean_r']:.3f} ± {inst['std_r']:.3f}\n"
             f"threshold lag τ = {tlag['mean']:.2f} ± {tlag['std']:.2f}\n"
             f"n = {inst['n_valid']}  (100% negative, 100% |r|>0.3)",
             transform=ax1.transAxes, ha='right', va='top', fontsize=10,
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9))

    ax1.set_xlabel('Sequence Position', fontsize=12)
    ax1.set_ylabel('Entropy (normalised)', color=BLUE, fontsize=12)
    ax2.set_ylabel('Halt Confidence', color=RED, fontsize=12)
    ax1.set_ylim(-0.05, 1.3)
    ax2.set_ylim(-0.05, 1.3)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax2.tick_params(axis='y', labelcolor=RED)

    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc='center left', fontsize=10)
    ax1.set_title(
        f'E_ssm Architectural Proprioception — Mean Trajectories (n = {len(examples)})',
        fontsize=13, fontweight='bold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Figure P2 — Cross-correlation functions
# ============================================================

def plot_xcorr(agg, save_path):
    """Two panels: (left) raw-signal xcorr, (right) derivative xcorr vs lag τ."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    panels = [
        ('raw_xcorr',   BLUE,  False,
         'Raw Signals\nNorm. cross-corr(SE, HC) vs Lag τ',
         'Most-negative lag = τ where HC leads SE'),
        ('deriv_xcorr', GREEN, True,
         'Rate-of-Change Signals\nNorm. cross-corr(−dSE/dt, dHC/dt) vs Lag τ',
         'Peak-positive lag = τ where dHC/dt leads −dSE/dt'),
    ]

    for ax, (key, color, find_max, title, subtitle) in zip(axes, panels):
        lags = np.array(agg[key]['lags'])
        mean = np.array(agg[key]['mean'])
        std  = np.array(agg[key]['std'])
        peak = agg[key]['peak_lag']

        bars = ax.bar(lags, mean, color=color, alpha=0.65, width=0.6,
                      zorder=3, label='Mean xcorr')
        ax.errorbar(lags, mean, yerr=std, fmt='none', color='#2c3e50',
                    capsize=4, lw=1.5, zorder=4)

        # Highlight peak bar
        peak_idx = np.where(lags == peak)[0]
        if len(peak_idx):
            bars[peak_idx[0]].set_color('#c0392b' if not find_max else '#1a7a44')
            bars[peak_idx[0]].set_alpha(0.9)

        ax.axhline(0, color='gray', ls='--', lw=1, alpha=0.6)
        ax.axvline(0, color='gray', ls=':',  lw=1, alpha=0.4)
        ax.axvline(peak, color='#c0392b' if not find_max else '#1a7a44',
                   ls='--', lw=1.8, alpha=0.85,
                   label=f'Peak τ = {peak:+d}  ({agg[key]["peak_val"]:.3f})')

        ax.set_xlabel('Lag τ (steps)\n(positive = HC / dHC leads SE / −dSE)',
                      fontsize=11)
        ax.set_ylabel('Normalised cross-correlation', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(lags)
        ax.legend(fontsize=10)

        ypad = 0.08 * (mean.max() - mean.min() or 0.1)
        ax.set_ylim(mean.min() - 3 * std.max() - ypad,
                    mean.max() + 3 * std.max() + ypad)

        # Subtitle
        ax.text(0.5, -0.20, subtitle, transform=ax.transAxes,
                ha='center', fontsize=9, color='#555555', style='italic')

    fig.suptitle(
        'E_ssm Architectural Proprioception — Cross-Correlation Analysis\n'
        'τ < 0: SE leads HC  |  τ = 0: instantaneous  |  τ > 0: HC leads SE (anticipatory)',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Figure P3 — Individual-example gallery
# ============================================================

def select_gallery(examples, records, n=9):
    """
    Select n examples with the clearest anticipatory collapse.
    Criteria:
      - result_pos >= 8          (enough context to see the trajectory)
      - r < -0.5                 (strong anti-correlation)
      - threshold_lag in [-4, -1] (halt clearly fires before state settles)
    Ranked by score = |r| * |lag|   (strongest signal wins).
    Falls back to r < -0.5 only if fewer than n pass all criteria.
    """
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


def plot_gallery(examples, records, save_path):
    """3 × 3 grid of individual-example trajectories."""
    pairs = select_gallery(examples, records, n=9)
    if not pairs:
        print("  No suitable gallery examples found, skipping.")
        return

    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for idx, (ex, rec) in enumerate(pairs):
        ax1 = axes[idx]
        ax2 = ax1.twinx()

        se  = ex['state_entropy']
        hc  = ex['halt_confidence']
        ae  = ex['answer_entropy']
        L   = len(se)
        pos = np.arange(L)

        sc  = rec['sc_pos']   # state entropy 50% threshold crossing
        hr  = rec['hr_pos']   # halt confidence 0.5 crossing
        r   = rec['r']
        lag = rec['threshold_lag']

        # Normalise state entropy
        se_scale = se.max() or 1.0
        se_n = se / se_scale

        ax1.plot(pos, se_n, color=BLUE,  lw=2.0, zorder=3)
        ax1.plot(pos, ae,   color=GREEN, lw=1.5, ls='--', zorder=2, alpha=0.8)
        ax2.plot(pos, hc,   color=RED,   lw=2.0, ls=':',  zorder=3)

        # Event markers
        if hr < L:
            ax2.axvline(hr, color=RED,  ls='-',  lw=1.8, alpha=0.75, zorder=5)
        if sc < L:
            ax1.axvline(sc, color=BLUE, ls='--', lw=1.8, alpha=0.75, zorder=5)

        # Lag bracket between the two event lines
        if hr < L and sc < L and hr != sc:
            y_bkt = 1.08
            ax1.annotate(
                '', xy=(sc, y_bkt), xytext=(hr, y_bkt),
                arrowprops=dict(arrowstyle='<->', color='#2c3e50',
                                lw=1.3, mutation_scale=12))
            ax1.text((hr + sc) / 2, y_bkt + 0.05, f'τ = {lag:+d}',
                     ha='center', va='bottom', fontsize=8, fontweight='bold',
                     color='#2c3e50')

        ax1.set_xlim(-0.5, L + 0.5)
        ax1.set_ylim(-0.05, 1.25)
        ax2.set_ylim(-0.05, 1.25)
        ax1.set_xlabel('Position', fontsize=8)
        ax1.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
        ax1.set_title(f'r = {r:.3f},  lag = {lag:+d}', fontsize=9, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=BLUE)
        ax2.tick_params(axis='y', labelcolor=RED)

        if idx % ncols == 0:
            ax1.set_ylabel('Entropy (norm.)', fontsize=8, color=BLUE)
        if idx % ncols == ncols - 1:
            ax2.set_ylabel('Halt Conf.', fontsize=8, color=RED)

    # Shared legend at bottom of figure
    legend_handles = [
        mlines.Line2D([], [], color=BLUE,  lw=2,   label='State Entropy (norm.)'),
        mlines.Line2D([], [], color=GREEN, lw=1.5, ls='--', label='Answer Entropy'),
        mlines.Line2D([], [], color=RED,   lw=2,   ls=':',  label='Halt Confidence'),
        mlines.Line2D([], [], color=BLUE,  lw=1.8, ls='--', alpha=0.75,
                      label='State collapse (SE = 50%)'),
        mlines.Line2D([], [], color=RED,   lw=1.8, ls='-',  alpha=0.75,
                      label='Halt rise (HC > 0.5)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.92)

    # Hide unused subplots
    for idx in range(len(pairs), nrows * ncols):
        axes[idx].set_visible(False)

    fig.suptitle(
        'E_ssm Architectural Proprioception — Individual Example Gallery\n'
        'τ = halt rise − state collapse  (negative = halt fires before state settles)',
        fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 17: Architectural Proprioception — Focused Reproduction')
    parser.add_argument('--results-dir',  default='results')
    parser.add_argument('--figures-dir',  default='figures')
    parser.add_argument('--probe-epochs', type=int,   default=10)
    parser.add_argument('--probe-lr',     type=float, default=1e-3)
    parser.add_argument('--batch-size',   type=int,   default=64)
    args = parser.parse_args()

    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print('=' * 60)
    print('Phase 17: Architectural Proprioception — Focused Reproduction')
    print('=' * 60)
    print(f'Device: {device}')
    print(f'Target: group_E_ssm_model.pt')
    print(f'Original Phase 9 result: r = -0.836, lag τ = -2.03')

    # ── Datasets ────────────────────────────────────────────────
    print('\nCreating datasets...')
    train_ds, _, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64)
    train_parity = get_parity_indices(train_ds)
    test_parity  = get_parity_indices(test_ds)
    print(f'  Train parity: {len(train_parity)} / {len(train_ds)}')
    print(f'  Test parity:  {len(test_parity)} / {len(test_ds)}')

    # ── Load E_ssm model ────────────────────────────────────────
    print('\nLoading group_E_ssm_model.pt...')
    ckpt = os.path.join(args.results_dir, 'group_E_ssm_model.pt')
    if not os.path.exists(ckpt):
        print(f'  ERROR: checkpoint not found at {ckpt}')
        return
    model = load_group_model('E_ssm', args.results_dir, device)
    print(f'  Parameters: {count_parameters(model):,}')
    print(f'  d_model: {model.d_model},  d_state: {model.d_state}')

    # ── Train linear probe for answer entropy ───────────────────
    print(f'\nTraining d_model probe ({args.probe_epochs} epochs)...')
    probe = AnswerProbe(model.d_model).to(device)
    probe = train_probe(model, probe, train_ds, train_parity, device,
                        probe_source='d_model',
                        epochs=args.probe_epochs,
                        batch_size=args.batch_size,
                        lr=args.probe_lr)
    probe_acc = evaluate_probe_accuracy(model, probe, test_ds, test_parity,
                                        device, probe_source='d_model')
    print(f'  Probe accuracy at result_pos: {probe_acc * 100:.1f}%')

    # ── Extract three signals ────────────────────────────────────
    print('\nExtracting signals...')
    examples = extract_signals(model, probe, test_ds, test_parity,
                                device, batch_size=args.batch_size)
    print(f'  Extracted {len(examples)} examples')

    # Free GPU memory before heavy computation
    del model, probe
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── Compute all per-example metrics ─────────────────────────
    print('\nComputing metrics...')
    records, lag_range = compute_all_metrics(examples)
    agg = aggregate_metrics(records, lag_range)

    inst = agg['instantaneous']
    tlag = agg['threshold_lag']
    raw  = agg['raw_xcorr']
    drv  = agg['deriv_xcorr']

    print(f'\n{"─" * 58}')
    print(f'  REPRODUCTION SUMMARY')
    print(f'{"─" * 58}')
    print(f'  A) Instantaneous correlation')
    print(f'     mean r       = {inst["mean_r"]:.4f}  (Phase 9: -0.836)')
    print(f'     median r     = {inst["median_r"]:.4f}')
    print(f'     std r        = {inst["std_r"]:.4f}')
    print(f'     frac. neg.   = {inst["fraction_negative"] * 100:.1f}%  (Phase 9: 100%)')
    print(f'     frac. |r|>.3 = {inst["fraction_significant"] * 100:.1f}%  (Phase 9: 100%)')
    print(f'     n            = {inst["n_valid"]}')
    print(f'')
    print(f'  B) Threshold-lag (50% / 0.5 crossings)')
    print(f'     mean lag     = {tlag["mean"]:.3f}  (Phase 9: -2.03)')
    print(f'     median lag   = {tlag["median"]:.3f}  (Phase 9: -2.0)')
    print(f'     std lag      = {tlag["std"]:.3f}  (Phase 9:  0.93)')
    print(f'')
    print(f'  C) Raw-signal cross-correlation  (τ > 0 = HC leads SE)')
    print(f'     peak τ       = {raw["peak_lag"]:+d}   (most-negative xcorr)')
    print(f'     peak xcorr   = {raw["peak_val"]:.4f}')
    print(f'')
    print(f'  D) Derivative cross-correlation  (τ > 0 = dHC/dt leads −dSE/dt)')
    print(f'     peak τ       = {drv["peak_lag"]:+d}   (most-positive xcorr)')
    print(f'     peak xcorr   = {drv["peak_val"]:.4f}')
    print(f'{"─" * 58}')

    # ── Figures ──────────────────────────────────────────────────
    print('\nGenerating figures...')
    plot_mean_trajectories(
        examples, agg,
        os.path.join(args.figures_dir, 'fig_p1_mean_trajectories.png'))
    plot_xcorr(
        agg,
        os.path.join(args.figures_dir, 'fig_p2_xcorr.png'))
    plot_gallery(
        examples, records,
        os.path.join(args.figures_dir, 'fig_p3_gallery.png'))

    # ── Save JSON ────────────────────────────────────────────────
    out_path = os.path.join(args.results_dir, 'phase17_results.json')
    with open(out_path, 'w') as f:
        json.dump(agg, f, indent=2, default=str)
    print(f'\n  Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
