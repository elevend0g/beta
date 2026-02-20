"""
Phase 17b: Architectural Proprioception — Multi-Group Control Comparison

Extends Phase 17 by running the same analysis across all three SSM groups
(C, D, E_ssm) to establish that the proprioceptive signal is specific to
halt training rather than a general property of SSM architecture.

Phase 17 result (E_ssm only):
  r(state_entropy, halt_confidence) = -0.836, τ = -2.03

Phase 9 reference values:
  C     (SSM+CE,   no halt training)  r = -0.290
  D     (SSM+Lth,  thermodynamic)     r = -0.725
  E_ssm (SSM+halt, explicit)          r = -0.836

This script reuses Phase 17's analysis functions and adds:
  — 1×3 mean-trajectory comparison panel (fig_p1)
  — 2×3 cross-correlation comparison grid (fig_p2)
  — r distribution + lag histogram across groups (fig_p3)
  — E_ssm individual-example gallery (fig_p4, same as Phase 17)

Figures
-------
  figures/fig_p1b_mean_trajectories.png
  figures/fig_p2b_xcorr_comparison.png
  figures/fig_p3b_r_comparison.png
  figures/fig_p4b_gallery.png

Output
------
  results/phase17b_results.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets
from models import count_parameters
from train import get_device
from entropy_halt_correlation import (
    AnswerProbe, train_probe, evaluate_probe_accuracy,
    get_parity_indices, load_group_model,
)

# Reuse all analysis functions from Phase 17
from phase17_proprioception_repro import (
    extract_signals,
    compute_all_metrics,
    aggregate_metrics,
    select_gallery,
    plot_gallery,
)

# ── Per-group colours / labels ───────────────────────────────
GROUP_COLORS = {'C': '#2ecc71', 'D': '#e74c3c', 'E_ssm': '#e67e22'}
GROUP_LABELS = {
    'C':     'SSM+CE\n(no halt)',
    'D':     'SSM+Lth\n(thermo)',
    'E_ssm': 'SSM+halt\n(explicit)',
}

BLUE  = '#2980b9'
GREEN = '#27ae60'
RED   = '#e74c3c'


# ============================================================
# Figure P1b — 1×3 mean-trajectory comparison
# ============================================================

def plot_mean_trajectories(group_data, save_path):
    """One panel per group showing mean SE (norm.), answer entropy, halt conf."""
    groups = list(group_data.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 5.5),
                             sharey=False)
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        examples = group_data[group]['examples']
        agg      = group_data[group]['agg']
        color    = GROUP_COLORS[group]

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
        pos = np.arange(max_len)

        se_scale = np.nanmax(se_m) or 1.0
        se_n, se_s_n = se_m / se_scale, se_s / se_scale

        ax2 = ax.twinx()

        ax.plot(pos, se_n, color=BLUE,  lw=2.5, label='State Ent. (norm.)')
        ax.fill_between(pos, np.clip(se_n - se_s_n, 0, None),
                        np.clip(se_n + se_s_n, None, 1.3), color=BLUE, alpha=0.15)
        ax.plot(pos, ae_m, color=GREEN, lw=2, ls='--', label='Answer Ent.')
        ax.fill_between(pos, np.clip(ae_m - ae_s, 0, None),
                        np.clip(ae_m + ae_s, None, 1.3), color=GREEN, alpha=0.15)
        ax2.plot(pos, hc_m, color=RED, lw=2, ls=':', label='Halt Conf.')
        ax2.fill_between(pos, np.clip(hc_m - hc_s, 0, None),
                         np.clip(hc_m + hc_s, None, 1.3), color=RED, alpha=0.15)

        inst = agg['instantaneous']
        tlag = agg['threshold_lag']
        ax.text(0.97, 0.97,
                f"r = {inst['mean_r']:.3f} ± {inst['std_r']:.3f}\n"
                f"τ = {tlag['mean']:.2f} ± {tlag['std']:.2f}",
                transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=color, linewidth=1.5, alpha=0.92))

        ax.set_xlabel('Sequence Position', fontsize=11)
        ax.set_ylim(-0.05, 1.3)
        ax2.set_ylim(-0.05, 1.3)
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax2.tick_params(axis='y', labelcolor=RED)
        if ax is axes[0]:
            ax.set_ylabel('Entropy (normalised)', color=BLUE, fontsize=11)
        if ax is axes[-1]:
            ax2.set_ylabel('Halt Confidence', color=RED, fontsize=11)
        ax.set_title(f'{group}  —  {GROUP_LABELS[group].replace(chr(10), "  ")}',
                     fontsize=12, fontweight='bold', color=color)

    legend_handles = [
        mlines.Line2D([], [], color=BLUE,  lw=2.5, label='State Entropy (norm.)'),
        mlines.Line2D([], [], color=GREEN, lw=2,   ls='--', label='Answer Entropy'),
        mlines.Line2D([], [], color=RED,   lw=2,   ls=':',  label='Halt Confidence'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)
    fig.suptitle('Mean Signal Trajectories: SSM Groups C / D / E_ssm',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ============================================================
# Figure P2b — 2×3 cross-correlation comparison grid
# ============================================================

def plot_xcorr_comparison(group_data, save_path):
    """Rows = raw / derivative xcorr.  Cols = C, D, E_ssm."""
    groups = list(group_data.keys())
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 9), sharey='row')
    if n == 1:
        axes = axes.reshape(2, 1)

    row_meta = [
        ('raw_xcorr',   False, 'Raw: cross-corr(SE, HC)'),
        ('deriv_xcorr', True,  'Rate: cross-corr(−dSE/dt, dHC/dt)'),
    ]

    for row, (key, find_max, row_title) in enumerate(row_meta):
        for col, group in enumerate(groups):
            ax    = axes[row][col]
            agg   = group_data[group]['agg']
            color = GROUP_COLORS[group]

            lags = np.array(agg[key]['lags'])
            mean = np.array(agg[key]['mean'])
            std  = np.array(agg[key]['std'])
            peak = agg[key]['peak_lag']

            bars = ax.bar(lags, mean, color=color, alpha=0.55, width=0.6, zorder=3)
            ax.errorbar(lags, mean, yerr=std, fmt='none', color='#2c3e50',
                        capsize=3, lw=1.2, zorder=4)

            peak_idx = np.where(lags == peak)[0]
            if len(peak_idx):
                bars[peak_idx[0]].set_alpha(0.95)
                bars[peak_idx[0]].set_edgecolor('#2c3e50')
                bars[peak_idx[0]].set_linewidth(1.5)

            ax.axhline(0, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.axvline(0, color='gray', ls=':',  lw=0.8, alpha=0.4)
            ax.axvline(peak, color=color, ls='--', lw=1.8, alpha=0.9,
                       label=f'Peak τ = {peak:+d}  ({agg[key]["peak_val"]:.3f})')

            ax.set_xticks(lags)
            ax.set_xlabel('Lag τ (steps)', fontsize=9)
            ax.legend(fontsize=8.5,
                      loc='lower left' if not find_max else 'upper left')

            if col == 0:
                ax.set_ylabel('Norm. cross-correlation', fontsize=10)
            if row == 0:
                ax.set_title(
                    f'{group}  —  {GROUP_LABELS[group].replace(chr(10), "  ")}',
                    fontsize=11, fontweight='bold', color=color)

        # Row label on left margin
        axes[row][0].annotate(
            row_title, xy=(-0.28, 0.5), xycoords='axes fraction',
            ha='center', va='center', fontsize=10, fontweight='bold', rotation=90)

    fig.suptitle(
        'Cross-Correlation Analysis: C vs D vs E_ssm\n'
        'τ > 0 = HC (or dHC/dt) leads SE (or −dSE/dt) — anticipatory collapse',
        fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0.04, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ============================================================
# Figure P3b — r distribution + lag histogram comparison
# ============================================================

def plot_r_comparison(group_data, save_path):
    """
    Left:  per-example Pearson r violin+strip for each group.
    Right: threshold-lag histograms overlaid for each group.
    """
    import pandas as pd

    groups = list(group_data.keys())
    r_rows, lag_rows = [], []
    for group in groups:
        records = group_data[group]['records']
        label   = GROUP_LABELS[group].replace('\n', ' ')
        for rec in records:
            r_val = rec.get('r', float('nan'))
            if not np.isnan(r_val):
                r_rows.append({'Group': label, 'r': r_val})
            lag_rows.append({'Group': label, 'lag': rec['threshold_lag']})

    df_r   = pd.DataFrame(r_rows)
    df_lag = pd.DataFrame(lag_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    palette = {GROUP_LABELS[g].replace('\n', ' '): GROUP_COLORS[g] for g in groups}
    order   = [GROUP_LABELS[g].replace('\n', ' ') for g in groups]

    sns.violinplot(data=df_r, x='Group', y='r', hue='Group', ax=ax1,
                   palette=palette, order=order,
                   inner=None, alpha=0.4, cut=0, legend=False)
    sns.stripplot(data=df_r, x='Group', y='r', hue='Group', ax=ax1,
                  palette=palette, order=order,
                  size=2, alpha=0.35, jitter=0.18, legend=False)

    ax1.axhline(0,    color='gray', ls='--', lw=1, alpha=0.5)
    ax1.axhline(-0.3, color='tomato', ls=':', lw=1, alpha=0.45, label='|r| = 0.3')
    ax1.axhline(-0.8, color='navy',   ls=':', lw=1, alpha=0.45, label='|r| = 0.8')

    for i, group in enumerate(groups):
        mean_r = group_data[group]['agg']['instantaneous']['mean_r']
        ax1.text(i, mean_r + 0.03, f'{mean_r:.3f}', ha='center', va='bottom',
                 fontsize=9.5, fontweight='bold', color=GROUP_COLORS[group])

    ax1.set_ylabel('Pearson r  (state entropy vs halt confidence)', fontsize=11)
    ax1.set_xlabel('')
    ax1.legend(fontsize=9)
    ax1.set_title('Per-Example Correlation Distribution\n'
                  '(more negative = stronger proprioceptive signal)',
                  fontsize=11, fontweight='bold')

    lag_bins = np.arange(-8.5, 9.5, 1)
    for group in groups:
        lag_vals = [r['threshold_lag'] for r in group_data[group]['records']]
        mean_lag = group_data[group]['agg']['threshold_lag']['mean']
        label    = (f"{group}  ({GROUP_LABELS[group].replace(chr(10), '/')})  "
                    f"μ = {mean_lag:.2f}")
        ax2.hist(lag_vals, bins=lag_bins, color=GROUP_COLORS[group], alpha=0.45,
                 label=label, density=True)
        ax2.axvline(mean_lag, color=GROUP_COLORS[group], ls='--', lw=2)

    ax2.axvline(0, color='black', ls='-', lw=1, alpha=0.4, label='τ = 0 (synchronous)')
    ax2.set_xlabel('Threshold lag τ  (halt_rise − state_collapse)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.legend(fontsize=8.5)
    ax2.set_title('Threshold-Lag Distribution\n'
                  '(negative = halt fires BEFORE state collapses)',
                  fontsize=11, fontweight='bold')

    fig.suptitle('Proprioception: C (no halt) vs D (thermo) vs E_ssm (explicit halt)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {save_path}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 17b: Proprioception Multi-Group Comparison')
    parser.add_argument('--groups',       default='C,D,E_ssm')
    parser.add_argument('--results-dir',  default='results')
    parser.add_argument('--figures-dir',  default='figures')
    parser.add_argument('--probe-epochs', type=int,   default=10)
    parser.add_argument('--probe-lr',     type=float, default=1e-3)
    parser.add_argument('--batch-size',   type=int,   default=64)
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(',')]
    device = get_device()
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print('=' * 62)
    print('Phase 17b: Architectural Proprioception — Control Comparison')
    print('=' * 62)
    print(f'Device : {device}')
    print(f'Groups : {groups}')
    print('Phase 9 ref: C r=-0.290 | D r=-0.725 | E_ssm r=-0.836')

    print('\nCreating datasets...')
    train_ds, _, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64)
    train_parity = get_parity_indices(train_ds)
    test_parity  = get_parity_indices(test_ds)
    print(f'  Train parity: {len(train_parity)} / {len(train_ds)}')
    print(f'  Test  parity: {len(test_parity)} / {len(test_ds)}')

    group_data = {}

    for group in groups:
        ckpt = os.path.join(args.results_dir, f'group_{group}_model.pt')
        if not os.path.exists(ckpt):
            print(f'\n  [SKIP] {group}: checkpoint not found at {ckpt}')
            continue

        print(f'\n{"─" * 62}')
        print(f'  Group {group}  ({GROUP_LABELS[group].replace(chr(10), " / ")})')
        print(f'{"─" * 62}')

        model = load_group_model(group, args.results_dir, device)
        print(f'  Params: {count_parameters(model):,}  '
              f'd_model={model.d_model}  d_state={model.d_state}')

        print(f'  Training probe ({args.probe_epochs} epochs)...')
        probe = AnswerProbe(model.d_model).to(device)
        probe = train_probe(model, probe, train_ds, train_parity, device,
                            probe_source='d_model',
                            epochs=args.probe_epochs,
                            batch_size=args.batch_size,
                            lr=args.probe_lr)
        probe_acc = evaluate_probe_accuracy(model, probe, test_ds, test_parity,
                                            device, probe_source='d_model')
        print(f'  Probe accuracy: {probe_acc * 100:.1f}%')

        print('  Extracting signals...')
        examples = extract_signals(model, probe, test_ds, test_parity,
                                   device, batch_size=args.batch_size)
        print(f'  {len(examples)} examples')

        del model, probe
        if device == 'cuda':
            torch.cuda.empty_cache()

        records, lag_range = compute_all_metrics(examples)
        agg = aggregate_metrics(records, lag_range)
        group_data[group] = {'examples': examples, 'records': records, 'agg': agg}

        inst = agg['instantaneous']
        tlag = agg['threshold_lag']
        print(f'  r = {inst["mean_r"]:.4f} ± {inst["std_r"]:.4f}  '
              f'({inst["fraction_negative"]*100:.0f}% neg, '
              f'{inst["fraction_significant"]*100:.0f}% |r|>0.3)')
        print(f'  threshold τ = {tlag["mean"]:.3f} ± {tlag["std"]:.3f}')
        print(f'  raw   xcorr peak τ = {agg["raw_xcorr"]["peak_lag"]:+d}')
        print(f'  deriv xcorr peak τ = {agg["deriv_xcorr"]["peak_lag"]:+d}')

    if not group_data:
        print('No groups processed — exiting.')
        return

    # Summary table
    print(f'\n{"═" * 62}')
    print('  COMPARISON SUMMARY')
    print(f'{"═" * 62}')
    print(f'{"Group":<8} {"mean r":>8} {"frac_neg":>9} {"τ mean":>8} '
          f'{"raw τ*":>7} {"drv τ*":>7}')
    print('─' * 62)
    for group in groups:
        if group not in group_data:
            continue
        agg = group_data[group]['agg']
        inst = agg['instantaneous']
        tlag = agg['threshold_lag']
        print(f'{group:<8} {inst["mean_r"]:>8.4f} '
              f'{inst["fraction_negative"]*100:>8.0f}% '
              f'{tlag["mean"]:>8.3f} '
              f'{agg["raw_xcorr"]["peak_lag"]:>+7d} '
              f'{agg["deriv_xcorr"]["peak_lag"]:>+7d}')
    print(f'{"═" * 62}')
    print('  τ* = peak xcorr lag  (positive = halt leads state)')

    print('\nGenerating figures...')

    plot_mean_trajectories(
        group_data,
        os.path.join(args.figures_dir, 'fig_p1b_mean_trajectories.png'))

    plot_xcorr_comparison(
        group_data,
        os.path.join(args.figures_dir, 'fig_p2b_xcorr_comparison.png'))

    plot_r_comparison(
        group_data,
        os.path.join(args.figures_dir, 'fig_p3b_r_comparison.png'))

    gallery_group = 'E_ssm' if 'E_ssm' in group_data else min(
        group_data, key=lambda g: group_data[g]['agg']['instantaneous']['mean_r'])
    plot_gallery(
        group_data[gallery_group]['examples'],
        group_data[gallery_group]['records'],
        os.path.join(args.figures_dir, 'fig_p4b_gallery.png'))

    out_path = os.path.join(args.results_dir, 'phase17b_results.json')
    with open(out_path, 'w') as f:
        json.dump({g: d['agg'] for g, d in group_data.items()},
                  f, indent=2, default=str)
    print(f'\n  Saved {out_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
