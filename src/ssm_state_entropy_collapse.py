"""
Experiment: SSM State Entropy Collapse (Phase 9)

Tests whether SSM recurrent state entropy collapses during reasoning and
whether this collapse is synchronized with rising halt confidence.

If the SSM state entropy shows a monotonic collapse pattern that mirrors
halt confidence, it provides direct evidence that SSM state dynamics
implement "measurement-as-collapse" — the state compresses as the model
resolves uncertainty, and the halt head reads this compression.

Expected pattern for SSM groups with halt training (D, E_ssm):
    State Entropy:   High -> ... -> Drop -> Near-zero
    Answer Entropy:  High -> ... -> Drop -> Near-zero
    Halt Confidence: Low  -> ... -> Rise -> High

Groups: C (SSM+CE), D (SSM+L_th), E_ssm (SSM+halt)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE
from models import create_model, count_parameters
from entropy_halt_correlation import (
    HiddenStateCapture, AnswerProbe, train_probe, evaluate_probe_accuracy,
    get_parity_indices, get_answer_label, load_group_model,
    GROUP_CONFIG, COLORS,
)

# Only SSM groups for this experiment
SSM_GROUPS = ['C', 'D', 'E_ssm']


# ============================================================
# State Entropy Computation
# ============================================================

def compute_state_entropy(states, method='energy'):
    """Compute entropy of SSM state vector at each timestep.

    Args:
        states: [B, L, d_state] SSM hidden states
        method: 'energy' (squared activations) or 'softmax'

    Returns:
        [B, L] entropy values in bits
    """
    if method == 'energy':
        energy = states ** 2
        probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
    elif method == 'softmax':
        probs = F.softmax(states, dim=-1)
    else:
        raise ValueError(f"Unknown method: {method}")
    return -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)


# ============================================================
# Three-Signal Extraction
# ============================================================

@torch.no_grad()
def extract_signals(model, probe, dataset, parity_indices, device,
                    batch_size=64, entropy_method='energy'):
    """Extract state entropy, answer entropy, and halt confidence per example.

    Returns:
        list of dicts, each with 'state_entropy', 'answer_entropy',
        'halt_confidence', 'result_pos', 'answer' trajectories
    """
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, parity_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    examples = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        result_pos = batch['result_pos'].to(device)
        B, L = input_ids.shape

        # Forward pass with hidden state capture
        with HiddenStateCapture(model) as cap:
            outputs = model(input_ids)
            h_dmodel = cap.hidden_states  # [B, L, d_model]

        states_seq = outputs['states_sequence']  # [B, L, d_state]
        halt_conf = outputs['halt_confidence']    # [B, L, 1]

        # State entropy at each timestep
        state_ent = compute_state_entropy(states_seq, method=entropy_method)  # [B, L]

        # Answer entropy from probe
        probe_logits = probe(h_dmodel)  # [B, L, 2]
        probe_probs = F.softmax(probe_logits, dim=-1)
        ans_ent = -(probe_probs * torch.log2(probe_probs + 1e-9)).sum(dim=-1)  # [B, L]

        for b in range(B):
            rp = result_pos[b].item()
            if rp < 3:
                continue

            ans = get_answer_label(targets[b], rp)
            examples.append({
                'result_pos': rp,
                'answer': ans,
                'state_entropy': state_ent[b, :rp + 1].cpu().numpy().tolist(),
                'answer_entropy': ans_ent[b, :rp + 1].cpu().numpy().tolist(),
                'halt_confidence': halt_conf[b, :rp + 1, 0].cpu().numpy().tolist(),
            })

    return examples


# ============================================================
# Correlation & Synchrony Metrics
# ============================================================

def compute_pairwise_correlations(examples):
    """Compute pairwise Pearson correlations between the three signals.

    Returns:
        dict with per-pair stats and per-example correlation values
    """
    pairs = [
        ('state_entropy', 'halt_confidence', 'state_halt'),
        ('state_entropy', 'answer_entropy', 'state_answer'),
        ('answer_entropy', 'halt_confidence', 'answer_halt'),
    ]

    pair_correlations = {name: [] for _, _, name in pairs}

    for ex in examples:
        for sig_a, sig_b, name in pairs:
            a = np.array(ex[sig_a])
            b = np.array(ex[sig_b])
            if len(a) >= 3 and np.std(a) > 1e-8 and np.std(b) > 1e-8:
                r, _ = sp_stats.pearsonr(a, b)
                pair_correlations[name].append(float(r))
            else:
                pair_correlations[name].append(float('nan'))

    # Aggregate
    stats = {}
    for name in pair_correlations:
        rs = [r for r in pair_correlations[name] if not np.isnan(r)]
        if rs:
            stats[name] = {
                'mean_r': float(np.mean(rs)),
                'std_r': float(np.std(rs)),
                'median_r': float(np.median(rs)),
                'fraction_negative': float(np.mean(np.array(rs) < 0)),
                'fraction_significant': float(np.mean(np.abs(rs) > 0.3)),
                'n_valid': len(rs),
            }
        else:
            stats[name] = {'mean_r': float('nan'), 'n_valid': 0}

    return {
        'per_example': pair_correlations,
        'stats': stats,
    }


def compute_collapse_timing(examples):
    """Compute when state entropy drops below 50% of initial and halt exceeds 50%.

    Returns:
        dict with per-example collapse positions and mean lag
    """
    state_collapse_pos = []
    halt_rise_pos = []
    lags = []

    for ex in examples:
        se = np.array(ex['state_entropy'])
        hc = np.array(ex['halt_confidence'])
        L = len(se)

        if L < 3:
            continue

        # State entropy collapse: first position where se < 50% of se[0]
        threshold = se[0] * 0.5
        sc_pos = L  # default: never collapses
        for t in range(L):
            if se[t] < threshold:
                sc_pos = t
                break
        state_collapse_pos.append(sc_pos)

        # Halt confidence rise: first position where hc > 0.5
        hr_pos = L  # default: never rises
        for t in range(L):
            if hc[t] > 0.5:
                hr_pos = t
                break
        halt_rise_pos.append(hr_pos)

        lags.append(hr_pos - sc_pos)

    return {
        'state_collapse_pos': state_collapse_pos,
        'halt_rise_pos': halt_rise_pos,
        'lags': lags,
        'mean_lag': float(np.mean(lags)) if lags else float('nan'),
        'median_lag': float(np.median(lags)) if lags else float('nan'),
        'std_lag': float(np.std(lags)) if lags else float('nan'),
    }


# ============================================================
# Visualization: Figure 12 — Triple-Signal Trajectories
# ============================================================

def plot_state_entropy_collapse(group_data, save_path):
    """1x3 grid: state entropy, answer entropy, halt confidence per SSM group."""
    groups = [g for g in SSM_GROUPS if g in group_data]
    n = len(groups)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5))
    if n == 1:
        axes = [axes]

    for idx, group in enumerate(groups):
        ax = axes[idx]
        data = group_data[group]
        examples = data['examples']
        corr_stats = data['correlations']['stats']

        max_plot_len = 50
        se_traces = []
        ae_traces = []
        hc_traces = []

        for ex in examples:
            L = min(len(ex['state_entropy']), max_plot_len)
            se_traces.append(ex['state_entropy'][:L])
            ae_traces.append(ex['answer_entropy'][:L])
            hc_traces.append(ex['halt_confidence'][:L])

        if not se_traces:
            ax.set_title(f"{group} ({GROUP_CONFIG[group]['label']})\nNo data")
            continue

        # Pad and compute means
        max_len = max(len(t) for t in se_traces)

        def pad_and_mean(traces):
            padded = np.full((len(traces), max_len), np.nan)
            for i, t in enumerate(traces):
                padded[i, :len(t)] = t
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            return mean, std, padded

        se_mean, se_std, _ = pad_and_mean(se_traces)
        ae_mean, ae_std, _ = pad_and_mean(ae_traces)
        hc_mean, hc_std, _ = pad_and_mean(hc_traces)

        positions = np.arange(max_len)

        # Normalize state entropy to [0,1] for visual comparison
        se_max = np.nanmax(se_mean) if np.nanmax(se_mean) > 0 else 1.0
        se_norm = se_mean / se_max
        se_std_norm = se_std / se_max

        # Plot normalized signals
        ax.plot(positions, se_norm, color='#3498db', linewidth=2.5,
                label='State Entropy (norm.)', zorder=3)
        ax.fill_between(positions,
                         np.clip(se_norm - se_std_norm, 0, None),
                         np.clip(se_norm + se_std_norm, None, 1.2),
                         color='#3498db', alpha=0.15)

        ax.plot(positions, ae_mean, color='#2ecc71', linewidth=2.5,
                linestyle='--', label='Answer Entropy', zorder=3)
        ax.fill_between(positions,
                         np.clip(ae_mean - ae_std, 0, None),
                         np.clip(ae_mean + ae_std, None, 1.2),
                         color='#2ecc71', alpha=0.15)

        ax.plot(positions, hc_mean, color='#e74c3c', linewidth=2.5,
                linestyle=':', label='Halt Confidence', zorder=3)
        ax.fill_between(positions,
                         np.clip(hc_mean - hc_std, 0, None),
                         np.clip(hc_mean + hc_std, None, 1.2),
                         color='#e74c3c', alpha=0.15)

        # Annotate correlations
        r_sh = corr_stats.get('state_halt', {}).get('mean_r', float('nan'))
        r_sa = corr_stats.get('state_answer', {}).get('mean_r', float('nan'))
        r_ah = corr_stats.get('answer_halt', {}).get('mean_r', float('nan'))

        annot_lines = []
        if not np.isnan(r_sh):
            annot_lines.append(f"r(state,halt)={r_sh:.3f}")
        if not np.isnan(r_sa):
            annot_lines.append(f"r(state,ans)={r_sa:.3f}")
        if not np.isnan(r_ah):
            annot_lines.append(f"r(ans,halt)={r_ah:.3f}")

        if annot_lines:
            ax.text(0.97, 0.97, '\n'.join(annot_lines),
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85),
                    family='monospace')

        ax.set_xlabel('Sequence Position')
        ax.set_ylabel('Signal Value (normalized)')
        ax.set_ylim(-0.05, 1.15)
        ax.set_title(f"{group} ({GROUP_CONFIG[group]['label']})", fontweight='bold',
                     fontsize=13)
        ax.legend(fontsize=9, loc='center right')

    fig.suptitle('SSM State Entropy Collapse During Reasoning',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Visualization: Figure 13 — Collapse Synchrony
# ============================================================

def plot_collapse_synchrony(group_data, save_path):
    """2-panel: (left) pairwise correlation bars, (right) collapse timing scatter."""
    import pandas as pd

    groups = [g for g in SSM_GROUPS if g in group_data]
    if not groups:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Grouped bar chart of mean pairwise correlations
    pair_names = ['state_halt', 'state_answer', 'answer_halt']
    pair_labels = ['State Ent.\nvs Halt Conf.', 'State Ent.\nvs Ans. Ent.', 'Ans. Ent.\nvs Halt Conf.']

    bar_data = []
    for group in groups:
        stats = group_data[group]['correlations']['stats']
        for pair_name, pair_label in zip(pair_names, pair_labels):
            mean_r = stats.get(pair_name, {}).get('mean_r', float('nan'))
            std_r = stats.get(pair_name, {}).get('std_r', 0.0)
            bar_data.append({
                'Group': f"{group}\n{GROUP_CONFIG[group]['label']}",
                'Pair': pair_label,
                'Mean r': mean_r,
                'Std r': std_r,
                'color': COLORS[group],
            })

    df_bars = pd.DataFrame(bar_data)

    x = np.arange(len(pair_labels))
    width = 0.25
    offsets = np.linspace(-width, width, len(groups))

    for i, group in enumerate(groups):
        grp_label = f"{group}\n{GROUP_CONFIG[group]['label']}"
        grp_data = df_bars[df_bars['Group'] == grp_label]
        means = grp_data['Mean r'].values
        stds = grp_data['Std r'].values
        ax1.bar(x + offsets[i], means, width, yerr=stds,
                color=COLORS[group], alpha=0.8, label=grp_label.replace('\n', ' '),
                capsize=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(pair_labels, fontsize=10)
    ax1.set_ylabel('Mean Pearson r')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.3, color='red', linestyle=':', alpha=0.3)
    ax1.axhline(y=0.3, color='red', linestyle=':', alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_title('Pairwise Signal Correlations', fontweight='bold')

    # Panel 2: Collapse timing scatter
    for group in groups:
        timing = group_data[group]['timing']
        sc = timing['state_collapse_pos']
        hr = timing['halt_rise_pos']
        if sc and hr:
            # Add jitter for visibility
            jitter = np.random.normal(0, 0.15, len(sc))
            ax2.scatter(np.array(sc) + jitter, np.array(hr) + jitter,
                        c=COLORS[group], alpha=0.4, s=20,
                        label=f"{group} ({GROUP_CONFIG[group]['label']})")

    # Identity line
    lim_max = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([0, lim_max], [0, lim_max], 'k--', alpha=0.3, label='synchronous')
    ax2.set_xlabel('State Entropy Collapse Position\n(first t where H < 50% of H[0])')
    ax2.set_ylabel('Halt Confidence Rise Position\n(first t where conf > 0.5)')
    ax2.set_title('Collapse Timing: State Entropy vs Halt Confidence', fontweight='bold')
    ax2.legend(fontsize=9)

    fig.suptitle('SSM State Entropy Collapse Synchrony',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Results Output
# ============================================================

def save_results(group_data, save_path):
    """Save per-group results to JSON."""
    output = {}
    for group, data in group_data.items():
        corr = data['correlations']['stats']
        timing = data['timing']

        # Mean trajectory (state entropy)
        examples = data['examples']
        max_len = max(len(ex['state_entropy']) for ex in examples) if examples else 0
        padded = np.full((len(examples), max_len), np.nan)
        for i, ex in enumerate(examples):
            se = ex['state_entropy']
            padded[i, :len(se)] = se
        mean_traj = np.nanmean(padded, axis=0).tolist() if max_len > 0 else []

        output[group] = {
            'n_examples': len(examples),
            'probe_accuracy': data['probe_accuracy'],
            'entropy_method': data.get('entropy_method', 'energy'),
            'correlations': corr,
            'timing': {
                'mean_lag': timing['mean_lag'],
                'median_lag': timing['median_lag'],
                'std_lag': timing['std_lag'],
            },
            'mean_state_entropy_trajectory': mean_traj,
        }

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSM State Entropy Collapse Experiment (Phase 9)")
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--probe-epochs', type=int, default=10)
    parser.add_argument('--probe-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--entropy-method', default='energy',
                        choices=['energy', 'softmax'],
                        help='State entropy computation method')
    parser.add_argument('--groups', default='C,D,E_ssm',
                        help='Comma-separated SSM groups to evaluate')
    args = parser.parse_args()

    device = args.device
    groups = [g.strip() for g in args.groups.split(',')]
    # Filter to SSM groups only
    groups = [g for g in groups if g in SSM_GROUPS]

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"SSM State Entropy Collapse Experiment")
    print(f"Device: {device}")
    print(f"Groups: {groups}")
    print(f"Entropy method: {args.entropy_method}")

    # Create datasets
    print("\nCreating datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64
    )

    # Get parity-only indices
    print("Filtering parity examples...")
    train_parity = get_parity_indices(train_ds)
    test_parity = get_parity_indices(test_ds)
    print(f"  Train parity: {len(train_parity)}/{len(train_ds)}")
    print(f"  Test parity:  {len(test_parity)}/{len(test_ds)}")

    group_data = {}

    for group in groups:
        cfg = GROUP_CONFIG[group]
        print(f"\n{'='*60}")
        print(f"Group {group} ({cfg['label']})")
        print(f"{'='*60}")

        # Load model
        ckpt_path = os.path.join(args.results_dir, f"group_{group}_model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping")
            continue

        model = load_group_model(group, args.results_dir, device)
        d_model = model.d_model
        print(f"  Loaded model (d_model={d_model}, params={count_parameters(model):,})")

        # Train d_model probe for answer entropy
        print(f"\n  Training d_model probe (dim={d_model})...")
        probe = AnswerProbe(d_model).to(device)
        probe = train_probe(
            model, probe, train_ds, train_parity, device,
            probe_source='d_model', epochs=args.probe_epochs,
            batch_size=args.batch_size, lr=args.probe_lr,
        )
        probe_acc = evaluate_probe_accuracy(
            model, probe, test_ds, test_parity, device, probe_source='d_model'
        )
        print(f"    Probe accuracy at result_pos: {probe_acc*100:.1f}%")

        # Extract three signals
        print(f"\n  Extracting signals (method={args.entropy_method})...")
        examples = extract_signals(
            model, probe, test_ds, test_parity, device,
            batch_size=args.batch_size, entropy_method=args.entropy_method,
        )
        print(f"    Extracted {len(examples)} examples")

        # Compute correlations
        print(f"\n  Computing pairwise correlations...")
        correlations = compute_pairwise_correlations(examples)

        # Print correlation summary
        for pair_name, pair_stats in correlations['stats'].items():
            mr = pair_stats.get('mean_r', float('nan'))
            fn = pair_stats.get('fraction_negative', 0)
            fs = pair_stats.get('fraction_significant', 0)
            nv = pair_stats.get('n_valid', 0)
            print(f"    {pair_name}: r={mr:.4f} (neg={fn*100:.0f}%, "
                  f"|r|>0.3={fs*100:.0f}%, n={nv})")

        # Compute collapse timing
        print(f"\n  Computing collapse timing...")
        timing = compute_collapse_timing(examples)
        print(f"    Mean lag (halt_rise - state_collapse): {timing['mean_lag']:.2f} "
              f"(median={timing['median_lag']:.2f})")

        group_data[group] = {
            'examples': examples,
            'correlations': correlations,
            'timing': timing,
            'probe_accuracy': probe_acc,
            'entropy_method': args.entropy_method,
        }

        # Free memory
        del model, probe
        if device == 'cuda':
            torch.cuda.empty_cache()

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    plot_state_entropy_collapse(
        group_data,
        os.path.join(args.figures_dir, 'fig12_state_entropy_collapse.png'),
    )
    plot_collapse_synchrony(
        group_data,
        os.path.join(args.figures_dir, 'fig13_collapse_synchrony.png'),
    )

    # Save JSON
    save_results(
        group_data,
        os.path.join(args.results_dir, 'ssm_state_entropy_collapse.json'),
    )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: SSM State Entropy Collapse")
    print(f"{'='*60}")
    print(f"\n{'Group':<10s} {'Probe':>8s} {'r(S,H)':>8s} {'r(S,A)':>8s} "
          f"{'r(A,H)':>8s} {'Lag':>6s}")
    print("-" * 55)

    for group in groups:
        if group not in group_data:
            continue
        d = group_data[group]
        stats = d['correlations']['stats']
        r_sh = stats.get('state_halt', {}).get('mean_r', float('nan'))
        r_sa = stats.get('state_answer', {}).get('mean_r', float('nan'))
        r_ah = stats.get('answer_halt', {}).get('mean_r', float('nan'))
        lag = d['timing']['mean_lag']

        print(f"{group:<10s} {d['probe_accuracy']*100:>7.1f}% {r_sh:>8.3f} "
              f"{r_sa:>8.3f} {r_ah:>8.3f} {lag:>6.1f}")

    print(f"\n  r(S,H) = state_entropy vs halt_confidence (expect negative)")
    print(f"  r(S,A) = state_entropy vs answer_entropy (expect positive)")
    print(f"  r(A,H) = answer_entropy vs halt_confidence (expect negative)")
    print(f"  Lag    = mean positions between state collapse and halt rise")
    print(f"\nResults: {args.results_dir}/ssm_state_entropy_collapse.json")
    print(f"Figures: {args.figures_dir}/fig12_*.png, fig13_*.png")


if __name__ == '__main__':
    main()
