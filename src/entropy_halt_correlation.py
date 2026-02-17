"""
Experiment: Entropy-Halt Correlation

Tests whether halt confidence tracks genuine answer uncertainty reduction
or just surface-level pattern matching (e.g., detecting 'Result:' tokens).

Method:
  1. Train linear probes (frozen model) to predict binary answer from hidden states
  2. Compute answer_entropy[t] = H(softmax(probe(h_t))) at each timestep
  3. Correlate with halt_confidence[t] using Pearson correlation
  4. Negative correlation = halt tracks real uncertainty collapse
  5. No correlation = halt uses surface features

Groups: A, B, C, D, E_trans, E_ssm (all 6 trained models)
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE, VOCAB
from models import create_model, count_parameters

# ============================================================
# Configuration
# ============================================================

GROUP_CONFIG = {
    'A':       {'arch_group': 'A', 'is_ssm': False, 'label': 'Trans+CE'},
    'B':       {'arch_group': 'B', 'is_ssm': False, 'label': 'Trans+L_th'},
    'C':       {'arch_group': 'C', 'is_ssm': True,  'label': 'SSM+CE'},
    'D':       {'arch_group': 'D', 'is_ssm': True,  'label': 'SSM+L_th'},
    'E_trans': {'arch_group': 'A', 'is_ssm': False, 'label': 'Trans+halt'},
    'E_ssm':   {'arch_group': 'C', 'is_ssm': True,  'label': 'SSM+halt'},
}

COLORS = {
    'A': '#95a5a6', 'B': '#3498db', 'C': '#2ecc71', 'D': '#e74c3c',
    'E_trans': '#9b59b6', 'E_ssm': '#e67e22',
}


# ============================================================
# Hidden State Capture (forward hook, no model modifications)
# ============================================================

class HiddenStateCapture:
    """Context manager that hooks model.norm to capture post-LayerNorm hidden states."""

    def __init__(self, model):
        self.model = model
        self.hidden_states = None
        self._handle = None

    def _hook_fn(self, module, input, output):
        self.hidden_states = output.detach()

    def __enter__(self):
        self._handle = self.model.norm.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ============================================================
# Linear Probe
# ============================================================

class AnswerProbe(nn.Module):
    """Linear probe: predicts P(answer=0), P(answer=1) from hidden state."""

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, h):
        """h: [..., input_dim] -> logits: [..., 2]"""
        return self.linear(h)


# ============================================================
# Data Helpers
# ============================================================

def get_parity_indices(dataset):
    """Return indices of parity-only examples (binary answer: 0 or 1).

    Excludes arithmetic examples whose answers may be multi-digit.
    """
    token_0 = VOCAB['0']
    token_1 = VOCAB['1']
    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        rp = item['result_pos'].item()
        targets = item['targets']
        if rp < len(targets):
            ans_token = targets[rp].item()
            if ans_token in (token_0, token_1):
                indices.append(i)
    return indices


def get_answer_label(targets, result_pos):
    """Extract binary answer (0 or 1) from targets at result_pos."""
    ans_token = targets[result_pos].item()
    return 0 if ans_token == VOCAB['0'] else 1


# ============================================================
# Model Loading
# ============================================================

def load_group_model(group, results_dir, device):
    """Load trained model checkpoint for any of the 6 groups."""
    cfg = GROUP_CONFIG[group]
    model = create_model(cfg['arch_group'], VOCAB_SIZE, device=device)
    ckpt_path = os.path.join(results_dir, f"group_{group}_model.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


# ============================================================
# Probe Training
# ============================================================

def train_probe(model, probe, dataset, parity_indices, device,
                probe_source='d_model', epochs=10, batch_size=64, lr=1e-3):
    """Train a linear probe on frozen model hidden states.

    Args:
        probe_source: 'd_model' for post-norm hidden states, 'd_state' for SSM states
    """
    subset = Subset(dataset, parity_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            result_pos = batch['result_pos'].to(device)
            B, L = input_ids.shape

            # Get hidden states
            with torch.no_grad():
                if probe_source == 'd_model':
                    with HiddenStateCapture(model) as cap:
                        outputs = model(input_ids)
                        h = cap.hidden_states  # [B, L, d_model]
                else:  # d_state
                    outputs = model(input_ids)
                    h = outputs['states_sequence']  # [B, L, d_state]

            # Extract answer labels: [B]
            answer_labels = torch.zeros(B, dtype=torch.long, device=device)
            for b in range(B):
                rp = result_pos[b].item()
                answer_labels[b] = get_answer_label(targets[b], rp)

            # Probe at all timesteps: [B, L, 2]
            probe_logits = probe(h)

            # Broadcast answer label over sequence: [B*L]
            labels_expanded = answer_labels.unsqueeze(1).expand(B, L).reshape(-1)
            logits_flat = probe_logits.reshape(-1, 2)

            loss = F.cross_entropy(logits_flat, labels_expanded)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"      Probe epoch {epoch}: loss={total_loss/n_batches:.4f}")

    probe.eval()
    return probe


def evaluate_probe_accuracy(model, probe, dataset, parity_indices, device,
                            probe_source='d_model', batch_size=64):
    """Sanity check: probe accuracy at result_pos."""
    subset = Subset(dataset, parity_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    probe.eval()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            result_pos = batch['result_pos'].to(device)
            B = input_ids.size(0)

            if probe_source == 'd_model':
                with HiddenStateCapture(model) as cap:
                    outputs = model(input_ids)
                    h = cap.hidden_states
            else:
                outputs = model(input_ids)
                h = outputs['states_sequence']

            for b in range(B):
                rp = result_pos[b].item()
                ans = get_answer_label(targets[b], rp)
                pred = probe(h[b, rp, :].unsqueeze(0)).argmax(dim=-1).item()
                if pred == ans:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0.0


# ============================================================
# Correlation Computation
# ============================================================

@torch.no_grad()
def compute_correlations(model, probes, dataset, parity_indices, device,
                         batch_size=64):
    """Compute per-example Pearson correlation between answer entropy and halt confidence.

    Args:
        probes: dict mapping probe_source ('d_model', 'd_state') to trained AnswerProbe

    Returns:
        dict with 'per_example' list and 'aggregate' stats
    """
    subset = Subset(dataset, parity_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    per_example = []

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        result_pos = batch['result_pos'].to(device)
        B, L = input_ids.shape

        # Forward pass with hook
        with HiddenStateCapture(model) as cap:
            outputs = model(input_ids)
            h_dmodel = cap.hidden_states  # [B, L, d_model]

        halt_conf = outputs['halt_confidence']  # [B, L, 1]
        states_seq = outputs.get('states_sequence')  # [B, L, d_state] or None

        for b in range(B):
            rp = result_pos[b].item()
            ans = get_answer_label(targets[b], rp)

            # Need at least 3 positions for meaningful correlation
            if rp < 3:
                continue

            # Extract halt confidence trajectory up to result_pos (inclusive)
            halt_traj = halt_conf[b, :rp + 1, 0].cpu().numpy()

            example_result = {
                'result_pos': rp,
                'answer': ans,
                'halt_confidence': halt_traj.tolist(),
            }

            # Compute answer entropy for each probe
            for source, probe in probes.items():
                if source == 'd_model':
                    h = h_dmodel[b, :rp + 1, :]
                elif source == 'd_state' and states_seq is not None:
                    h = states_seq[b, :rp + 1, :]
                else:
                    continue

                probe_logits = probe(h)  # [rp+1, 2]
                probe_probs = F.softmax(probe_logits, dim=-1)
                ans_entropy = -(probe_probs * torch.log2(probe_probs + 1e-9)).sum(dim=-1)
                ans_entropy_np = ans_entropy.cpu().numpy()

                example_result[f'answer_entropy_{source}'] = ans_entropy_np.tolist()

                # Pearson correlation
                if len(halt_traj) >= 3 and np.std(halt_traj) > 1e-8 and np.std(ans_entropy_np) > 1e-8:
                    r, p = sp_stats.pearsonr(ans_entropy_np, halt_traj)
                    example_result[f'pearson_r_{source}'] = float(r)
                    example_result[f'pearson_p_{source}'] = float(p)
                else:
                    example_result[f'pearson_r_{source}'] = float('nan')
                    example_result[f'pearson_p_{source}'] = float('nan')

            per_example.append(example_result)

    # Aggregate statistics
    aggregate = {}
    for source in probes.keys():
        rs = [ex[f'pearson_r_{source}'] for ex in per_example
              if not np.isnan(ex.get(f'pearson_r_{source}', float('nan')))]
        if rs:
            aggregate[f'{source}_mean_r'] = float(np.mean(rs))
            aggregate[f'{source}_std_r'] = float(np.std(rs))
            aggregate[f'{source}_median_r'] = float(np.median(rs))
            aggregate[f'{source}_fraction_negative'] = float(np.mean(np.array(rs) < 0))
            aggregate[f'{source}_fraction_significant'] = float(np.mean(np.abs(rs) > 0.3))
            aggregate[f'{source}_n_valid'] = len(rs)

    return {'per_example': per_example, 'aggregate': aggregate}


# ============================================================
# Visualization: Figure 10 — Trajectory Plots
# ============================================================

def plot_entropy_halt_trajectories(all_results, save_path):
    """2x3 grid: answer entropy + halt confidence over sequence position per group."""
    groups = [g for g in ['A', 'B', 'C', 'D', 'E_trans', 'E_ssm'] if g in all_results]
    n_groups = len(groups)
    ncols = min(3, n_groups)
    nrows = (n_groups + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, group in enumerate(groups):
        ax1 = axes[idx]
        res = all_results[group]
        examples = res['per_example']

        # Compute mean trajectories (align to same length by truncating to min length)
        max_plot_len = 45  # enough for 8-bit parity
        entropy_traces = []
        halt_traces = []

        for ex in examples:
            ent = ex.get('answer_entropy_d_model', [])
            halt = ex.get('halt_confidence', [])
            if len(ent) >= 3:
                entropy_traces.append(ent[:max_plot_len])
                halt_traces.append(halt[:max_plot_len])

        if not entropy_traces:
            ax1.set_title(f"{group} ({GROUP_CONFIG[group]['label']})\nNo data")
            continue

        # Pad traces to same length for averaging
        max_len = max(len(t) for t in entropy_traces)
        padded_ent = np.full((len(entropy_traces), max_len), np.nan)
        padded_halt = np.full((len(halt_traces), max_len), np.nan)
        for i, (e, h) in enumerate(zip(entropy_traces, halt_traces)):
            padded_ent[i, :len(e)] = e
            padded_halt[i, :len(h)] = h

        mean_ent = np.nanmean(padded_ent, axis=0)
        mean_halt = np.nanmean(padded_halt, axis=0)
        positions = np.arange(max_len)

        # Plot 3 individual examples (thin, alpha)
        for i in range(min(3, len(entropy_traces))):
            t = np.arange(len(entropy_traces[i]))
            ax1.plot(t, entropy_traces[i], color='#3498db', alpha=0.15, linewidth=0.8)

        # Mean trajectory
        ax1.plot(positions, mean_ent, color='#3498db', linewidth=2.5, label='Answer Entropy')
        ax1.set_ylabel('Answer Entropy (bits)', color='#3498db')
        ax1.set_ylim(-0.05, 1.1)
        ax1.tick_params(axis='y', labelcolor='#3498db')

        # Halt confidence on right y-axis
        ax2 = ax1.twinx()
        for i in range(min(3, len(halt_traces))):
            t = np.arange(len(halt_traces[i]))
            ax2.plot(t, halt_traces[i], color='#e74c3c', alpha=0.15, linewidth=0.8)
        ax2.plot(positions, mean_halt, color='#e74c3c', linewidth=2.5,
                 linestyle='--', label='Halt Confidence')
        ax2.set_ylabel('Halt Confidence', color='#e74c3c')
        ax2.set_ylim(-0.05, 1.1)
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        # Annotations
        mean_r = res['aggregate'].get('d_model_mean_r', float('nan'))
        r_str = f"r = {mean_r:.3f}" if not np.isnan(mean_r) else "r = N/A"
        ax1.text(0.95, 0.95, r_str, transform=ax1.transAxes,
                 ha='right', va='top', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax1.set_xlabel('Sequence Position')
        ax1.set_title(f"{group} ({GROUP_CONFIG[group]['label']})", fontweight='bold')

    # Hide unused subplots
    for idx in range(n_groups, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Answer Entropy vs Halt Confidence Trajectories', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Visualization: Figure 11 — Correlation Distributions
# ============================================================

def plot_entropy_halt_correlation(all_results, save_path):
    """Violin+strip plot of per-example Pearson r, plus scatter."""
    groups = [g for g in ['A', 'B', 'C', 'D', 'E_trans', 'E_ssm'] if g in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel 1: Violin + strip plot of Pearson r per group
    plot_data = []
    plot_groups = []
    for group in groups:
        rs = [ex.get('pearson_r_d_model', float('nan'))
              for ex in all_results[group]['per_example']]
        valid_rs = [r for r in rs if not np.isnan(r)]
        plot_data.extend(valid_rs)
        plot_groups.extend([f"{group}\n{GROUP_CONFIG[group]['label']}"] * len(valid_rs))

    if plot_data:
        import pandas as pd
        df = pd.DataFrame({'Pearson r': plot_data, 'Group': plot_groups})

        palette = {f"{g}\n{GROUP_CONFIG[g]['label']}": COLORS[g] for g in groups}
        sns.violinplot(data=df, x='Group', y='Pearson r', hue='Group', ax=ax1,
                       palette=palette, inner=None, alpha=0.4, cut=0, legend=False)
        sns.stripplot(data=df, x='Group', y='Pearson r', hue='Group', ax=ax1,
                      palette=palette, size=2, alpha=0.4, jitter=0.2, legend=False)

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=-0.3, color='red', linestyle=':', alpha=0.3, label='r = -0.3 threshold')
        ax1.set_title('Per-Example Correlation Distribution', fontweight='bold')
        ax1.set_ylabel('Pearson r (answer entropy vs halt confidence)')
        ax1.legend(fontsize=9)

        # Add mean annotations
        unique_groups = df['Group'].unique()
        for i, grp in enumerate(unique_groups):
            grp_data = df[df['Group'] == grp]['Pearson r']
            mean_r = grp_data.mean()
            ax1.annotate(f'{mean_r:.3f}', xy=(i, mean_r), fontsize=9,
                         ha='center', va='bottom', fontweight='bold',
                         color='black',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # Panel 2: Scatter of mean answer entropy vs mean halt confidence per example
    for group in groups:
        examples = all_results[group]['per_example']
        mean_ents = []
        mean_halts = []
        for ex in examples:
            ent = ex.get('answer_entropy_d_model', [])
            halt = ex.get('halt_confidence', [])
            if len(ent) >= 3:
                mean_ents.append(np.mean(ent))
                mean_halts.append(np.mean(halt))

        if mean_ents:
            ax2.scatter(mean_ents, mean_halts, c=COLORS[group], alpha=0.3, s=15,
                        label=f"{group} ({GROUP_CONFIG[group]['label']})")

    ax2.set_xlabel('Mean Answer Entropy (bits)')
    ax2.set_ylabel('Mean Halt Confidence')
    ax2.set_title('Answer Entropy vs Halt Confidence (per example)', fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    fig.suptitle('Entropy-Halt Correlation Analysis', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_path}")


# ============================================================
# Results Output
# ============================================================

def interpret_correlation(mean_r, frac_neg):
    """Categorize the correlation pattern."""
    if np.isnan(mean_r):
        return "insufficient_data"
    if mean_r < -0.3 and frac_neg > 0.7:
        return "genuine_tracking"
    if abs(mean_r) < 0.15:
        return "no_correlation"
    if mean_r > 0.3:
        return "positive_correlation"
    if mean_r < -0.15:
        return "weak_tracking"
    return "ambiguous"


def save_results(all_results, probe_accuracies, save_path):
    """Save per-group correlation statistics to JSON."""
    output = {}
    for group, res in all_results.items():
        agg = res['aggregate']
        group_out = {
            'n_examples': len(res['per_example']),
        }

        for source in ['d_model', 'd_state']:
            key = f'{source}_mean_r'
            if key in agg:
                mean_r = agg[f'{source}_mean_r']
                frac_neg = agg.get(f'{source}_fraction_negative', 0)
                group_out[f'{source}_probe'] = {
                    'probe_accuracy': probe_accuracies.get(f'{group}_{source}', None),
                    'mean_pearson_r': mean_r,
                    'std_pearson_r': agg[f'{source}_std_r'],
                    'median_pearson_r': agg[f'{source}_median_r'],
                    'fraction_negative': frac_neg,
                    'fraction_significant': agg[f'{source}_fraction_significant'],
                    'n_valid': agg[f'{source}_n_valid'],
                    'interpretation': interpret_correlation(mean_r, frac_neg),
                }

        output[group] = group_out

    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Entropy-Halt Correlation Experiment")
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--probe-epochs', type=int, default=10)
    parser.add_argument('--probe-lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--groups', default='A,B,C,D,E_trans,E_ssm',
                        help='Comma-separated list of groups to evaluate')
    args = parser.parse_args()

    device = args.device
    groups = [g.strip() for g in args.groups.split(',')]
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Groups: {groups}")

    # Create datasets (identical to training)
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

    all_results = {}
    probe_accuracies = {}

    for group in groups:
        if group not in GROUP_CONFIG:
            print(f"\nSkipping unknown group: {group}")
            continue

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

        # Create and train probes
        probes = {}

        # d_model probe (all groups)
        print(f"\n  Training d_model probe (dim={d_model})...")
        probe_dm = AnswerProbe(d_model)
        probe_dm = train_probe(
            model, probe_dm, train_ds, train_parity, device,
            probe_source='d_model', epochs=args.probe_epochs,
            batch_size=args.batch_size, lr=args.probe_lr,
        )
        acc_dm = evaluate_probe_accuracy(
            model, probe_dm, test_ds, test_parity, device, probe_source='d_model'
        )
        print(f"    Probe accuracy at result_pos: {acc_dm*100:.1f}%")
        probes['d_model'] = probe_dm
        probe_accuracies[f'{group}_d_model'] = acc_dm

        # d_state probe (SSM groups only)
        if cfg['is_ssm']:
            d_state = 16
            print(f"\n  Training d_state probe (dim={d_state})...")
            probe_ds = AnswerProbe(d_state)
            probe_ds = train_probe(
                model, probe_ds, train_ds, train_parity, device,
                probe_source='d_state', epochs=args.probe_epochs,
                batch_size=args.batch_size, lr=args.probe_lr,
            )
            acc_ds = evaluate_probe_accuracy(
                model, probe_ds, test_ds, test_parity, device, probe_source='d_state'
            )
            print(f"    Probe accuracy at result_pos: {acc_ds*100:.1f}%")
            probes['d_state'] = probe_ds
            probe_accuracies[f'{group}_d_state'] = acc_ds

        # Compute correlations
        print(f"\n  Computing correlations on test set...")
        results = compute_correlations(
            model, probes, test_ds, test_parity, device,
            batch_size=args.batch_size,
        )
        all_results[group] = results

        # Print summary
        agg = results['aggregate']
        for source in probes.keys():
            key = f'{source}_mean_r'
            if key in agg:
                print(f"    {source}: mean r={agg[key]:.4f} "
                      f"(std={agg[f'{source}_std_r']:.4f}, "
                      f"neg={agg[f'{source}_fraction_negative']*100:.0f}%, "
                      f"sig={agg[f'{source}_fraction_significant']*100:.0f}%)")

        # Free model memory
        del model, probes
        torch.cuda.empty_cache() if device == 'cuda' else None

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating figures...")
    print(f"{'='*60}")

    plot_entropy_halt_trajectories(
        all_results,
        os.path.join(args.figures_dir, 'fig10_entropy_halt_trajectories.png'),
    )
    plot_entropy_halt_correlation(
        all_results,
        os.path.join(args.figures_dir, 'fig11_entropy_halt_correlation.png'),
    )

    # Save JSON results
    save_results(
        all_results, probe_accuracies,
        os.path.join(args.results_dir, 'entropy_halt_correlation.json'),
    )

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Entropy-Halt Correlation")
    print(f"{'='*60}")
    print(f"\n{'Group':<12s} {'Probe Acc':>10s} {'Mean r':>8s} {'% Neg':>7s} {'% |r|>0.3':>10s} {'Interpret':<20s}")
    print("-" * 72)

    for group in groups:
        if group not in all_results:
            continue
        agg = all_results[group]['aggregate']
        acc = probe_accuracies.get(f'{group}_d_model', 0)
        mean_r = agg.get('d_model_mean_r', float('nan'))
        frac_neg = agg.get('d_model_fraction_negative', 0)
        frac_sig = agg.get('d_model_fraction_significant', 0)
        interp = interpret_correlation(mean_r, frac_neg)
        print(f"{group:<12s} {acc*100:>9.1f}% {mean_r:>8.4f} {frac_neg*100:>6.0f}% {frac_sig*100:>9.0f}% {interp:<20s}")

    print(f"\nResults saved to {args.results_dir}/entropy_halt_correlation.json")
    print(f"Figures saved to {args.figures_dir}/fig10_*.png, fig11_*.png")


if __name__ == '__main__':
    main()
