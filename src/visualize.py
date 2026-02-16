"""
PNA-SSM Experiment: Generate publication-ready figures from results.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_results(results_dir):
    """Load all results from JSON files."""
    results = {}
    for group in ['A', 'B', 'C', 'D']:
        path = os.path.join(results_dir, f"group_{group}_results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[group] = json.load(f)

    for fname in ['stability_results.json', 'generalization_results.json']:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                key = fname.replace('_results.json', '')
                results[key] = json.load(f)

    return results


def plot_training_curves(results, save_dir):
    """Plot training loss and accuracy curves for all groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'A': '#95a5a6', 'B': '#3498db', 'C': '#2ecc71', 'D': '#e74c3c'}
    labels = {
        'A': 'Trans+CE', 'B': 'Trans+L_th',
        'C': 'SSM+CE', 'D': 'SSM+L_th'
    }

    for group in ['A', 'B', 'C', 'D']:
        if group not in results:
            continue
        history = results[group].get('history', [])
        epochs = [h['epoch'] for h in history]
        train_loss = [h['train']['total_loss'] for h in history]
        val_acc = [h['val']['accuracy'] for h in history]

        ax1.plot(epochs, train_loss, color=colors[group], linewidth=2, label=labels[group])
        ax2.plot(epochs, val_acc, color=colors[group], linewidth=2, label=labels[group])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves', fontweight='bold')
    ax1.legend()

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy', fontweight='bold')
    ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.3, label='95% target')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig0_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig0_training_curves.png")


def plot_entropy_comparison(results, save_dir):
    """Four-way entropy collapse comparison (main result figure)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entropy Collapse Dynamics Across Architectures and Loss Functions',
                 fontsize=16, fontweight='bold')

    groups = [
        ('A', 'Transformer + CE (Baseline)', axes[0, 0]),
        ('B', 'Transformer + L_th (PNA v1)', axes[0, 1]),
        ('C', 'SSM + CE (Architecture)', axes[1, 0]),
        ('D', 'SSM + L_th (PNA-SSM)', axes[1, 1])
    ]

    colors = {'A': '#95a5a6', 'B': '#3498db', 'C': '#2ecc71', 'D': '#e74c3c'}

    for group_id, title, ax in groups:
        if group_id not in results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        entropy_data = results[group_id].get('entropy_data', {})
        trajectories = entropy_data.get('trajectories', [])
        mean = entropy_data.get('mean', [])
        std_vals = entropy_data.get('std', [])

        if not mean:
            ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Plot sample trajectories
        for traj in trajectories[:20]:
            ax.plot(traj, alpha=0.1, color='gray', linewidth=0.5)

        positions = range(len(mean))
        ax.plot(positions, mean, color=colors[group_id], linewidth=3, label='Mean')
        ax.fill_between(positions,
                        np.array(mean) - np.array(std_vals),
                        np.array(mean) + np.array(std_vals),
                        alpha=0.2, color=colors[group_id])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Entropy (bits)')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig1_entropy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig1_entropy_comparison.png")


def plot_dual_entropy(results, save_dir):
    """Token vs State entropy for SSM groups."""
    for group_id in ['C', 'D']:
        if group_id not in results:
            continue

        entropy_data = results[group_id].get('entropy_data', {})
        token_mean = entropy_data.get('mean', [])
        state_mean = entropy_data.get('state_mean', [])

        if not token_mean or not state_mean:
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))

        positions = range(len(token_mean))

        ax1.set_xlabel('Token Position', fontsize=12)
        ax1.set_ylabel('Token Entropy H(next_token)', color='tab:blue', fontsize=12)
        ax1.plot(positions, token_mean, color='tab:blue', linewidth=2, label='Token Entropy')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        ax2.set_ylabel('State Entropy H(h_t)', color='tab:red', fontsize=12)
        ax2.plot(positions, state_mean, color='tab:red', linewidth=2, label='State Entropy')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        label = {'C': 'SSM + CE', 'D': 'SSM + L_th (PNA-SSM)'}[group_id]
        plt.title(f'Group {group_id} ({label}): Token vs State Entropy',
                  fontsize=14, fontweight='bold')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'fig2_{group_id}_dual_entropy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved fig2_{group_id}_dual_entropy.png")


def plot_stability(results, save_dir):
    """Training stability under thermodynamic pressure."""
    stability = results.get('stability', {})
    if not stability:
        print("  No stability data, skipping fig3")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for group, color, marker in [('B', '#3498db', 'o'), ('D', '#e74c3c', 's')]:
        if group not in stability:
            continue
        data = stability[group]
        alphas = sorted([float(a) for a in data.keys()])
        loss_vars = [data[str(a)]['loss_variance'] for a in alphas]
        grad_vars = [data[str(a)]['grad_norm_variance'] for a in alphas]

        label = {'B': 'Transformer + L_th', 'D': 'SSM + L_th'}[group]
        ax1.plot(alphas, loss_vars, f'{marker}-', color=color, linewidth=2, markersize=8, label=label)
        ax2.plot(alphas, grad_vars, f'{marker}-', color=color, linewidth=2, markersize=8, label=label)

    ax1.set_xlabel('Thermodynamic Pressure (alpha)')
    ax1.set_ylabel('Training Loss Variance')
    ax1.set_title('Training Stability vs Pressure', fontweight='bold')
    ax1.legend()
    ax1.set_yscale('log')

    ax2.set_xlabel('Thermodynamic Pressure (alpha)')
    ax2.set_ylabel('Gradient Norm Variance')
    ax2.set_title('Gradient Stability vs Pressure', fontweight='bold')
    ax2.legend()
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig3_training_stability.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig3_training_stability.png")


def plot_generalization(results, save_dir):
    """Length generalization curves."""
    gen = results.get('generalization', {})
    if not gen:
        print("  No generalization data, skipping fig4")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'A': '#95a5a6', 'B': '#3498db', 'C': '#2ecc71', 'D': '#e74c3c'}
    labels = {'A': 'Trans+CE', 'B': 'Trans+L_th', 'C': 'SSM+CE', 'D': 'SSM+L_th'}

    for group in ['A', 'B', 'C', 'D']:
        if group not in gen:
            continue
        lengths = gen[group]['sequence_lengths']
        accs = gen[group]['accuracies']
        ax.plot(lengths, accs, 'o-', color=colors[group], linewidth=2, markersize=8, label=labels[group])

    ax.axvspan(2, 5, alpha=0.1, color='green', label='Training dist (2-5 bits)')
    ax.set_xlabel('Sequence Length (bits)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Length Generalization: Out-of-Distribution Performance',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig4_generalization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig4_generalization.png")


def generate_comparison_table(results, save_dir):
    """Generate the main results table."""
    rows = []
    for group in ['A', 'B', 'C', 'D']:
        if group not in results:
            continue
        r = results[group]
        rows.append({
            'Group': group,
            'Architecture': r.get('architecture', '?'),
            'Loss': r.get('loss_type', '?'),
            'Params': f"{r.get('n_params', 0):,}",
            'Accuracy': f"{r.get('test_accuracy', 0)*100:.1f}%",
            'Reasoning Tokens': f"{r.get('token_mean', 0):.1f} +/- {r.get('token_std', 0):.1f}",
            'Halt F1': f"{r.get('halt_f1', 0)*100:.1f}%",
        })

    if not rows:
        print("  No results data for table")
        return

    # Write as formatted text
    with open(os.path.join(save_dir, 'table1_results.txt'), 'w') as f:
        headers = list(rows[0].keys())
        widths = [max(len(h), max(len(str(r.get(h, ''))) for r in rows)) for h in headers]
        header_line = ' | '.join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = '-+-'.join('-' * w for w in widths)

        f.write(header_line + '\n')
        f.write(sep_line + '\n')
        for row in rows:
            f.write(' | '.join(str(row[h]).ljust(w) for h, w in zip(headers, widths)) + '\n')

        # Statistical tests
        if 'A' in results and 'D' in results:
            f.write('\n--- Statistical Summary ---\n')
            a_tok = results['A'].get('token_mean', 0)
            b_tok = results.get('B', {}).get('token_mean', a_tok)
            c_tok = results.get('C', {}).get('token_mean', a_tok)
            d_tok = results.get('D', {}).get('token_mean', a_tok)

            if a_tok > 0:
                b_imp = (a_tok - b_tok) / a_tok * 100
                c_imp = (a_tok - c_tok) / a_tok * 100
                d_imp = (a_tok - d_tok) / a_tok * 100
                additive = b_imp + c_imp

                f.write(f'Token reduction B vs A: {b_imp:.1f}%\n')
                f.write(f'Token reduction C vs A: {c_imp:.1f}%\n')
                f.write(f'Token reduction D vs A: {d_imp:.1f}%\n')
                f.write(f'Expected additive: {additive:.1f}%\n')
                f.write(f'Synergy: {d_imp - additive:.1f}%\n')

    print("  Saved table1_results.txt")


def generate_all_figures(results_dir, figures_dir):
    """Generate all paper figures."""
    os.makedirs(figures_dir, exist_ok=True)
    results = load_results(results_dir)

    print(f"\nGenerating figures from {results_dir} -> {figures_dir}")
    print(f"  Groups found: {[g for g in ['A','B','C','D'] if g in results]}")

    plot_training_curves(results, figures_dir)
    plot_entropy_comparison(results, figures_dir)
    plot_dual_entropy(results, figures_dir)
    plot_stability(results, figures_dir)
    plot_generalization(results, figures_dir)
    generate_comparison_table(results, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--figures-dir', default='figures')
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.figures_dir)
