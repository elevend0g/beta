"""
PNA-SSM Experiment: Visualization & Analysis
============================================

Generates the key figures for the paper:
1. Four-way entropy collapse comparison (Groups A, B, C, D)
2. SSM-specific: Token vs State entropy (Groups C & D only)
3. Training dynamics under pressure (Groups B & D)
4. Length generalization curves (all groups)
5. Statistical comparison tables

Run this after training all four groups.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def plot_four_way_entropy_comparison(results_dict, save_path="entropy_comparison.png"):
    """
    The money shot: Side-by-side entropy collapse for all four groups.
    
    Expected pattern:
    - Group A: Slow, gradual decline (baseline)
    - Group B: Step-function, but noisy
    - Group C: Smoother than A, but not step-function
    - Group D: Sharpest step-function (hypothesis confirmation)
    
    Args:
        results_dict: {
            'A': {'trajectories': [...], 'mean': [...], 'std': [...]},
            'B': ..., 'C': ..., 'D': ...
        }
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entropy Collapse Dynamics Across Architectures and Loss Functions', 
                 fontsize=16, fontweight='bold')
    
    groups = [
        ('A', 'Transformer + CE (Baseline)', axes[0, 0]),
        ('B', 'Transformer + L_th (PNA v1)', axes[0, 1]),
        ('C', 'SSM + CE (Architecture)', axes[1, 0]),
        ('D', 'SSM + L_th (PNA-SSM)', axes[1, 1])
    ]
    
    for group_id, title, ax in groups:
        data = results_dict[group_id]
        
        # Plot individual trajectories (thin, transparent)
        for traj in data['trajectories'][:20]:  # Sample 20 examples
            ax.plot(traj, alpha=0.1, color='gray', linewidth=0.5)
        
        # Plot mean trajectory (thick)
        mean = data['mean']
        std = data['std']
        positions = range(len(mean))
        
        color = {
            'A': '#95a5a6',  # Gray
            'B': '#3498db',  # Blue
            'C': '#2ecc71',  # Green
            'D': '#e74c3c'   # Red (highlight)
        }[group_id]
        
        ax.plot(positions, mean, color=color, linewidth=3, label='Mean')
        ax.fill_between(positions, 
                        np.array(mean) - np.array(std),
                        np.array(mean) + np.array(std),
                        alpha=0.2, color=color)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Entropy (bits)')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 6)
        
        # Add step-function indicator if present
        if _detect_step_function(mean):
            ax.text(0.05, 0.95, 'Step-function detected ✓', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_dual_entropy_ssm(ssm_results, group_label, save_path="dual_entropy.png"):
    """
    SSM-specific: Token entropy vs State entropy on the same plot.
    
    Key insight to look for: Does state entropy lead token entropy?
    I.e., does h_t "know" the answer before the model generates it?
    
    Args:
        ssm_results: {
            'token_entropy': [...],
            'state_entropy': [...],
            'positions': [...]
        }
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    positions = ssm_results['positions']
    token_entropy = ssm_results['token_entropy']
    state_entropy = ssm_results['state_entropy']
    
    # Token entropy (primary y-axis)
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('Token Entropy H(next_token)', color='tab:blue', fontsize=12)
    ax1.plot(positions, token_entropy, color='tab:blue', linewidth=2, 
            label='Token Entropy', marker='o', markersize=4)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(alpha=0.3)
    
    # State entropy (secondary y-axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel('State Entropy H(h_t)', color='tab:red', fontsize=12)
    ax2.plot(positions, state_entropy, color='tab:red', linewidth=2, 
            label='State Entropy', marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Highlight where state entropy drops before token entropy
    state_drops = np.where(np.diff(state_entropy) < -0.5)[0]
    token_drops = np.where(np.diff(token_entropy) < -0.5)[0]
    
    if len(state_drops) > 0 and len(token_drops) > 0:
        if state_drops[0] < token_drops[0]:
            ax1.axvline(state_drops[0], color='red', linestyle='--', alpha=0.5, 
                       label='State collapses first')
    
    plt.title(f'{group_label}: Token vs State Entropy Dynamics', 
             fontsize=14, fontweight='bold')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_training_stability_under_pressure(results_b, results_d, save_path="stability.png"):
    """
    Compare Groups B (Transformer+L_th) vs D (SSM+L_th) training dynamics
    as α (thermodynamic pressure) increases.
    
    Hypothesis: Group D remains stable at higher α where Group B becomes unstable.
    
    Args:
        results_b: {alpha_value: {'loss_variance': ..., 'grad_norm_variance': ...}}
        results_d: Same structure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    alphas = sorted(results_b.keys())
    
    # Loss variance
    b_loss_var = [results_b[a]['loss_variance'] for a in alphas]
    d_loss_var = [results_d[a]['loss_variance'] for a in alphas]
    
    ax1.plot(alphas, b_loss_var, 'o-', color='#3498db', linewidth=2, 
            markersize=8, label='Transformer + L_th')
    ax1.plot(alphas, d_loss_var, 's-', color='#e74c3c', linewidth=2, 
            markersize=8, label='SSM + L_th')
    ax1.set_xlabel('Thermodynamic Pressure (α)', fontsize=12)
    ax1.set_ylabel('Training Loss Variance', fontsize=12)
    ax1.set_title('Training Stability vs Thermodynamic Pressure', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Gradient norm variance
    b_grad_var = [results_b[a]['grad_norm_variance'] for a in alphas]
    d_grad_var = [results_d[a]['grad_norm_variance'] for a in alphas]
    
    ax2.plot(alphas, b_grad_var, 'o-', color='#3498db', linewidth=2, 
            markersize=8, label='Transformer + L_th')
    ax2.plot(alphas, d_grad_var, 's-', color='#e74c3c', linewidth=2, 
            markersize=8, label='SSM + L_th')
    ax2.set_xlabel('Thermodynamic Pressure (α)', fontsize=12)
    ax2.set_ylabel('Gradient Norm Variance', fontsize=12)
    ax2.set_title('Gradient Stability vs Thermodynamic Pressure', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    # Highlight instability threshold
    instability_alpha = None
    for a in alphas:
        if results_b[a]['loss_variance'] > 0.5:  # Arbitrary threshold
            instability_alpha = a
            break
    
    if instability_alpha:
        ax1.axvline(instability_alpha, color='#3498db', linestyle='--', alpha=0.5,
                   label=f'Transformer unstable >{instability_alpha:.2f}')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_length_generalization(results_dict, save_path="generalization.png"):
    """
    Test all four groups on increasingly long sequences (out of distribution).
    
    X-axis: Sequence length (bits in parity task)
    Y-axis: Accuracy
    
    Hypothesis: Group D maintains accuracy longest.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'A': '#95a5a6',
        'B': '#3498db',
        'C': '#2ecc71',
        'D': '#e74c3c'
    }
    
    labels = {
        'A': 'Transformer + CE',
        'B': 'Transformer + L_th',
        'C': 'SSM + CE',
        'D': 'SSM + L_th'
    }
    
    for group in ['A', 'B', 'C', 'D']:
        lengths = results_dict[group]['sequence_lengths']
        accuracies = results_dict[group]['accuracies']
        
        ax.plot(lengths, accuracies, 'o-', color=colors[group], 
               linewidth=2, markersize=8, label=labels[group])
    
    # Shade the training region
    ax.axvspan(2, 5, alpha=0.1, color='green', label='Training distribution (2-5 bits)')
    
    ax.set_xlabel('Sequence Length (bits)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Length Generalization: Out-of-Distribution Performance', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")


def generate_statistical_comparison_table(results_dict, save_path="comparison_table.csv"):
    """
    Generate the main results table for the paper.
    
    Columns: Group | Architecture | Loss | Accuracy | Tokens | Halt F1 | State ΔH
    
    Includes significance tests (Mann-Whitney U) for key comparisons.
    """
    rows = []
    
    for group_id in ['A', 'B', 'C', 'D']:
        data = results_dict[group_id]
        
        arch = 'Transformer' if group_id in ['A', 'B'] else 'SSM'
        loss = 'CE' if group_id in ['A', 'C'] else 'L_th'
        
        row = {
            'Group': group_id,
            'Architecture': arch,
            'Loss Function': loss,
            'Accuracy (%)': f"{data['accuracy_mean']*100:.2f} ± {data['accuracy_std']*100:.2f}",
            'Reasoning Tokens': f"{data['token_mean']:.1f} ± {data['token_std']:.1f}",
            'Halt F1': f"{data['halt_f1']:.3f}" if 'halt_f1' in data else 'N/A',
            'Token ΔH (bits/step)': f"{data['token_delta_h']:.3f}",
            'State ΔH (bits/step)': f"{data['state_delta_h']:.3f}" if 'state_delta_h' in data else 'N/A'
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Add significance annotations
    # Test 1: Group B vs A (thermodynamic effect in Transformer)
    p_b_vs_a = _mann_whitney_u(results_dict['B']['token_counts'], 
                                results_dict['A']['token_counts'])
    
    # Test 2: Group D vs C (thermodynamic effect in SSM)
    p_d_vs_c = _mann_whitney_u(results_dict['D']['token_counts'], 
                                results_dict['C']['token_counts'])
    
    # Test 3: Group D vs B (SSM advantage with thermodynamic loss)
    p_d_vs_b = _mann_whitney_u(results_dict['D']['token_counts'], 
                                results_dict['B']['token_counts'])
    
    # Test 4: Interaction effect (synergy)
    # H0: D improvement = B improvement + C improvement - A baseline
    a_mean = results_dict['A']['token_mean']
    b_mean = results_dict['B']['token_mean']
    c_mean = results_dict['C']['token_mean']
    d_mean = results_dict['D']['token_mean']
    
    b_improvement = (a_mean - b_mean) / a_mean
    c_improvement = (a_mean - c_mean) / a_mean
    expected_d = a_mean * (1 - b_improvement - c_improvement)
    synergy = (expected_d - d_mean) / a_mean
    
    # Append significance notes
    notes = [
        "",
        f"Significance tests (Mann-Whitney U, p < 0.05):",
        f"  B vs A: p = {p_b_vs_a:.4f} {'*' if p_b_vs_a < 0.05 else '(ns)'}",
        f"  D vs C: p = {p_d_vs_c:.4f} {'*' if p_d_vs_c < 0.05 else '(ns)'}",
        f"  D vs B: p = {p_d_vs_b:.4f} {'*' if p_d_vs_b < 0.05 else '(ns)'}",
        "",
        f"Synergy analysis (Group D):",
        f"  Expected additive: {expected_d:.1f} tokens",
        f"  Actual: {d_mean:.1f} tokens",
        f"  Synergy: {synergy*100:.1f}% {'(positive)' if synergy > 0 else '(none)'}"
    ]
    
    # Save
    df.to_csv(save_path, index=False)
    with open(save_path.replace('.csv', '_notes.txt'), 'w') as f:
        f.write('\n'.join(notes))
    
    print(f"Saved: {save_path}")
    print("\nSummary:")
    print(df.to_string(index=False))
    print('\n'.join(notes))


def _detect_step_function(entropy_trajectory, threshold=0.5):
    """
    Heuristic: A step-function has large drops (>threshold) separated by flat regions.
    Returns True if pattern detected.
    """
    diffs = np.diff(entropy_trajectory)
    large_drops = np.where(diffs < -threshold)[0]
    
    if len(large_drops) < 2:
        return False
    
    # Check if drops are separated by flat regions
    for i in range(len(large_drops) - 1):
        flat_region = entropy_trajectory[large_drops[i]+1:large_drops[i+1]]
        if len(flat_region) > 2 and np.std(flat_region) < 0.1:
            return True
    
    return False


def _mann_whitney_u(sample_a, sample_b):
    """Wrapper for Mann-Whitney U test."""
    statistic, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative='less')
    return p_value


def generate_full_paper_figures(results_dict, output_dir="figures/"):
    """
    Generate all figures needed for the paper in one call.
    
    Args:
        results_dict: Nested dict with all experimental data
            {
                'A': {'trajectories': ..., 'mean': ..., 'token_counts': ..., ...},
                'B': ...,
                'C': {..., 'state_entropy_trajectories': ...},
                'D': {..., 'state_entropy_trajectories': ...},
                'training_stability': {'B': {...}, 'D': {...}},
                'generalization': {'A': {...}, 'B': ..., 'C': ..., 'D': {...}}
            }
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating paper figures...")
    
    # Figure 1: Four-way entropy comparison
    plot_four_way_entropy_comparison(
        {k: results_dict[k] for k in ['A', 'B', 'C', 'D']},
        save_path=os.path.join(output_dir, "fig1_entropy_comparison.png")
    )
    
    # Figure 2: Dual entropy (SSM groups only)
    for group in ['C', 'D']:
        plot_dual_entropy_ssm(
            results_dict[group]['dual_entropy'],
            group_label=f"Group {group}",
            save_path=os.path.join(output_dir, f"fig2_{group}_dual_entropy.png")
        )
    
    # Figure 3: Training stability
    plot_training_stability_under_pressure(
        results_dict['training_stability']['B'],
        results_dict['training_stability']['D'],
        save_path=os.path.join(output_dir, "fig3_training_stability.png")
    )
    
    # Figure 4: Length generalization
    plot_length_generalization(
        results_dict['generalization'],
        save_path=os.path.join(output_dir, "fig4_generalization.png")
    )
    
    # Table 1: Statistical comparison
    generate_statistical_comparison_table(
        {k: results_dict[k] for k in ['A', 'B', 'C', 'D']},
        save_path=os.path.join(output_dir, "table1_comparison.csv")
    )
    
    print(f"\nAll figures saved to {output_dir}")
    print("\nPaper-ready figures:")
    print("  - fig1_entropy_comparison.png (Main result)")
    print("  - fig2_C_dual_entropy.png (SSM baseline)")
    print("  - fig2_D_dual_entropy.png (PNA-SSM, shows state leading)")
    print("  - fig3_training_stability.png (Stability advantage)")
    print("  - fig4_generalization.png (Length generalization)")
    print("  - table1_comparison.csv (Quantitative results)")


if __name__ == "__main__":
    # This would be called after training all four groups
    # Example usage:
    
    # results = load_experimental_results("experiments/pna_ssm/")
    # generate_full_paper_figures(results)
    
    print("Visualization module loaded.")
    print("Run generate_full_paper_figures(results_dict) after training.")
