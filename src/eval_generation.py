"""
Autoregressive generation evaluation for PNA-SSM experiment.
Runs free generation on all 4 groups and produces:
- Comprehensive metrics (accuracy, token efficiency, halt behavior)
- Statistical comparisons (t-tests, effect sizes)
- Publication-ready figures (fig7, fig8, fig9)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from generate import (
    FreeGenerator, GenerationConfig, GenerationResult,
    load_model, generate_test_inputs, compute_parity
)

sns.set_theme(style="whitegrid", font_scale=1.1)

GROUPS = ['A', 'B', 'C', 'D']
COLORS = {'A': '#95a5a6', 'B': '#3498db', 'C': '#2ecc71', 'D': '#e74c3c'}
LABELS = {'A': 'Trans+CE', 'B': 'Trans+L_th', 'C': 'SSM+CE', 'D': 'SSM+L_th'}


def evaluate_group(group, model, test_inputs, device):
    """Run free generation for one group on all test inputs."""
    use_halt = group in ('B', 'D')
    config = GenerationConfig(
        max_length=200,
        halt_confidence_threshold=0.95,
        use_halt_head=use_halt,
        temperature=0.0,
    )
    generator = FreeGenerator(model, config, device=device)

    results = []
    t0 = time.time()
    for i, bits in enumerate(test_inputs):
        result = generator.generate(bits)
        results.append(result)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{len(test_inputs)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"    Done: {len(results)} examples in {elapsed:.1f}s")
    return results


def compute_metrics(results, group_label=""):
    """Compute aggregate metrics from generation results."""
    n = len(results)
    if n == 0:
        return {}

    correct = [r.is_correct for r in results]
    valid = [r.is_valid_syntax for r in results]
    reasoning_tokens = [r.reasoning_token_count for r in results]
    total_tokens = [r.total_token_count for r in results]
    has_halt = [r.halt_position is not None for r in results]

    # Stop reason distribution
    stop_counts = defaultdict(int)
    for r in results:
        stop_counts[r.stop_reason] += 1

    # Halt position error (for examples that halted and have valid syntax)
    # "Optimal" halt = right after Result:X — measure offset from there
    halt_errors = []
    for r in results:
        if r.halt_position is not None and r.is_valid_syntax:
            # Ideal halt is at halt_position; error is 0 if it stopped right
            # after Result:X. We measure distance from the result token.
            # In ideal output: ...Result:X<HALT> — the halt is ~2 tokens after Result:
            halt_errors.append(0)  # Token-level; we track the trajectory instead

    metrics = {
        "group": group_label,
        "n_examples": n,
        "accuracy": float(np.mean(correct)),
        "valid_syntax_rate": float(np.mean(valid)),

        # Token efficiency
        "mean_reasoning_tokens": float(np.mean(reasoning_tokens)),
        "median_reasoning_tokens": float(np.median(reasoning_tokens)),
        "std_reasoning_tokens": float(np.std(reasoning_tokens)),
        "min_reasoning_tokens": int(np.min(reasoning_tokens)),
        "max_reasoning_tokens": int(np.max(reasoning_tokens)),

        "mean_total_tokens": float(np.mean(total_tokens)),

        # Halt behavior
        "halt_token_rate": float(np.mean(has_halt)),

        # Stop reasons
        "stop_reasons": dict(stop_counts),

        # Stratified by input length
        "by_input_length": {},
    }

    # Stratify by input length
    for length in range(2, 11):
        subset = [r for r in results if len(r.input_bits) == length]
        if subset:
            sub_correct = [r.is_correct for r in subset]
            sub_tokens = [r.reasoning_token_count for r in subset]
            sub_valid = [r.is_valid_syntax for r in subset]
            metrics["by_input_length"][str(length)] = {
                "n": len(subset),
                "accuracy": float(np.mean(sub_correct)),
                "mean_reasoning_tokens": float(np.mean(sub_tokens)),
                "std_reasoning_tokens": float(np.std(sub_tokens)),
                "valid_syntax_rate": float(np.mean(sub_valid)),
            }

    return metrics


def compare_groups(all_results):
    """Statistical comparison between groups."""
    comparisons = {}

    tokens = {}
    for group in GROUPS:
        if group in all_results:
            tokens[group] = [r.reasoning_token_count for r in all_results[group]]

    if 'D' not in tokens or 'A' not in tokens:
        return comparisons

    for other in ['A', 'B', 'C']:
        if other not in tokens:
            continue

        d_tok = np.array(tokens['D'])
        other_tok = np.array(tokens[other])

        # Welch's t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(d_tok, other_tok, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(d_tok)**2 + np.std(other_tok)**2) / 2)
        cohens_d = (np.mean(other_tok) - np.mean(d_tok)) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(d_tok, other_tok, alternative='two-sided')

        mean_reduction = (np.mean(other_tok) - np.mean(d_tok)) / np.mean(other_tok) if np.mean(other_tok) > 0 else 0

        comparisons[f"D_vs_{other}"] = {
            "D_mean": float(np.mean(d_tok)),
            "D_std": float(np.std(d_tok)),
            f"{other}_mean": float(np.mean(other_tok)),
            f"{other}_std": float(np.std(other_tok)),
            "mean_reduction_pct": float(mean_reduction * 100),
            "t_statistic": float(t_stat),
            "t_pvalue": float(p_val),
            "cohens_d": float(cohens_d),
            "mann_whitney_U": float(u_stat),
            "mann_whitney_p": float(u_pval),
        }

    # Also compare B vs A (thermodynamic effect on Transformer alone)
    if 'B' in tokens and 'A' in tokens:
        b_tok = np.array(tokens['B'])
        a_tok = np.array(tokens['A'])
        t_stat, p_val = stats.ttest_ind(b_tok, a_tok, equal_var=False)
        pooled_std = np.sqrt((np.std(b_tok)**2 + np.std(a_tok)**2) / 2)
        cohens_d = (np.mean(a_tok) - np.mean(b_tok)) / pooled_std if pooled_std > 0 else 0

        comparisons["B_vs_A"] = {
            "B_mean": float(np.mean(b_tok)),
            "A_mean": float(np.mean(a_tok)),
            "mean_reduction_pct": float((np.mean(a_tok) - np.mean(b_tok)) / np.mean(a_tok) * 100) if np.mean(a_tok) > 0 else 0,
            "t_statistic": float(t_stat),
            "t_pvalue": float(p_val),
            "cohens_d": float(cohens_d),
        }

    # Compare C vs A (SSM architecture effect alone)
    if 'C' in tokens and 'A' in tokens:
        c_tok = np.array(tokens['C'])
        a_tok = np.array(tokens['A'])
        t_stat, p_val = stats.ttest_ind(c_tok, a_tok, equal_var=False)
        pooled_std = np.sqrt((np.std(c_tok)**2 + np.std(a_tok)**2) / 2)
        cohens_d = (np.mean(a_tok) - np.mean(c_tok)) / pooled_std if pooled_std > 0 else 0

        comparisons["C_vs_A"] = {
            "C_mean": float(np.mean(c_tok)),
            "A_mean": float(np.mean(a_tok)),
            "mean_reduction_pct": float((np.mean(a_tok) - np.mean(c_tok)) / np.mean(a_tok) * 100) if np.mean(a_tok) > 0 else 0,
            "t_statistic": float(t_stat),
            "t_pvalue": float(p_val),
            "cohens_d": float(cohens_d),
        }

    return comparisons


# ============================================================
# Figures
# ============================================================

def plot_generation_comparison(all_metrics, all_results, save_dir):
    """fig7: Box plots of reasoning length by group."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Reasoning token count boxplots
    ax = axes[0]
    data = []
    group_labels = []
    for group in GROUPS:
        if group in all_results:
            tokens = [r.reasoning_token_count for r in all_results[group]]
            data.append(tokens)
            group_labels.append(f"{group}\n{LABELS[group]}")

    bp = ax.boxplot(data, tick_labels=group_labels, patch_artist=True, widths=0.6)
    for i, group in enumerate([g for g in GROUPS if g in all_results]):
        bp['boxes'][i].set_facecolor(COLORS[group])
        bp['boxes'][i].set_alpha(0.7)
    ax.set_ylabel('Reasoning Tokens', fontsize=12)
    ax.set_title('Reasoning Chain Length by Group', fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.3)

    # Add mean annotations
    for i, group in enumerate([g for g in GROUPS if g in all_results]):
        m = all_metrics[group]["mean_reasoning_tokens"]
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'{m:.1f}',
                ha='center', fontsize=10, fontweight='bold', color=COLORS[group])

    # Panel 2: Accuracy by group
    ax = axes[1]
    groups_present = [g for g in GROUPS if g in all_metrics]
    accs = [all_metrics[g]["accuracy"] * 100 for g in groups_present]
    bars = ax.bar(range(len(groups_present)), accs,
                  color=[COLORS[g] for g in groups_present], alpha=0.8)
    ax.set_xticks(range(len(groups_present)))
    ax.set_xticklabels([f"{g}\n{LABELS[g]}" for g in groups_present])
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Free Generation Accuracy', fontweight='bold', fontsize=13)
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.4, label='95% target')
    ax.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: Stop reason distribution
    ax = axes[2]
    reasons = ["halt_token", "eos", "halt_confidence", "max_length"]
    reason_labels = ["<HALT>", "<EOS>", "Halt Conf", "Max Len"]
    x = np.arange(len(groups_present))
    width = 0.2
    for i, (reason, rlabel) in enumerate(zip(reasons, reason_labels)):
        counts = []
        for g in groups_present:
            total = all_metrics[g]["n_examples"]
            count = all_metrics[g]["stop_reasons"].get(reason, 0)
            counts.append(count / total * 100)
        ax.bar(x + i * width, counts, width, label=rlabel, alpha=0.8)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"{g}\n{LABELS[g]}" for g in groups_present])
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Stop Reason Distribution', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig7_generation_comparison.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig7_generation_comparison.png")


def plot_halt_placement(all_results, save_dir):
    """fig8: Halt confidence trajectories and placement analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Halt Confidence During Free Generation', fontsize=16, fontweight='bold')

    for idx, group in enumerate(GROUPS):
        ax = axes[idx // 2][idx % 2]
        if group not in all_results:
            ax.text(0.5, 0.5, 'No data', ha='center', transform=ax.transAxes)
            ax.set_title(f'Group {group} ({LABELS[group]})')
            continue

        results = all_results[group]

        # Plot halt confidence trajectories for first 50 examples
        n_show = min(50, len(results))
        for i in range(n_show):
            traj = results[i].halt_confidence_trajectory
            if traj:
                ax.plot(traj, alpha=0.15, color=COLORS[group], linewidth=0.8)

        # Compute and plot mean trajectory
        max_len = max((len(r.halt_confidence_trajectory) for r in results if r.halt_confidence_trajectory), default=0)
        if max_len > 0:
            padded = np.zeros((len(results), max_len))
            counts = np.zeros(max_len)
            for r in results:
                t = r.halt_confidence_trajectory
                if t:
                    padded_row = np.array(t[:max_len])
                    padded[0, :len(padded_row)] = padded_row  # just for mean calc
            # Proper mean
            all_trajs = []
            for r in results:
                if r.halt_confidence_trajectory:
                    all_trajs.append(r.halt_confidence_trajectory)
            if all_trajs:
                # Pad to same length
                padded = np.zeros((len(all_trajs), max_len))
                for i, t in enumerate(all_trajs):
                    padded[i, :len(t)] = t
                mean_traj = padded.mean(axis=0)
                ax.plot(mean_traj, color='black', linewidth=2.5, label='Mean')

        ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='Threshold (0.95)')
        ax.set_title(f'Group {group} ({LABELS[group]})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Generation Step')
        ax.set_ylabel('Halt Confidence')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig8_halt_placement.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig8_halt_placement.png")


def plot_adaptive_reasoning(all_metrics, all_results, save_dir):
    """fig9: Token count vs problem difficulty (scatter)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Reasoning tokens vs input length (scatter with means)
    ax = axes[0]
    for group in GROUPS:
        if group not in all_results:
            continue
        lengths = [len(r.input_bits) for r in all_results[group]]
        tokens = [r.reasoning_token_count for r in all_results[group]]

        # Jitter for visibility
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(lengths))
        ax.scatter(np.array(lengths) + jitter, tokens,
                   color=COLORS[group], alpha=0.15, s=15, label=None)

        # Mean line
        by_len = defaultdict(list)
        for l, t in zip(lengths, tokens):
            by_len[l].append(t)
        xs = sorted(by_len.keys())
        means = [np.mean(by_len[x]) for x in xs]
        ax.plot(xs, means, 'o-', color=COLORS[group], linewidth=2.5,
                markersize=8, label=LABELS[group])

    ax.set_xlabel('Input Length (bits)', fontsize=12)
    ax.set_ylabel('Reasoning Tokens', fontsize=12)
    ax.set_title('Reasoning Length vs Problem Difficulty', fontweight='bold', fontsize=13)
    ax.axvspan(2, 8.5, alpha=0.08, color='green', label='In-dist')
    ax.axvspan(8.5, 10.5, alpha=0.08, color='red', label='OOD')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Token count variance comparison
    ax = axes[1]
    groups_present = [g for g in GROUPS if g in all_metrics]
    stds = [all_metrics[g]["std_reasoning_tokens"] for g in groups_present]
    bars = ax.bar(range(len(groups_present)), stds,
                  color=[COLORS[g] for g in groups_present], alpha=0.8)
    ax.set_xticks(range(len(groups_present)))
    ax.set_xticklabels([f"{g}\n{LABELS[g]}" for g in groups_present])
    ax.set_ylabel('Std Dev of Reasoning Tokens', fontsize=12)
    ax.set_title('Adaptive Reasoning: Token Count Variability', fontweight='bold', fontsize=13)
    for bar, std in zip(bars, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{std:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fig9_adaptive_reasoning.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig9_adaptive_reasoning.png")


def print_summary(all_metrics, comparisons):
    """Print a summary table to console."""
    print(f"\n{'='*80}")
    print("AUTOREGRESSIVE GENERATION RESULTS")
    print(f"{'='*80}")

    header = f"{'Group':<8} {'Accuracy':>10} {'Valid':>8} {'Mean Tok':>10} {'Std Tok':>10} {'Halt Rate':>10} {'Stop Mode':>15}"
    print(header)
    print("-" * len(header))

    for group in GROUPS:
        if group not in all_metrics:
            continue
        m = all_metrics[group]
        # Most common stop reason
        sr = m["stop_reasons"]
        top_stop = max(sr, key=sr.get) if sr else "?"
        top_pct = sr.get(top_stop, 0) / m["n_examples"] * 100

        print(f"{group} ({LABELS[group]:<10s}) {m['accuracy']*100:>7.1f}% {m['valid_syntax_rate']*100:>6.1f}% "
              f"{m['mean_reasoning_tokens']:>9.1f} {m['std_reasoning_tokens']:>9.1f} "
              f"{m['halt_token_rate']*100:>8.1f}% {top_stop:>10s}({top_pct:.0f}%)")

    if comparisons:
        print(f"\n{'='*80}")
        print("STATISTICAL COMPARISONS")
        print(f"{'='*80}")
        for key, comp in comparisons.items():
            print(f"\n  {key}:")
            if "mean_reduction_pct" in comp:
                print(f"    Mean token reduction: {comp['mean_reduction_pct']:.1f}%")
            if "t_pvalue" in comp:
                sig = "***" if comp["t_pvalue"] < 0.001 else "**" if comp["t_pvalue"] < 0.01 else "*" if comp["t_pvalue"] < 0.05 else "ns"
                print(f"    t-test: t={comp['t_statistic']:.3f}, p={comp['t_pvalue']:.4f} ({sig})")
            if "cohens_d" in comp:
                d = abs(comp["cohens_d"])
                size = "large" if d > 0.8 else "medium" if d > 0.5 else "small" if d > 0.2 else "negligible"
                print(f"    Cohen's d: {comp['cohens_d']:.3f} ({size})")
            if "mann_whitney_p" in comp:
                print(f"    Mann-Whitney U: U={comp['mann_whitney_U']:.0f}, p={comp['mann_whitney_p']:.4f}")


def save_example_traces(all_results, save_dir, n_per_group=5):
    """Save qualitative example traces for each group."""
    traces = {}
    for group in GROUPS:
        if group not in all_results:
            continue
        results = all_results[group]
        # Pick diverse examples: 2 short, 2 medium, 1 long
        by_len = defaultdict(list)
        for r in results:
            by_len[len(r.input_bits)].append(r)

        selected = []
        for n_bits in [2, 4, 6, 8, 10]:
            if n_bits in by_len and by_len[n_bits]:
                selected.append(by_len[n_bits][0])
            if len(selected) >= n_per_group:
                break

        # Fill remaining from any length
        if len(selected) < n_per_group:
            for r in results:
                if r not in selected:
                    selected.append(r)
                if len(selected) >= n_per_group:
                    break

        traces[group] = []
        for r in selected:
            traces[group].append({
                "input": r.input_bits,
                "ground_truth": r.ground_truth,
                "generated": r.generated_text,
                "parsed_answer": r.parsed_answer,
                "correct": r.is_correct,
                "valid_syntax": r.is_valid_syntax,
                "reasoning_tokens": r.reasoning_token_count,
                "total_tokens": r.total_token_count,
                "stop_reason": r.stop_reason,
            })

    path = os.path.join(save_dir, "generation_traces.json")
    with open(path, 'w') as f:
        json.dump(traces, f, indent=2)
    print(f"  Saved generation_traces.json ({sum(len(v) for v in traces.values())} examples)")


def main():
    parser = argparse.ArgumentParser(description="Autoregressive Generation Evaluation")
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--figures-dir', default='figures')
    parser.add_argument('--groups', default='A,B,C,D')
    parser.add_argument('--n-in-dist', type=int, default=400)
    parser.add_argument('--n-ood', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.figures_dir, exist_ok=True)
    groups = [g.strip() for g in args.groups.split(',')]
    device = args.device
    print(f"Device: {device}")

    # Generate test inputs (same for all groups)
    print(f"\nGenerating test set: {args.n_in_dist} in-dist + {args.n_ood} OOD")
    test_inputs = generate_test_inputs(args.n_in_dist, args.n_ood)
    print(f"  Total: {len(test_inputs)} examples")

    # Run generation for each group
    all_results = {}
    all_metrics = {}

    for group in groups:
        checkpoint = os.path.join(args.results_dir, f"group_{group}_model.pt")
        if not os.path.exists(checkpoint):
            print(f"\n  Skipping Group {group}: no checkpoint at {checkpoint}")
            continue

        print(f"\n{'='*60}")
        print(f"Group {group} ({LABELS[group]})")
        print(f"{'='*60}")

        model = load_model(group, checkpoint, device)
        results = evaluate_group(group, model, test_inputs, device)
        all_results[group] = results
        all_metrics[group] = compute_metrics(results, group)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Statistical comparisons
    print("\nRunning statistical comparisons...")
    comparisons = compare_groups(all_results)

    # Print summary
    print_summary(all_metrics, comparisons)

    # Save results
    save_path = os.path.join(args.results_dir, "generation_results.json")
    with open(save_path, 'w') as f:
        json.dump({
            "metrics": all_metrics,
            "comparisons": comparisons,
        }, f, indent=2, default=str)
    print(f"\n  Saved metrics to {save_path}")

    # Save example traces
    save_example_traces(all_results, args.results_dir)

    # Generate figures
    print("\nGenerating figures...")
    plot_generation_comparison(all_metrics, all_results, args.figures_dir)
    plot_halt_placement(all_results, args.figures_dir)
    plot_adaptive_reasoning(all_metrics, all_results, args.figures_dir)

    print(f"\nAll autoregressive evaluation complete!")
    print(f"  Results: {args.results_dir}/generation_results.json")
    print(f"  Traces:  {args.results_dir}/generation_traces.json")
    print(f"  Figures: {args.figures_dir}/fig7_*.png, fig8_*.png, fig9_*.png")


if __name__ == '__main__':
    main()
