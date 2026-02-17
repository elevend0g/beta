"""
Ablation: Halt-only training (L_ce + β·L_halt, no energy or state penalty).
Tests whether 99% halt F1 comes from supervised halt training alone,
or whether thermodynamic context (α·L_energy, γ·L_state) contributes.

Groups:
  E_trans: Transformer + L_ce + β·L_halt  (compare to B: Trans + full L_th)
  E_ssm:   SSM + L_ce + β·L_halt          (compare to D: SSM + full L_th)
"""

import os
import sys
import json
import time
import copy
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE, VOCAB
from models import create_model, count_parameters
from losses import ThermodynamicLoss
from train import (
    get_device, get_cosine_schedule, train_one_epoch,
    evaluate, compute_halt_f1, compute_reasoning_length,
    compute_entropy_trajectories
)
from generate import (
    FreeGenerator, GenerationConfig, load_model,
    generate_test_inputs, compute_parity
)


def train_ablation_group(name, arch_group, train_ds, val_ds, test_ds,
                         device, results_dir, epochs=50, batch_size=32, lr=3e-4):
    """
    Train a halt-only ablation group.
    arch_group: 'A' or 'C' (determines architecture: Transformer or SSM)
    """
    print(f"\n{'='*60}")
    print(f"Ablation Group {name} (arch={arch_group}, loss=CE+halt)")
    print(f"{'='*60}")

    model = create_model(arch_group, VOCAB_SIZE, device=device)
    n_params = count_parameters(model)
    arch = "Transformer" if arch_group in ('A', 'B') else "SSM"
    print(f"  Architecture: {arch}, Params: {n_params:,}")
    print(f"  Loss: L_ce + 0.1*L_halt (alpha=0, gamma=0)")

    # Halt-only loss: alpha=0 (no energy), gamma=0 (no state), beta=0.1 (halt)
    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0, pad_token_id=VOCAB['<PAD>'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train (no governor for halt-only)
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, None, optimizer, scheduler, device, arch_group
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
        })

        val_loss = val_metrics['total_loss']
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train_loss={train_metrics['total_loss']:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f} "
                  f"({elapsed:.1f}s){improved}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Evaluate
    print(f"\n  Final evaluation:")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    token_mean, token_std = compute_reasoning_length(model, test_loader, device)
    halt_f1 = compute_halt_f1(model, test_loader, device)
    entropy_data = compute_entropy_trajectories(model, test_loader, device)

    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Halt F1:  {halt_f1:.4f}")
    print(f"    Reasoning tokens: {token_mean:.1f} ± {token_std:.1f}")

    # Save
    results = {
        'group': name,
        'architecture': arch,
        'loss_type': 'CE+halt (ablation)',
        'n_params': n_params,
        'test_accuracy': test_metrics['accuracy'],
        'test_loss': test_metrics['total_loss'],
        'token_mean': token_mean,
        'token_std': token_std,
        'halt_f1': halt_f1,
        'test_metrics': test_metrics,
        'entropy_data': entropy_data,
        'history': history,
    }

    checkpoint_path = os.path.join(results_dir, f"group_{name}_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    with open(os.path.join(results_dir, f"group_{name}_results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return model, results


def run_ablation_generation(name, model, arch_group, test_inputs, device):
    """Run autoregressive generation for an ablation group."""
    # Halt-only groups DO have halt training, so use halt head
    config = GenerationConfig(
        max_length=200,
        halt_confidence_threshold=0.95,
        use_halt_head=True,
        temperature=0.0,
    )
    gen = FreeGenerator(model, config, device=device)

    correct = 0
    valid = 0
    total = len(test_inputs)
    stop_counts = defaultdict(int)
    reasoning_tokens = []

    for bits in test_inputs:
        result = gen.generate(bits)
        if result.is_correct:
            correct += 1
        if result.is_valid_syntax:
            valid += 1
        stop_counts[result.stop_reason] += 1
        reasoning_tokens.append(result.reasoning_token_count)

    return {
        'accuracy': correct / total,
        'valid_syntax_rate': valid / total,
        'mean_reasoning_tokens': float(np.mean(reasoning_tokens)),
        'std_reasoning_tokens': float(np.std(reasoning_tokens)),
        'stop_reasons': dict(stop_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Halt-only ablation")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.results_dir, exist_ok=True)
    print(f"Device: {device}")

    # Datasets
    print("Creating datasets...")
    train_ds, val_ds, test_ds = create_datasets(train_n=8000, val_n=1000, test_n=1000, max_seq_len=64)

    # Train ablation groups
    models = {}
    results = {}

    for name, arch_group in [('E_trans', 'A'), ('E_ssm', 'C')]:
        model, res = train_ablation_group(
            name, arch_group, train_ds, val_ds, test_ds,
            device, args.results_dir, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
        )
        models[name] = model
        results[name] = res

    # Autoregressive generation
    print(f"\n{'='*60}")
    print("Autoregressive Generation (Ablation)")
    print(f"{'='*60}")

    test_inputs = generate_test_inputs(400, 100)
    gen_results = {}
    for name, arch_group in [('E_trans', 'A'), ('E_ssm', 'C')]:
        print(f"\n  {name}:")
        gen_results[name] = run_ablation_generation(
            name, models[name], arch_group, test_inputs, device
        )
        gr = gen_results[name]
        print(f"    Free Gen Accuracy: {gr['accuracy']*100:.1f}%")
        print(f"    Valid Syntax:      {gr['valid_syntax_rate']*100:.1f}%")
        print(f"    Mean Tokens:       {gr['mean_reasoning_tokens']:.1f}")
        print(f"    Stop reasons:      {gr['stop_reasons']}")

    # Comparison table
    print(f"\n{'='*60}")
    print("ABLATION COMPARISON: Does thermodynamic context improve halt F1?")
    print(f"{'='*60}")

    # Load B and D results for comparison
    comparisons = [
        ('B (Trans+full L_th)', 'group_B_results.json'),
        ('E_trans (Trans+halt only)', f'group_E_trans_results.json'),
        ('D (SSM+full L_th)', 'group_D_results.json'),
        ('E_ssm (SSM+halt only)', f'group_E_ssm_results.json'),
    ]

    print(f"\n{'Group':<30s} {'TF Acc':>8s} {'Halt F1':>8s} {'Free Acc':>9s}")
    print("-" * 60)

    for label, fname in comparisons:
        path = os.path.join(args.results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            tf_acc = f"{r['test_accuracy']*100:.1f}%"
            halt_f1 = f"{r['halt_f1']*100:.1f}%"

            # Get free generation accuracy
            gen_name = label.split('(')[0].strip()
            if gen_name in gen_results:
                free_acc = f"{gen_results[gen_name]['accuracy']*100:.1f}%"
            elif 'B' in label:
                free_acc = "70.2%"  # From previous run
            elif 'D' in label:
                free_acc = "71.2%"  # From previous run
            else:
                free_acc = "N/A"

            print(f"{label:<30s} {tf_acc:>8s} {halt_f1:>8s} {free_acc:>9s}")

    # Verdict
    e_trans_f1 = results['E_trans']['halt_f1']
    e_ssm_f1 = results['E_ssm']['halt_f1']

    b_path = os.path.join(args.results_dir, 'group_B_results.json')
    d_path = os.path.join(args.results_dir, 'group_D_results.json')
    b_f1 = json.load(open(b_path))['halt_f1'] if os.path.exists(b_path) else 0
    d_f1 = json.load(open(d_path))['halt_f1'] if os.path.exists(d_path) else 0

    print(f"\n  Transformer: B halt F1 = {b_f1*100:.1f}%  vs  E_trans halt F1 = {e_trans_f1*100:.1f}%")
    print(f"  SSM:         D halt F1 = {d_f1*100:.1f}%  vs  E_ssm halt F1 = {e_ssm_f1*100:.1f}%")

    trans_diff = abs(b_f1 - e_trans_f1)
    ssm_diff = abs(d_f1 - e_ssm_f1)

    if trans_diff < 0.02 and ssm_diff < 0.02:
        print(f"\n  VERDICT: Halt F1 is EQUIVALENT (within 2%)")
        print(f"  → The 99% halt F1 comes purely from supervised halt training (β·L_halt)")
        print(f"  → Thermodynamic context (α·L_energy, γ·L_state) does NOT improve halt calibration")
    elif e_trans_f1 < b_f1 - 0.02 or e_ssm_f1 < d_f1 - 0.02:
        print(f"\n  VERDICT: Thermodynamic context IMPROVES halt F1")
        print(f"  → Full L_th outperforms halt-only by {trans_diff*100:.1f}% (Trans), {ssm_diff*100:.1f}% (SSM)")
        print(f"  → Energy/state penalties contribute to halt calibration")
    else:
        print(f"\n  VERDICT: Halt-only OUTPERFORMS full thermodynamic")
        print(f"  → Energy/state penalties may HURT halt calibration")

    # Save ablation results
    ablation_summary = {
        'E_trans': {
            'teacher_forced_accuracy': results['E_trans']['test_accuracy'],
            'halt_f1': results['E_trans']['halt_f1'],
            'free_gen': gen_results['E_trans'],
        },
        'E_ssm': {
            'teacher_forced_accuracy': results['E_ssm']['test_accuracy'],
            'halt_f1': results['E_ssm']['halt_f1'],
            'free_gen': gen_results['E_ssm'],
        },
        'comparison': {
            'B_halt_f1': b_f1,
            'E_trans_halt_f1': e_trans_f1,
            'D_halt_f1': d_f1,
            'E_ssm_halt_f1': e_ssm_f1,
            'trans_diff': trans_diff,
            'ssm_diff': ssm_diff,
        }
    }
    with open(os.path.join(args.results_dir, 'ablation_halt_results.json'), 'w') as f:
        json.dump(ablation_summary, f, indent=2, default=str)

    print(f"\nResults saved to {args.results_dir}/ablation_halt_results.json")


if __name__ == '__main__':
    main()
