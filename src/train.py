"""
PNA-SSM Experiment: Training and Evaluation Pipeline
Trains all 4 groups and runs full evaluation suite.
"""

import os
import sys
import json
import time
import math
import copy
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_datasets, VOCAB_SIZE, VOCAB, detokenize
from models import create_model, count_parameters
from losses import create_loss_fn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_cosine_schedule(optimizer, total_steps, warmup_steps=100):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, dataloader, loss_fn, governor, optimizer, scheduler, device, group):
    """Train for one epoch. Returns average metrics."""
    model.train()
    metrics_accum = defaultdict(float)
    n_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        reasoning_mask = batch['reasoning_mask'].to(device)
        result_pos = batch['result_pos'].to(device)

        outputs = model(input_ids)

        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=targets,
            halt_confidence=outputs['halt_confidence'],
            states_sequence=outputs.get('states_sequence'),
            reasoning_mask=reasoning_mask,
            result_token_positions=result_pos,
        )

        loss = loss_dict['total']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update governor for thermodynamic groups
        if governor is not None:
            alpha, gamma = governor.step(loss_dict)
            loss_fn.alpha = alpha
            loss_fn.gamma = gamma

        for k, v in loss_dict.items():
            if k != 'total' and not isinstance(v, bool):
                metrics_accum[k] += v
            elif k == 'state_leads':
                metrics_accum['state_leads_count'] += int(v)
        metrics_accum['total_loss'] += loss.item()
        n_batches += 1

    return {k: v / n_batches for k, v in metrics_accum.items()}


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Evaluate model. Returns metrics dict."""
    model.eval()
    metrics_accum = defaultdict(float)
    n_batches = 0
    all_correct = 0
    all_total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        reasoning_mask = batch['reasoning_mask'].to(device)
        result_pos = batch['result_pos'].to(device)

        outputs = model(input_ids)
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=targets,
            halt_confidence=outputs['halt_confidence'],
            states_sequence=outputs.get('states_sequence'),
            reasoning_mask=reasoning_mask,
            result_token_positions=result_pos,
        )

        # Compute accuracy at result positions
        preds = outputs['logits'].argmax(dim=-1)
        for b in range(input_ids.size(0)):
            rp = result_pos[b].item()
            if rp < targets.size(1):
                if preds[b, rp] == targets[b, rp]:
                    all_correct += 1
                all_total += 1

        for k, v in loss_dict.items():
            if k != 'total' and not isinstance(v, bool):
                metrics_accum[k] += v
        metrics_accum['total_loss'] += loss_dict['total'].item()
        n_batches += 1

    result = {k: v / n_batches for k, v in metrics_accum.items()}
    result['accuracy'] = all_correct / max(1, all_total)
    return result


def compute_reasoning_length(model, dataloader, device):
    """Measure average reasoning token count."""
    model.eval()
    lengths = []
    with torch.no_grad():
        for batch in dataloader:
            reasoning_mask = batch['reasoning_mask']
            for b in range(reasoning_mask.size(0)):
                lengths.append(reasoning_mask[b].sum().item())
    return np.mean(lengths), np.std(lengths)


@torch.no_grad()
def compute_entropy_trajectories(model, dataloader, device, n_examples=100):
    """Compute token entropy at each position for visualization."""
    model.eval()
    trajectories = []
    state_trajectories = []

    count = 0
    for batch in dataloader:
        if count >= n_examples:
            break
        input_ids = batch['input_ids'].to(device)
        outputs = model(input_ids)
        logits = outputs['logits']

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)  # [B, T]

        for b in range(min(entropy.size(0), n_examples - count)):
            trajectories.append(entropy[b].cpu().numpy())
            if outputs.get('states_sequence') is not None:
                states = outputs['states_sequence'][b]  # [T, d_state]
                state_probs = F.softmax(states, dim=-1)
                state_ent = -(state_probs * torch.log2(state_probs + 1e-9)).sum(dim=-1)
                state_trajectories.append(state_ent.cpu().numpy())
            count += 1

    # Compute mean/std
    max_len = max(len(t) for t in trajectories)
    padded = np.zeros((len(trajectories), max_len))
    for i, t in enumerate(trajectories):
        padded[i, :len(t)] = t

    result = {
        'trajectories': trajectories,
        'mean': padded.mean(axis=0).tolist(),
        'std': padded.std(axis=0).tolist(),
    }

    if state_trajectories:
        padded_s = np.zeros((len(state_trajectories), max_len))
        for i, t in enumerate(state_trajectories):
            padded_s[i, :len(t)] = t
        result['state_trajectories'] = state_trajectories
        result['state_mean'] = padded_s.mean(axis=0).tolist()
        result['state_std'] = padded_s.std(axis=0).tolist()

    return result


@torch.no_grad()
def compute_halt_f1(model, dataloader, device):
    """Compute halt F1 score — precision/recall of halt confidence > 0.5 at result positions."""
    model.eval()
    tp = fp = fn = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        result_pos = batch['result_pos']
        outputs = model(input_ids)
        halt_conf = outputs['halt_confidence'].squeeze(-1).cpu()

        for b in range(input_ids.size(0)):
            rp = result_pos[b].item()
            T = halt_conf.size(1)
            for t in range(T):
                pred_halt = halt_conf[b, t].item() > 0.5
                actual_halt = t >= rp
                if pred_halt and actual_halt:
                    tp += 1
                elif pred_halt and not actual_halt:
                    fp += 1
                elif not pred_halt and actual_halt:
                    fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return f1


def train_group(group, train_dataset, val_dataset, test_dataset,
                device, results_dir, epochs=50, batch_size=32, lr=3e-4):
    """Train a single experimental group end-to-end."""
    print(f"\n{'='*60}")
    print(f"Training Group {group}")
    print(f"{'='*60}")

    model = create_model(group, VOCAB_SIZE, device=device)
    n_params = count_parameters(model)
    arch = "Transformer" if group in ('A', 'B') else "SSM"
    loss_type = "Cross-Entropy" if group in ('A', 'C') else "Thermodynamic"
    print(f"  Architecture: {arch}, Loss: {loss_type}, Params: {n_params:,}")

    loss_fn, governor = create_loss_fn(group, pad_token_id=VOCAB['<PAD>'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, governor, optimizer, scheduler, device, group)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
            'alpha': governor.alpha if governor else 0,
            'gamma': governor.gamma if governor else 0,
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

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    print(f"\n  Final evaluation on test set:")
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"    Loss: {test_metrics['total_loss']:.4f}")

    # Reasoning length
    token_mean, token_std = compute_reasoning_length(model, test_loader, device)
    print(f"    Reasoning tokens: {token_mean:.1f} ± {token_std:.1f}")

    # Halt F1 (thermodynamic groups)
    halt_f1 = compute_halt_f1(model, test_loader, device)
    print(f"    Halt F1: {halt_f1:.4f}")

    # Entropy trajectories
    entropy_data = compute_entropy_trajectories(model, test_loader, device)

    # Save results
    group_results = {
        'group': group,
        'architecture': arch,
        'loss_type': loss_type,
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

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(results_dir, f"group_{group}_model.pt"))
    # Save results JSON
    with open(os.path.join(results_dir, f"group_{group}_results.json"), 'w') as f:
        json.dump(group_results, f, indent=2, default=str)

    return model, group_results


def run_stability_sweep(group, train_dataset, val_dataset, device, results_dir,
                        alphas=(0.1, 0.2, 0.3, 0.4, 0.5), epochs=15, batch_size=32):
    """Training stability sweep at different fixed alpha values."""
    print(f"\n  Stability sweep for Group {group}...")
    results = {}

    for alpha in alphas:
        model = create_model(group, VOCAB_SIZE, device=device)
        from losses import ThermodynamicLoss
        loss_fn = ThermodynamicLoss(alpha=alpha, beta=0.1, gamma=0.05, pad_token_id=VOCAB['<PAD>'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        scheduler = get_cosine_schedule(optimizer, epochs * len(train_loader))

        losses_history = []
        grad_norms = []

        model.train()
        for epoch in range(1, epochs + 1):
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                targets = batch['targets'].to(device)
                reasoning_mask = batch['reasoning_mask'].to(device)
                result_pos = batch['result_pos'].to(device)

                outputs = model(input_ids)
                loss_dict = loss_fn(
                    outputs['logits'], targets, outputs['halt_confidence'],
                    outputs.get('states_sequence'), reasoning_mask, result_pos,
                )
                loss = loss_dict['total']
                optimizer.zero_grad()
                loss.backward()

                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                losses_history.append(loss.item())

        results[alpha] = {
            'loss_variance': float(np.var(losses_history[-100:])),
            'grad_norm_variance': float(np.var(grad_norms[-100:])),
            'final_loss': float(np.mean(losses_history[-100:])),
        }
        print(f"    α={alpha:.1f}: loss_var={results[alpha]['loss_variance']:.4f}, "
              f"grad_var={results[alpha]['grad_norm_variance']:.4f}")

    return results


def run_generalization_test(model, group, device, max_bits=10):
    """Length generalization: test on longer sequences than trained."""
    from dataset import ReasoningDataset
    results = {'sequence_lengths': [], 'accuracies': []}

    model.eval()
    for n_bits in range(2, max_bits + 1):
        dataset = ReasoningDataset(200, bit_range=(n_bits, n_bits),
                                   max_seq_len=128, include_arithmetic=False,
                                   seed=789 + n_bits)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        from losses import create_loss_fn
        loss_fn, _ = create_loss_fn(group, pad_token_id=VOCAB['<PAD>'])

        metrics = evaluate(model, loader, loss_fn, device)
        results['sequence_lengths'].append(n_bits)
        results['accuracies'].append(metrics['accuracy'])
        print(f"    {n_bits} bits: accuracy={metrics['accuracy']:.3f}")

    return results


def check_success_criteria(all_results):
    """Check the 6 success criteria for hypothesis confirmation."""
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")

    d = all_results.get('D', {})
    a = all_results.get('A', {})
    b = all_results.get('B', {})
    c = all_results.get('C', {})

    criteria = []

    # 1. Accuracy: Group D ≥ 95%
    d_acc = d.get('test_accuracy', 0)
    c1 = d_acc >= 0.95
    criteria.append(c1)
    print(f"  1. Accuracy D ≥ 95%: {d_acc*100:.1f}% {'PASS' if c1 else 'FAIL'}")

    # 2. Efficiency: Group D reasoning tokens < 0.6 × Group A
    d_tokens = d.get('token_mean', 999)
    a_tokens = a.get('token_mean', 1)
    c2 = d_tokens < 0.6 * a_tokens
    criteria.append(c2)
    print(f"  2. Efficiency D < 0.6×A: {d_tokens:.1f} vs {0.6*a_tokens:.1f} {'PASS' if c2 else 'FAIL'}")

    # 3. Halt precision: Group D F1 > 93%
    d_f1 = d.get('halt_f1', 0)
    c3 = d_f1 > 0.93
    criteria.append(c3)
    print(f"  3. Halt F1 D > 93%: {d_f1*100:.1f}% {'PASS' if c3 else 'FAIL'}")

    # 4. State collapse: visual inspection (auto-check via entropy slope)
    entropy_data = d.get('entropy_data', {})
    state_mean = entropy_data.get('state_mean', [])
    if len(state_mean) > 5:
        diffs = np.diff(state_mean)
        max_drop = abs(min(diffs)) if len(diffs) > 0 else 0
        c4 = max_drop > 0.3
    else:
        c4 = False
    criteria.append(c4)
    print(f"  4. State step-function: {'PASS' if c4 else 'FAIL'}")

    # 5. Synergy: D improvement > B improvement + C improvement - A
    b_tokens = b.get('token_mean', a_tokens)
    c_tokens = c.get('token_mean', a_tokens)
    b_imp = (a_tokens - b_tokens) / a_tokens if a_tokens > 0 else 0
    c_imp = (a_tokens - c_tokens) / a_tokens if a_tokens > 0 else 0
    d_imp = (a_tokens - d_tokens) / a_tokens if a_tokens > 0 else 0
    additive = b_imp + c_imp
    c5 = d_imp > additive
    criteria.append(c5)
    print(f"  5. Synergy: D_imp={d_imp:.3f} > additive={additive:.3f} {'PASS' if c5 else 'FAIL'}")

    # 6. Generalization: Group D accuracy on 6-10 bits > others
    gen = all_results.get('generalization', {})
    if gen:
        d_gen = gen.get('D', {}).get('accuracies', [])
        a_gen = gen.get('A', {}).get('accuracies', [])
        # Compare mean accuracy on bits 6-10 (indices 4-8 if starting from 2)
        d_gen_mean = np.mean(d_gen[4:]) if len(d_gen) > 4 else 0
        a_gen_mean = np.mean(a_gen[4:]) if len(a_gen) > 4 else 0
        c6 = d_gen_mean > a_gen_mean
    else:
        c6 = False
    criteria.append(c6)
    print(f"  6. Generalization: {'PASS' if c6 else 'FAIL'}")

    passed = sum(criteria)
    print(f"\n  Result: {passed}/6 criteria met")
    if passed >= 6:
        print("  >>> HYPOTHESIS STRONGLY CONFIRMED <<<")
    elif passed >= 4:
        print("  >>> HYPOTHESIS PARTIALLY CONFIRMED (publishable) <<<")
    else:
        print("  >>> HYPOTHESIS NOT CONFIRMED <<<")

    return criteria


def main():
    parser = argparse.ArgumentParser(description="PNA-SSM Experiment")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--groups', type=str, default='A,B,C,D',
                        help='Comma-separated list of groups to train')
    parser.add_argument('--skip-stability', action='store_true')
    parser.add_argument('--skip-generalization', action='store_true')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    groups = [g.strip() for g in args.groups.split(',')]

    # Create datasets
    print("Creating datasets...")
    train_ds, val_ds, test_ds = create_datasets(
        train_n=8000, val_n=1000, test_n=1000, max_seq_len=64
    )
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"  Vocab size: {VOCAB_SIZE}")

    # Train all groups
    all_results = {}
    models = {}
    for group in groups:
        model, results = train_group(
            group, train_ds, val_ds, test_ds, device, args.results_dir,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )
        all_results[group] = results
        models[group] = model

    # Stability sweep (Groups B & D)
    if not args.skip_stability:
        stability_results = {}
        for group in ['B', 'D']:
            if group in groups:
                stability_results[group] = run_stability_sweep(
                    group, train_ds, val_ds, device, args.results_dir
                )
        all_results['training_stability'] = stability_results

        with open(os.path.join(args.results_dir, 'stability_results.json'), 'w') as f:
            json.dump(stability_results, f, indent=2, default=str)

    # Generalization tests
    if not args.skip_generalization:
        gen_results = {}
        for group in groups:
            print(f"\n  Generalization test for Group {group}:")
            gen_results[group] = run_generalization_test(models[group], group, device)
        all_results['generalization'] = gen_results

        with open(os.path.join(args.results_dir, 'generalization_results.json'), 'w') as f:
            json.dump(gen_results, f, indent=2, default=str)

    # Check success criteria
    check_success_criteria(all_results)

    # Save combined results
    with open(os.path.join(args.results_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll results saved to {args.results_dir}/")
    return all_results


if __name__ == '__main__':
    main()
