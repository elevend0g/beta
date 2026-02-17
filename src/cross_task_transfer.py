"""
Experiment 2: Cross-Task Halt Transfer

Tests whether halt detection generalizes across tasks:
- Train on parity (already done in Phases 1-9)
- Freeze halt head
- Fine-tune on arithmetic
- Measure halt F1 on arithmetic test set

Hypothesis:
- SSMs: Strong transfer (70-85% F1) - halt tracks general uncertainty
- Transformers: Weak transfer (<35% F1) - halt pattern-matches syntax
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize
from models import create_model, count_parameters
from entropy_halt_correlation import GROUP_CONFIG, COLORS, load_group_model


# ============================================================================
# Arithmetic Dataset (New Task)
# ============================================================================

class ArithmeticChainDataset(Dataset):
    """
    Multi-step arithmetic: addition and subtraction chains.

    Format matches parity structure for fair comparison:
    Input:3+5-2 3+5=8 8-2=6 Result:6<HALT>

    Key differences from parity:
    - Different operators (+, - instead of ^)
    - Different intermediate computations (arithmetic vs XOR)
    - Different result range (0-99 vs 0-1)
    - Same structural pattern (Input:... reasoning... Result:X<HALT>)
    """

    def __init__(self, num_samples=10000, min_ops=2, max_ops=6,
                 max_seq_len=64, seed=42):
        self.max_seq_len = max_seq_len
        self.examples = []

        rng = np.random.RandomState(seed)

        for _ in range(num_samples):
            ex = self._generate_example(min_ops, max_ops, rng)
            self.examples.append(ex)

    def _generate_example(self, min_ops, max_ops, rng):
        """Generate a single arithmetic chain example."""
        num_ops = rng.randint(min_ops, max_ops + 1)

        # Start with a random single digit 1-9
        current = int(rng.randint(1, 10))
        expression_parts = [str(current)]
        reasoning_steps = []

        for _ in range(num_ops):
            op = rng.choice(['+', '-'])
            operand = int(rng.randint(1, 10))

            expression_parts.append(op)
            expression_parts.append(str(operand))

            if op == '+':
                next_val = current + operand
            else:
                next_val = current - operand

            # Clamp to [0, 99]
            next_val = max(0, min(99, next_val))

            reasoning_steps.append(f"{current}{op}{operand}={next_val}")
            current = next_val

        # Format: Input:3+5-2 3+5=8 8-2=6 Result:6<HALT>
        input_str = f"Input:{''.join(expression_parts)}"
        reasoning_str = ' '.join(reasoning_steps)
        # Result is a number 0-99 — use last digit mod 10 for single-token result
        # Actually, keep multi-digit for authenticity
        result_str = f"Result:{current}"
        full_text = f"{input_str} {reasoning_str} {result_str}"

        # Tokenize using the shared tokenizer (handles multi-char tokens)
        tokens = [VOCAB['<BOS>']] + tokenize(full_text) + [VOCAB['<HALT>'], VOCAB['<EOS>']]

        # Find Result: token position
        result_pos = None
        for i, t in enumerate(tokens):
            if t == VOCAB['Result:']:
                result_pos = i
                break
        if result_pos is None:
            result_pos = len(tokens) - 3

        # Build reasoning mask
        reasoning_mask = [0] * len(tokens)
        input_pos = None
        for i, t in enumerate(tokens):
            if t == VOCAB['Input:']:
                input_pos = i
                break
        if input_pos is not None and result_pos is not None:
            for i in range(input_pos + 1, result_pos):
                reasoning_mask[i] = 1

        return {
            'tokens': tokens,
            'result_pos': result_pos,
            'reasoning_mask': reasoning_mask,
            'text': full_text,
            'answer': current,
            'num_ops': num_ops,
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        mask = ex['reasoning_mask']

        # Pad or truncate (same logic as ReasoningDataset)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(tokens)
            tokens = tokens + [VOCAB['<PAD>']] * pad_len
            mask = mask + [0] * pad_len

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        reasoning_mask = torch.tensor(mask[:-1], dtype=torch.float)
        result_pos = min(ex['result_pos'], self.max_seq_len - 2)

        return {
            'input_ids': input_ids,
            'targets': targets,
            'reasoning_mask': reasoning_mask,
            'result_pos': torch.tensor(result_pos, dtype=torch.long),
        }


# ============================================================================
# Transfer Training
# ============================================================================

class TransferTrainer:
    """Fine-tune model on arithmetic with halt head frozen."""

    def __init__(self, model, device, lr=1e-4):
        self.model = model
        self.device = device

        # Freeze halt head
        frozen_count = 0
        if hasattr(model, 'halt_head'):
            for param in model.halt_head.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
            print(f"  Halt head frozen ({frozen_count:,} params)")

        # Optimizer only on trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in trainable_params)
        print(f"  Trainable: {trainable:,} / {total:,} params")

    def train_epoch(self, dataloader):
        """Single training epoch on arithmetic task."""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in tqdm(dataloader, desc="  Training", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            logits = outputs['logits']

            # CE loss with PAD masking
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=VOCAB['<PAD>']
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            with torch.no_grad():
                mask = targets != VOCAB['<PAD>']
                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets)[mask].sum().item()
                total_tokens += mask.sum().item()

            total_loss += loss.item()

        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'accuracy': total_correct / max(total_tokens, 1),
        }

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on arithmetic set."""
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)

            outputs = self.model(input_ids)
            logits = outputs['logits']

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
                ignore_index=VOCAB['<PAD>']
            )

            mask = targets != VOCAB['<PAD>']
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets)[mask].sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()

        n = len(dataloader)
        return {
            'loss': total_loss / n,
            'accuracy': total_correct / max(total_tokens, 1),
        }


# ============================================================================
# Halt Evaluation on Arithmetic
# ============================================================================

@torch.no_grad()
def evaluate_halt_transfer(model, dataloader, device):
    """
    Measure halt F1 on arithmetic task.
    Halt target: confidence should be high at and after Result:X position.
    """
    model.eval()

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for batch in tqdm(dataloader, desc="  Halt eval", leave=False):
        input_ids = batch['input_ids'].to(device)
        result_positions = batch['result_pos']

        outputs = model(input_ids)
        if 'halt_confidence' not in outputs or outputs['halt_confidence'] is None:
            return {'halt_f1': 0.0, 'halt_precision': 0.0, 'halt_recall': 0.0}

        halt_conf = outputs['halt_confidence'].cpu()  # [B, L, 1]

        for b in range(halt_conf.shape[0]):
            seq_len = halt_conf.shape[1]
            res_pos = result_positions[b].item()

            # Target: 1.0 at result_pos and after, 0.0 before
            target = torch.zeros(seq_len)
            if res_pos < seq_len:
                target[res_pos:] = 1.0

            # Prediction: threshold at 0.5
            pred = (halt_conf[b, :, 0] > 0.5).float()

            # Only count non-padding positions
            # (PAD tokens are after <EOS>, halt should be 1 there too — fine)
            tp = ((pred == 1) & (target == 1)).sum().item()
            fp = ((pred == 1) & (target == 0)).sum().item()
            fn = ((pred == 0) & (target == 1)).sum().item()

            all_tp += tp
            all_fp += fp
            all_fn += fn

    precision = all_tp / (all_tp + all_fp + 1e-9)
    recall = all_tp / (all_tp + all_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'halt_f1': f1,
        'halt_precision': precision,
        'halt_recall': recall,
    }


# ============================================================================
# Main Transfer Pipeline
# ============================================================================

def load_parity_halt_f1(group, results_dir):
    """Load halt F1 from parity results JSON."""
    # Check group-specific results
    json_path = os.path.join(results_dir, f"group_{group}_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return data.get('halt_f1', None)

    # Check ablation results for E_ groups
    ablation_path = os.path.join(results_dir, "ablation_halt_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path) as f:
            data = json.load(f)
        if group in data:
            return data[group].get('halt_f1', None)

    return None


def run_transfer_experiment(group_name, results_dir, device,
                            epochs=20, early_stop=5, lr=1e-4):
    """
    Full transfer pipeline for one group:
    1. Load parity-trained model
    2. Freeze halt head
    3. Fine-tune on arithmetic
    4. Evaluate halt F1 on arithmetic
    """
    print(f"\n{'='*60}")
    print(f"Transfer Experiment: {group_name}")
    print(f"{'='*60}")

    if group_name not in GROUP_CONFIG:
        print(f"  Unknown group: {group_name}")
        return None

    cfg = GROUP_CONFIG[group_name]

    # Load parity-trained model
    ckpt_path = os.path.join(results_dir, f"group_{group_name}_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    model = load_group_model(group_name, results_dir, device)
    # Re-enable training mode (load_group_model sets eval)
    model.train()

    parity_halt_f1 = load_parity_halt_f1(group_name, results_dir)
    print(f"  Architecture: {cfg['arch_group']} ({'SSM' if cfg['is_ssm'] else 'Transformer'})")
    print(f"  Parity halt F1: {parity_halt_f1:.1%}" if parity_halt_f1 else "  Parity halt F1: N/A")
    print(f"  Parameters: {count_parameters(model):,}")

    # Create arithmetic datasets
    train_ds = ArithmeticChainDataset(num_samples=8000, min_ops=2, max_ops=6, seed=42)
    val_ds = ArithmeticChainDataset(num_samples=1000, min_ops=2, max_ops=6, seed=123)
    test_ds = ArithmeticChainDataset(num_samples=1000, min_ops=2, max_ops=8, seed=456)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    print(f"  Arithmetic train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Sample text for sanity check
    print(f"  Sample: {train_ds.examples[0]['text']}")

    # Baseline: evaluate halt BEFORE fine-tuning
    print("\n  Baseline (zero-shot transfer):")
    baseline_halt = evaluate_halt_transfer(model, test_loader, device)
    print(f"    Halt F1: {baseline_halt['halt_f1']:.1%} "
          f"(P={baseline_halt['halt_precision']:.1%}, R={baseline_halt['halt_recall']:.1%})")

    # Fine-tune with frozen halt head
    trainer = TransferTrainer(model, device, lr=lr)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        improved = val_metrics['accuracy'] > best_val_acc
        if improved:
            best_val_acc = val_metrics['accuracy']
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"  Epoch {epoch+1:2d} | "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f} "
              f"{'*' if improved else ''}")

        if patience_counter >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n  Final evaluation:")
    test_metrics = trainer.evaluate(test_loader)
    final_halt = evaluate_halt_transfer(model, test_loader, device)

    print(f"    Arithmetic accuracy: {test_metrics['accuracy']:.1%}")
    print(f"    Halt F1 (transfer): {final_halt['halt_f1']:.1%} "
          f"(P={final_halt['halt_precision']:.1%}, R={final_halt['halt_recall']:.1%})")

    transfer_delta = final_halt['halt_f1'] - baseline_halt['halt_f1']

    result = {
        'group': group_name,
        'architecture': 'SSM' if cfg['is_ssm'] else 'Transformer',
        'arch_group': cfg['arch_group'],
        'label': cfg['label'],
        'parity_halt_f1': parity_halt_f1,
        'baseline_halt_f1': baseline_halt['halt_f1'],
        'baseline_halt_precision': baseline_halt['halt_precision'],
        'baseline_halt_recall': baseline_halt['halt_recall'],
        'final_halt_f1': final_halt['halt_f1'],
        'final_halt_precision': final_halt['halt_precision'],
        'final_halt_recall': final_halt['halt_recall'],
        'transfer_delta': transfer_delta,
        'arithmetic_accuracy': test_metrics['accuracy'],
        'arithmetic_loss': test_metrics['loss'],
        'epochs_trained': epoch + 1,
        'best_val_accuracy': best_val_acc,
    }

    return result


# ============================================================================
# Visualization
# ============================================================================

def plot_transfer_results(all_results, save_path):
    """
    Two-panel figure:
    Left:  Grouped bar chart (parity F1, baseline F1, transfer F1) per group
    Right: SSM vs Transformer comparison (baseline + transfer)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    groups = [r['group'] for r in all_results]
    n = len(groups)
    x = np.arange(n)
    w = 0.25

    # Panel 1: Per-group bars
    parity_f1s = [r.get('parity_halt_f1', 0) or 0 for r in all_results]
    baseline_f1s = [r['baseline_halt_f1'] for r in all_results]
    final_f1s = [r['final_halt_f1'] for r in all_results]

    bars1 = ax1.bar(x - w, parity_f1s, w, label='Parity F1', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x, baseline_f1s, w, label='Baseline (zero-shot)', color='#e67e22', alpha=0.8)
    bars3 = ax1.bar(x + w, final_f1s, w, label='Transfer F1', color='#2ecc71', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r['group']}\n({r['label']})" for r in all_results], fontsize=9)
    ax1.set_ylabel('Halt F1')
    ax1.set_ylim(0, 1.05)
    ax1.set_title('Halt F1: Parity vs Arithmetic Transfer', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                         f'{h:.0%}', ha='center', va='bottom', fontsize=7)

    # Panel 2: Architecture comparison
    ssm_results = [r for r in all_results if r['architecture'] == 'SSM']
    trans_results = [r for r in all_results if r['architecture'] == 'Transformer']

    categories = ['Baseline\n(zero-shot)', 'After\nTransfer']
    x2 = np.arange(len(categories))
    w2 = 0.3

    if ssm_results:
        ssm_baseline = np.mean([r['baseline_halt_f1'] for r in ssm_results])
        ssm_final = np.mean([r['final_halt_f1'] for r in ssm_results])
        ax2.bar(x2 - w2/2, [ssm_baseline, ssm_final], w2,
                label='SSM', color='#e74c3c', alpha=0.8)

    if trans_results:
        trans_baseline = np.mean([r['baseline_halt_f1'] for r in trans_results])
        trans_final = np.mean([r['final_halt_f1'] for r in trans_results])
        ax2.bar(x2 + w2/2, [trans_baseline, trans_final], w2,
                label='Transformer', color='#3498db', alpha=0.8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('Mean Halt F1')
    ax2.set_ylim(0, 1.05)
    ax2.set_title('Architecture Comparison: Halt Transfer', fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    fig.suptitle('Cross-Task Halt Transfer: Parity → Arithmetic',
                 fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Task Halt Transfer Experiment')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--figures-dir', type=str, default='figures')
    parser.add_argument('--groups', type=str, default='B,D,E_trans,E_ssm',
                        help='Comma-separated group names')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--early-stop', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    results_dir = args.results_dir
    figures_dir = args.figures_dir
    os.makedirs(figures_dir, exist_ok=True)
    device = torch.device(args.device)
    groups = [g.strip() for g in args.groups.split(',')]

    print(f"Cross-Task Transfer Experiment")
    print(f"Device: {device}")
    print(f"Groups: {groups}")
    print(f"Epochs: {args.epochs}, Early stop: {args.early_stop}, LR: {args.lr}")

    all_results = []

    for group in groups:
        result = run_transfer_experiment(
            group_name=group,
            results_dir=results_dir,
            device=device,
            epochs=args.epochs,
            early_stop=args.early_stop,
            lr=args.lr,
        )
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("\nNo results produced.")
        return

    # Summary
    print(f"\n{'='*80}")
    print("TRANSFER SUMMARY")
    print(f"{'='*80}")
    print(f"{'Group':<12} {'Arch':<14} {'Parity F1':<12} {'Baseline':<12} "
          f"{'Transfer F1':<12} {'Delta':<10} {'Arith Acc':<10}")
    print(f"{'-'*80}")

    ssm_results = []
    trans_results = []

    for r in all_results:
        pf1 = f"{r['parity_halt_f1']:.1%}" if r['parity_halt_f1'] else "N/A"
        print(f"{r['group']:<12} {r['architecture']:<14} "
              f"{pf1:<12} {r['baseline_halt_f1']:<12.1%} "
              f"{r['final_halt_f1']:<12.1%} {r['transfer_delta']:+9.1%} "
              f"{r['arithmetic_accuracy']:<10.1%}")

        if r['architecture'] == 'SSM':
            ssm_results.append(r)
        else:
            trans_results.append(r)

    # Architecture comparison
    if ssm_results and trans_results:
        print(f"\n{'='*80}")
        print("ARCHITECTURE COMPARISON")
        print(f"{'='*80}")

        avg_ssm_baseline = np.mean([r['baseline_halt_f1'] for r in ssm_results])
        avg_ssm_final = np.mean([r['final_halt_f1'] for r in ssm_results])
        avg_trans_baseline = np.mean([r['baseline_halt_f1'] for r in trans_results])
        avg_trans_final = np.mean([r['final_halt_f1'] for r in trans_results])

        print(f"\nSSM Average:")
        print(f"  Baseline (zero-shot):  {avg_ssm_baseline:.1%}")
        print(f"  Final (after transfer): {avg_ssm_final:.1%}")
        print(f"  Improvement: {avg_ssm_final - avg_ssm_baseline:+.1%}")

        print(f"\nTransformer Average:")
        print(f"  Baseline (zero-shot):  {avg_trans_baseline:.1%}")
        print(f"  Final (after transfer): {avg_trans_final:.1%}")
        print(f"  Improvement: {avg_trans_final - avg_trans_baseline:+.1%}")

        advantage = avg_ssm_final - avg_trans_final
        print(f"\nSSM Advantage: {advantage:+.1%} F1")

        if advantage > 0.15:
            print("  => STRONG TRANSFER: SSMs generalize halt across tasks")
        elif advantage > 0.05:
            print("  => MODERATE TRANSFER: Some SSM generalization")
        else:
            print("  => WEAK TRANSFER: No clear architecture advantage")

    # Generate figure
    plot_transfer_results(
        all_results,
        os.path.join(figures_dir, 'fig14_cross_task_transfer.png')
    )

    # Save results
    output = {
        'results': all_results,
    }
    if ssm_results and trans_results:
        output['summary'] = {
            'ssm_avg_baseline': float(avg_ssm_baseline),
            'ssm_avg_final': float(avg_ssm_final),
            'trans_avg_baseline': float(avg_trans_baseline),
            'trans_avg_final': float(avg_trans_final),
            'ssm_advantage': float(advantage),
        }

    output_path = os.path.join(results_dir, 'cross_task_transfer_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
