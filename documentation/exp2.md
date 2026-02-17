# Experiment 2: Cross-Task Transfer Implementation

```python
# src/cross_task_transfer.py
"""
Experiment 2: Cross-Task Halt Transfer

Tests whether halt detection generalizes across tasks:
- Train on parity (already done in Phase 1-9)
- Freeze halt head
- Fine-tune on arithmetic
- Measure halt F1 on arithmetic test set

Hypothesis:
- SSMs: Strong transfer (70-85% F1) - halt tracks general uncertainty
- Transformers: Weak transfer (<35% F1) - halt pattern-matches syntax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from models import create_model, ARCH_CONFIGS
from dataset import VOCAB, VOCAB_SIZE

# ============================================================================
# Arithmetic Dataset (New Task)
# ============================================================================

class ArithmeticDataset(Dataset):
    """
    Multi-step arithmetic: addition and subtraction chains.
    
    Format matches parity structure for fair comparison:
    Input:3+5-2 4+3=7 7-2=5 Result:5<HALT>
    
    Key differences from parity:
    - Different operators (+, - instead of ^)
    - Different intermediate tokens (numbers 0-9)
    - Different semantic meaning (arithmetic vs XOR)
    - Same structural pattern (Input: ... Result:X<HALT>)
    """
    
    def __init__(self, num_samples=10000, min_ops=2, max_ops=6, split='train'):
        self.samples = []
        self.split = split
        
        # Generate examples
        np.random.seed(42 if split == 'train' else 43)
        for _ in range(num_samples):
            self.samples.append(self._generate_example(min_ops, max_ops))
    
    def _generate_example(self, min_ops, max_ops):
        """Generate a single arithmetic chain."""
        num_ops = np.random.randint(min_ops, max_ops + 1)
        
        # Start with a random number 0-9
        current = np.random.randint(0, 10)
        expression_parts = [str(current)]
        reasoning_steps = []
        
        for _ in range(num_ops):
            # Random operation
            op = np.random.choice(['+', '-'])
            operand = np.random.randint(1, 10)
            
            expression_parts.append(op)
            expression_parts.append(str(operand))
            
            # Compute result
            if op == '+':
                next_val = current + operand
            else:
                next_val = current - operand
            
            # Keep result in range [0, 99] for simplicity
            next_val = max(0, min(99, next_val))
            
            # Reasoning step
            reasoning_steps.append(f"{current}{op}{operand}={next_val}")
            current = next_val
        
        # Format: Input:3+5-2 3+5=8 8-2=6 Result:6<HALT>
        input_str = f"Input:{''.join(expression_parts)}"
        reasoning_str = ' '.join(reasoning_steps)
        result_str = f"Result:{current}<HALT>"
        
        full_sequence = f"{input_str} {reasoning_str} {result_str}"
        
        # Tokenize
        tokens = self._tokenize(full_sequence)
        
        # Find Result: position (for halt target)
        result_pos = None
        for i, tok in enumerate(tokens):
            if i > 0 and self._decode_token(tokens[i-1:i+1]) == "Result:":
                result_pos = i + 1  # Position after ':'
                break
        
        return {
            'text': full_sequence,
            'tokens': tokens,
            'result_pos': result_pos,
            'answer': current,
            'expression': ''.join(expression_parts),
            'num_ops': num_ops
        }
    
    def _tokenize(self, text):
        """Character-level tokenization matching VOCAB."""
        return [VOCAB.index(c) if c in VOCAB else VOCAB.index(' ') for c in text]
    
    def _decode_token(self, token_ids):
        """Decode token IDs to string."""
        return ''.join([VOCAB[t] for t in token_ids])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_arithmetic(batch):
    """Collate function for arithmetic dataloader."""
    max_len = max(len(ex['tokens']) for ex in batch)
    
    # Pad sequences
    input_ids = []
    target_ids = []
    result_positions = []
    answers = []
    
    for ex in batch:
        tokens = ex['tokens']
        padded = tokens + [VOCAB.index('<PAD>')] * (max_len - len(tokens))
        
        input_ids.append(padded[:-1])  # Input: all but last
        target_ids.append(padded[1:])  # Target: all but first
        result_positions.append(ex['result_pos'] - 1)  # Adjust for shift
        answers.append(ex['answer'])
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'target_ids': torch.tensor(target_ids, dtype=torch.long),
        'result_positions': torch.tensor(result_positions, dtype=torch.long),
        'answers': torch.tensor(answers, dtype=torch.long)
    }


# ============================================================================
# Transfer Training
# ============================================================================

class TransferTrainer:
    """
    Fine-tune base model on new task while keeping halt head frozen.
    """
    
    def __init__(self, model, device, lr=1e-4):
        self.model = model
        self.device = device
        
        # Freeze halt head
        if hasattr(model, 'halt_head'):
            for param in model.halt_head.parameters():
                param.requires_grad = False
            print("  ✓ Halt head frozen")
        
        # Optimizer only on trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)
        
        print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"  Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")
    
    def train_epoch(self, dataloader):
        """Single training epoch on arithmetic task."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        
        for batch in tqdm(dataloader, desc="  Training", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            # Cross-entropy loss (standard next-token prediction)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=VOCAB.index('<PAD>')
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask = target_ids != VOCAB.index('<PAD>')
                acc = (preds == target_ids)[mask].float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': total_acc / len(dataloader)
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on arithmetic validation set."""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            outputs = self.model(input_ids)
            logits = outputs['logits']
            
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                target_ids.reshape(-1),
                ignore_index=VOCAB.index('<PAD>')
            )
            
            preds = logits.argmax(dim=-1)
            mask = target_ids != VOCAB.index('<PAD>')
            acc = (preds == target_ids)[mask].float().mean()
            
            total_loss += loss.item()
            total_acc += acc.item()
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': total_acc / len(dataloader)
        }


# ============================================================================
# Halt Evaluation on Arithmetic
# ============================================================================

def evaluate_halt_transfer(model, dataloader, device):
    """
    Measure halt F1 on arithmetic task.
    
    Halt target: should fire right after Result:X token
    (same structure as parity, different content)
    """
    model.eval()
    
    halt_predictions = []
    halt_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Evaluating halt", leave=False):
            input_ids = batch['input_ids'].to(device)
            result_positions = batch['result_positions']
            
            outputs = model(input_ids)
            
            if 'halt_confidence' not in outputs:
                print("  ⚠ Model has no halt head, skipping halt eval")
                return {'halt_f1': 0.0, 'halt_precision': 0.0, 'halt_recall': 0.0}
            
            halt_conf = outputs['halt_confidence'].cpu()  # [B, L, 1]
            
            for b in range(halt_conf.shape[0]):
                # Target: 1.0 at result_pos and after, 0.0 before
                target = torch.zeros(halt_conf.shape[1])
                res_pos = result_positions[b].item()
                if res_pos < len(target):
                    target[res_pos:] = 1.0
                
                # Prediction: threshold at 0.5
                pred = (halt_conf[b, :, 0] > 0.5).float()
                
                halt_predictions.extend(pred.tolist())
                halt_targets.extend(target.tolist())
    
    # Compute F1
    halt_predictions = torch.tensor(halt_predictions)
    halt_targets = torch.tensor(halt_targets)
    
    tp = ((halt_predictions == 1) & (halt_targets == 1)).sum().item()
    fp = ((halt_predictions == 1) & (halt_targets == 0)).sum().item()
    fn = ((halt_predictions == 0) & (halt_targets == 1)).sum().item()
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    
    return {
        'halt_f1': f1,
        'halt_precision': precision,
        'halt_recall': recall
    }


# ============================================================================
# Main Transfer Pipeline
# ============================================================================

def run_transfer_experiment(group_name, results_dir, device, epochs=20, early_stop=5):
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
    
    # Load parity-trained model
    checkpoint_path = results_dir / f"group_{group_name}_model.pt"
    if not checkpoint_path.exists():
        print(f"  ✗ Checkpoint not found: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine architecture
    if group_name.startswith('E_'):
        # E_trans uses 'A' arch, E_ssm uses 'C' arch
        arch = 'A' if 'trans' in group_name else 'C'
    else:
        arch = group_name  # A, B, C, D
    
    model = create_model(arch, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"  Loaded from: {checkpoint_path}")
    print(f"  Architecture: {arch}")
    print(f"  Parity halt F1: {checkpoint.get('halt_f1', 'N/A'):.1%}")
    
    # Create arithmetic datasets
    train_dataset = ArithmeticDataset(num_samples=8000, min_ops=2, max_ops=6, split='train')
    val_dataset = ArithmeticDataset(num_samples=1000, min_ops=2, max_ops=6, split='val')
    test_dataset = ArithmeticDataset(num_samples=1000, min_ops=2, max_ops=8, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                              collate_fn=collate_arithmetic)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                            collate_fn=collate_arithmetic)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                             collate_fn=collate_arithmetic)
    
    print(f"  Arithmetic train: {len(train_dataset)} examples")
    print(f"  Arithmetic val: {len(val_dataset)} examples")
    print(f"  Arithmetic test: {len(test_dataset)} examples (ops=2-8)")
    
    # Baseline: Evaluate halt BEFORE fine-tuning
    print("\n  Baseline (zero-shot transfer):")
    baseline_halt = evaluate_halt_transfer(model, test_loader, device)
    print(f"    Halt F1: {baseline_halt['halt_f1']:.1%}")
    
    # Fine-tune with frozen halt head
    trainer = TransferTrainer(model, device, lr=1e-4)
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        
        # Check improvement
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_metrics['accuracy'],
            }, results_dir / f"transfer_{group_name}_best.pt")
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch+1:2d} | train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.3f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.3f} "
              f"{'*' if patience_counter == 0 else ''}")
        
        if patience_counter >= early_stop:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(results_dir / f"transfer_{group_name}_best.pt")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\n  Final evaluation:")
    test_metrics = trainer.evaluate(test_loader)
    final_halt = evaluate_halt_transfer(model, test_loader, device)
    
    print(f"    Arithmetic accuracy: {test_metrics['accuracy']:.1%}")
    print(f"    Halt F1 (after transfer): {final_halt['halt_f1']:.1%}")
    print(f"    Halt precision: {final_halt['halt_precision']:.1%}")
    print(f"    Halt recall: {final_halt['halt_recall']:.1%}")
    
    # Compute transfer metrics
    transfer_improvement = final_halt['halt_f1'] - baseline_halt['halt_f1']
    
    results = {
        'group': group_name,
        'architecture': arch,
        'parity_halt_f1': checkpoint.get('halt_f1', None),
        'baseline_halt_f1': baseline_halt['halt_f1'],
        'final_halt_f1': final_halt['halt_f1'],
        'transfer_improvement': transfer_improvement,
        'arithmetic_accuracy': test_metrics['accuracy'],
        'epochs_trained': epoch + 1,
        'halt_precision': final_halt['halt_precision'],
        'halt_recall': final_halt['halt_recall']
    }
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--groups', type=str, default='B,D,E_trans,E_ssm',
                       help='Comma-separated group names to test')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    device = torch.device(args.device)
    groups = args.groups.split(',')
    
    print(f"Cross-Task Transfer Experiment")
    print(f"Device: {device}")
    print(f"Testing groups: {groups}")
    
    all_results = []
    
    for group in groups:
        result = run_transfer_experiment(
            group_name=group.strip(),
            results_dir=results_dir,
            device=device,
            epochs=args.epochs
        )
        if result:
            all_results.append(result)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("TRANSFER SUMMARY")
    print(f"{'='*80}")
    print(f"{'Group':<12} {'Arch':<8} {'Parity F1':<12} {'Baseline':<12} {'Final F1':<12} {'Improvement':<12}")
    print(f"{'-'*80}")
    
    ssm_results = []
    trans_results = []
    
    for r in all_results:
        print(f"{r['group']:<12} {r['architecture']:<8} "
              f"{r['parity_halt_f1']:.1%}        "
              f"{r['baseline_halt_f1']:.1%}        "
              f"{r['final_halt_f1']:.1%}        "
              f"{r['transfer_improvement']:+.1%}")
        
        if 'ssm' in r['group'].lower() or r['architecture'] in ['C', 'D']:
            ssm_results.append(r)
        else:
            trans_results.append(r)
    
    # Architecture comparison
    if ssm_results and trans_results:
        print(f"\n{'='*80}")
        print("ARCHITECTURE COMPARISON")
        print(f"{'='*80}")
        
        avg_ssm_final = np.mean([r['final_halt_f1'] for r in ssm_results])
        avg_trans_final = np.mean([r['final_halt_f1'] for r in trans_results])
        
        avg_ssm_baseline = np.mean([r['baseline_halt_f1'] for r in ssm_results])
        avg_trans_baseline = np.mean([r['baseline_halt_f1'] for r in trans_results])
        
        print(f"SSM Average:")
        print(f"  Baseline (zero-shot):  {avg_ssm_baseline:.1%}")
        print(f"  Final (after transfer): {avg_ssm_final:.1%}")
        print(f"  Improvement: {avg_ssm_final - avg_ssm_baseline:+.1%}")
        
        print(f"\nTransformer Average:")
        print(f"  Baseline (zero-shot):  {avg_trans_baseline:.1%}")
        print(f"  Final (after transfer): {avg_trans_final:.1%}")
        print(f"  Improvement: {avg_trans_final - avg_trans_baseline:+.1%}")
        
        print(f"\nSSM Advantage: {avg_ssm_final - avg_trans_final:+.1%} F1")
        
        if avg_ssm_final > avg_trans_final + 0.15:
            print("  ✓ STRONG TRANSFER: SSMs generalize meta-cognition across tasks")
        elif avg_ssm_final > avg_trans_final + 0.05:
            print("  ⚠ MODERATE TRANSFER: Some generalization in SSMs")
        else:
            print("  ✗ WEAK TRANSFER: No architecture advantage")
    
    # Save results
    output_file = results_dir / 'cross_task_transfer_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'results': all_results,
            'summary': {
                'ssm_avg_final': avg_ssm_final if ssm_results else None,
                'trans_avg_final': avg_trans_final if trans_results else None,
                'ssm_avg_baseline': avg_ssm_baseline if ssm_results else None,
                'trans_avg_baseline': avg_trans_baseline if trans_results else None,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
```

---

## Usage

```bash
# Run transfer experiment on key groups
python src/cross_task_transfer.py --groups B,D,E_trans,E_ssm

# Quick test on just SSM group
python src/cross_task_transfer.py --groups E_ssm --epochs 10

# Full experiment with all groups
python src/cross_task_transfer.py --groups A,B,C,D,E_trans,E_ssm
```

---

## Expected Output

```
Cross-Task Transfer Experiment
Device: cuda
Testing groups: ['B', 'D', 'E_trans', 'E_ssm']

============================================================
Transfer Experiment: B
============================================================
  Loaded from: results/group_B_model.pt
  Architecture: A (Transformer)
  Parity halt F1: 98.8%

  Baseline (zero-shot transfer):
    Halt F1: 28.3%
  
  Trainable parameters: 5,051,034
  Frozen parameters: 321
  
  Epoch  1 | train_loss=0.7234 train_acc=0.812 val_loss=0.6891 val_acc=0.845 *
  ...
  Epoch 12 | train_loss=0.3421 train_acc=0.923 val_loss=0.4102 val_acc=0.911 *
  Early stopping at epoch 15

  Final evaluation:
    Arithmetic accuracy: 91.2%
    Halt F1 (after transfer): 34.7%
    Halt precision: 41.2%
    Halt recall: 29.8%

============================================================
Transfer Experiment: E_ssm
============================================================
  Loaded from: results/group_E_ssm_model.pt
  Architecture: C (SSM)
  Parity halt F1: 98.7%

  Baseline (zero-shot transfer):
    Halt F1: 71.2%
  
  ...
  
  Final evaluation:
    Arithmetic accuracy: 88.9%
    Halt F1 (after transfer): 82.4%
    Halt precision: 84.1%
    Halt recall: 80.8%

================================================================================
TRANSFER SUMMARY
================================================================================
Group        Arch     Parity F1    Baseline     Final F1     Improvement    
--------------------------------------------------------------------------------
B            A        98.8%        28.3%        34.7%        +6.4%
E_trans      A        99.8%        31.2%        37.1%        +5.9%
D            C        99.2%        68.4%        79.2%        +10.8%
E_ssm        C        98.7%        71.2%        82.4%        +11.2%

================================================================================
ARCHITECTURE COMPARISON
================================================================================
SSM Average:
  Baseline (zero-shot):  69.8%
  Final (after transfer): 80.8%
  Improvement: +11.0%

Transformer Average:
  Baseline (zero-shot):  29.8%
  Final (after transfer): 35.9%
  Improvement: +6.2%

SSM Advantage: +44.9% F1
  ✓ STRONG TRANSFER: SSMs generalize meta-cognition across tasks
```

---

## Key Validation Points

### 1. Zero-Shot Baseline
- **SSMs**: Expected 60-75% (halt head reads uncertainty, which exists in any task)
- **Transformers**: Expected 25-35% (halt head pattern-matches "Result:", which exists but with different context)

### 2. Post-Transfer Final F1
- **SSMs**: Expected 75-85% (learns new task structure, halt adapts)
- **Transformers**: Expected 30-40% (minimal improvement, still pattern matching)

### 3. Architecture Advantage
- **Target**: SSMs should outperform Transformers by 40-50% F1 points
- **Interpretation**: Meta-cognition (SSMs) transfers; heuristics (Transformers) don't

---

## What This Proves

If results match predictions:

**✅ SSM halt detection is task-general meta-cognition**
- 70%+ zero-shot transfer → halt head tracks generic uncertainty
- 80%+ after fine-tuning → adapts to new task structure
- Validates state entropy collapse as general mechanism

**✅ Transformer halt detection is task-specific pattern matching**
- <35% zero-shot transfer → halt head needs exact syntax
- Minimal improvement → can't adapt to new patterns
- Confirms syntactic heuristics, not meta-cognition

**✅ Architecture determines cognitive strategy**
- Same training procedure (supervised halt loss)
- Same external performance (99% F1 on parity)
- Different internal mechanisms
- Different generalization properties

This completes the empirical story for the paper.