"""
Phase 14F: Basin Deepening — Train a Fresh SSM Without Halt Head

Architecture: PNA_SSM_RIM minus halt head
  - d_model=512, d_state=16, n_layers=4
  - Keeps: state_corrector, value_decoder
  - Removed: halt_confidence output head
  - Stopping: <EOS> token (already in vocabulary)

Loss:
  L = L_CE + delta * L_constraint + lambda_term * L_terminal + lambda_app * L_approach

  L_CE:         Standard cross-entropy on next token prediction
  L_constraint: RuleConstraintLoss (identity, cancel, collapse rules on SSM states)
  L_terminal:   Extra CE weight on EOS prediction at result positions
  L_approach:   State convergence penalty — consecutive state deltas should
                decrease as reasoning approaches the answer

Training Schedule (70 epochs):
  Epochs  1-10:  L_CE only (establish sequence modeling)
  Epochs 10-30:  + L_constraint annealed 0 -> delta (build canals)
  Epochs 30-50:  + L_terminal + L_approach annealed 0 -> lambda (dig floors)
  Epochs 50-70:  All losses at full weight (consolidation)

Inference:
  Generate until <EOS> or max_length=200
  Parse answer from "Result:X"
  No halt_confidence check
"""

import os
import sys
import re
import json
import time
import copy
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, MambaBlock, count_parameters
from train import get_device, get_cosine_schedule
from compressible_task import (
    CompressibleArithmeticDataset,
    detect_oscillation, classify_convergence,
    TIER_NAMES, TIER_COLORS,
)
from rule_initialization import (
    RuleConstraintLoss, RIMDatasetWrapper, GeodesicPurity,
    evaluate_purity_corrected, find_op_token_positions,
    OP_PAD, OP_REAL, OP_IDENTITY, OP_CANCEL_START, OP_CANCEL_END,
    OP_STAR_ZERO, MAX_OPS,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================================
# Model: PNA_SSM without halt head, 4 layers
# ============================================================================

class PNA_SSM_Basin(PNA_SSM):
    """
    PNA_SSM_RIM architecture minus the halt head.

    Same: d_model=512, d_state=16, SSM layers, state_corrector, value_decoder
    Changed: n_layers=4 (from 6)
    Removed: halt_head, halt_confidence output
    Stopping: relies on <EOS> token prediction, not halt confidence
    """

    def __init__(self, vocab_size, d_model=512, n_layers=4, d_state=16,
                 max_seq_len=256, correction_scale=0.1):
        # Initialize PNA_SSM (which creates halt_head — we'll delete it)
        super().__init__(vocab_size, d_model, n_layers, d_state, max_seq_len)

        # Remove halt head — this model uses EOS tokens for stopping
        del self.halt_head

        self.correction_scale = correction_scale

        # Learned state correction (from RIM)
        self.state_corrector = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.ReLU(),
            nn.Linear(d_state * 2, d_state),
            nn.Tanh(),
        )

        # Value decoder probe
        self.value_decoder = nn.Linear(d_state, 20)

    def forward(self, x: torch.Tensor) -> dict:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)

        all_states = None
        for layer in self.layers:
            h, all_states = layer(h)

        h = self.norm(h)
        logits = self.token_head(h)

        # State correction
        correction = self.state_corrector(all_states)
        corrected_states = all_states + self.correction_scale * correction

        # Value decoder
        value_logits = self.value_decoder(corrected_states)

        return {
            "logits": logits,
            "halt_confidence": None,
            "states_sequence": corrected_states,
            "final_state": corrected_states[:, -1, :],
            "raw_states": all_states,
            "value_logits": value_logits,
        }


# ============================================================================
# Losses
# ============================================================================

class TerminalLoss(nn.Module):
    """
    Extra cross-entropy weight on EOS prediction at and after result positions.

    At result_pos and beyond, the model should strongly predict <EOS>.
    This teaches the model to terminate cleanly without a halt head.
    """

    def __init__(self, eos_id, ramp_steps=3):
        super().__init__()
        self.eos_id = eos_id
        self.ramp_steps = ramp_steps

    def forward(self, logits, result_token_positions):
        """
        Args:
            logits: [B, L, V] — full logit tensor
            result_token_positions: [B] — position of Result: token

        Returns: scalar loss
        """
        B, L, V = logits.shape
        device = logits.device

        # Build a weight mask: high weight near and after result position
        weight_mask = torch.zeros(B, L, device=device)
        for b in range(B):
            rp = result_token_positions[b].item()
            # Ramp up over `ramp_steps` before result
            for t in range(max(0, rp - self.ramp_steps), min(rp, L)):
                progress = (t - (rp - self.ramp_steps)) / self.ramp_steps
                weight_mask[b, t] = progress
            # Full weight at and after result
            for t in range(rp, L):
                weight_mask[b, t] = 1.0

        # Target: EOS at these positions
        eos_target = torch.full((B, L), self.eos_id, dtype=torch.long, device=device)

        # Compute per-position CE loss for EOS prediction
        log_probs = F.log_softmax(logits, dim=-1)
        eos_log_prob = log_probs[:, :, self.eos_id]  # [B, L]
        loss = -eos_log_prob * weight_mask

        return loss.sum() / (weight_mask.sum() + 1e-9)


class ApproachLoss(nn.Module):
    """
    State convergence loss — encourages decreasing state deltas as
    the model approaches the answer.

    In the reasoning region, consecutive state deltas ||s_t - s_{t-1}||
    should be non-increasing. Near the result position, states should
    have settled (small deltas).
    """

    def forward(self, states, reasoning_mask, result_token_positions):
        """
        Args:
            states: [B, L, D] — SSM state sequence
            reasoning_mask: [B, L] — 1 during reasoning, 0 elsewhere
            result_token_positions: [B] — position of Result: token

        Returns: scalar loss
        """
        B, L, D = states.shape
        device = states.device

        if L < 3:
            return torch.tensor(0.0, device=device)

        # Consecutive deltas
        deltas = (states[:, 1:, :] - states[:, :-1, :]).norm(dim=-1)  # [B, L-1]

        # Monotonicity penalty: delta_t should be <= delta_{t-1}
        # penalty = ReLU(delta_t - delta_{t-1}) for reasoning positions
        delta_increase = F.relu(deltas[:, 1:] - deltas[:, :-1])  # [B, L-2]

        # Weight by reasoning mask (shifted to align with deltas)
        mask = reasoning_mask[:, 2:]  # [B, L-2] aligns with delta_increase

        # Extra weight near result position (states should be most settled there)
        approach_weight = torch.ones(B, max(L - 2, 0), device=device)
        for b in range(B):
            rp = result_token_positions[b].item()
            for t in range(max(L - 2, 0)):
                dist = max(rp - t - 2, 0)  # distance to result
                if dist < 5:
                    approach_weight[b, t] = 2.0 + (5 - dist) * 0.5

        weighted = delta_increase * mask * approach_weight
        return weighted.sum() / (mask.sum() + 1e-9)


# ============================================================================
# Training
# ============================================================================

def get_loss_weights(epoch):
    """
    Phased loss schedule:
      Epochs  1-10:  L_CE only
      Epochs 10-30:  + L_constraint annealed 0 -> delta_max
      Epochs 30-50:  + L_terminal + L_approach annealed 0 -> lambda_max
      Epochs 50-70:  All at full weight
    """
    delta_max = 1.0
    lambda_max = 0.5

    # Constraint weight (delta)
    if epoch < 10:
        delta = 0.0
    elif epoch < 30:
        delta = delta_max * (epoch - 10) / 20
    else:
        delta = delta_max

    # Terminal + approach weight (lambda)
    if epoch < 30:
        lam = 0.0
    elif epoch < 50:
        lam = lambda_max * (epoch - 30) / 20
    else:
        lam = lambda_max

    return delta, lam


def train_one_epoch(model, dataloader, ce_loss_fn, constraint_loss_fn,
                    terminal_loss_fn, approach_loss_fn,
                    optimizer, scheduler, device,
                    delta, lam):
    """Train for one epoch with phased losses."""
    model.train()
    metrics = defaultdict(float)
    n_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        reasoning_mask = batch['reasoning_mask'].to(device)
        result_pos = batch['result_pos'].to(device)
        op_types = batch['op_types'].to(device)
        op_before_pos = batch['op_before_pos'].to(device)
        op_after_pos = batch['op_after_pos'].to(device)
        n_ops = batch['n_ops'].to(device)

        outputs = model(input_ids)
        logits = outputs['logits']
        raw_states = outputs.get('raw_states', outputs['states_sequence'])
        states = outputs['states_sequence']

        # L_CE
        B, L, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1),
            ignore_index=VOCAB['<PAD>'],
        )

        # L_constraint
        if delta > 0:
            constraint_dict = constraint_loss_fn(
                raw_states, op_types, op_before_pos, op_after_pos, n_ops
            )
            constraint_loss = constraint_dict['total']
        else:
            constraint_loss = torch.tensor(0.0, device=device)

        # L_terminal
        if lam > 0:
            terminal_loss = terminal_loss_fn(logits, result_pos)
            approach_loss = approach_loss_fn(states, reasoning_mask, result_pos)
        else:
            terminal_loss = torch.tensor(0.0, device=device)
            approach_loss = torch.tensor(0.0, device=device)

        # Combined
        total = ce_loss + delta * constraint_loss + lam * (terminal_loss + approach_loss)

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        metrics['total_loss'] += total.item()
        metrics['ce_loss'] += ce_loss.item()
        metrics['constraint_loss'] += constraint_loss.item()
        metrics['terminal_loss'] += terminal_loss.item()
        metrics['approach_loss'] += approach_loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics.items()}


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    """Evaluate accuracy on validation set."""
    model.eval()
    total_correct = 0
    total_count = 0
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        result_pos = batch['result_pos']

        outputs = model(input_ids)
        logits = outputs['logits']

        B, L, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1),
            ignore_index=VOCAB['<PAD>'],
        )
        total_loss += ce_loss.item()

        # Accuracy at result positions
        preds = logits.argmax(dim=-1)
        for b in range(B):
            rp = result_pos[b].item()
            if rp < L:
                if preds[b, rp] == targets[b, rp]:
                    total_correct += 1
                total_count += 1
        n_batches += 1

    acc = total_correct / max(total_count, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return {'accuracy': acc, 'loss': avg_loss}


# ============================================================================
# Generator (EOS-based stopping)
# ============================================================================

class BasinGenerator:
    """
    Autoregressive generation for PNA_SSM_Basin.
    Stops on <EOS> token or max_length. No halt confidence check.
    """

    def __init__(self, model, device='cpu', max_length=200):
        self.model = model
        self.device = device
        self.max_length = max_length
        self.eos_id = VOCAB['<EOS>']
        self.halt_id = VOCAB.get('<HALT>', -1)

    def generate(self, expression):
        prompt_text = f"Input:{expression} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        halt_confidences = []
        state_entropies = []
        state_vectors = []
        stop_reason = "max_length"

        self.model.eval()
        with torch.no_grad():
            for step in range(self.max_length):
                input_tensor = torch.tensor(
                    [generated_ids], dtype=torch.long, device=self.device
                )

                max_pos = getattr(self.model, 'max_seq_len', 256)
                if hasattr(self.model, 'pos_encoding'):
                    max_pos = self.model.pos_encoding.num_embeddings
                if input_tensor.size(1) > max_pos:
                    input_tensor = input_tensor[:, -max_pos:]

                outputs = self.model(input_tensor)
                logits = outputs["logits"][:, -1, :]

                # No halt confidence — set to 0
                halt_confidences.append(0.0)

                # State entropy
                if outputs.get("states_sequence") is not None:
                    last_state = outputs["states_sequence"][:, -1, :]
                    state_vec = last_state.squeeze(0).cpu()
                    state_vectors.append(state_vec)
                    energy = last_state ** 2
                    probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
                    entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).item()
                    state_entropies.append(entropy)
                else:
                    state_entropies.append(0.0)

                # Greedy decode
                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                # Stop on EOS or HALT token
                if next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break

        text = detokenize(generated_ids)
        parsed_answer = self._parse_arithmetic_result(text)
        reasoning_tokens = self._count_reasoning_tokens(generated_ids)

        return {
            'parsed_answer': parsed_answer,
            'generated_text': text,
            'generated_ids': generated_ids,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': len(generated_ids),
            'stop_reason': stop_reason,
            'state_vectors': state_vectors,
            'halt_confidences': halt_confidences,
            'state_entropies': state_entropies,
            'confusion_scores': [],
        }

    def _parse_arithmetic_result(self, text):
        match = re.search(r'Result:(\d+)', text)
        if match:
            return int(match.group(1))
        eq_matches = re.findall(r'=(\d+)', text)
        if eq_matches:
            return int(eq_matches[-1])
        num_matches = re.findall(r'(\d+)', text)
        if num_matches:
            return int(num_matches[-1])
        return None

    def _count_reasoning_tokens(self, generated_ids):
        result_id = VOCAB.get('Result:', -1)
        input_id = VOCAB.get('Input:', -1)
        input_pos = 0
        result_pos = len(generated_ids)
        for i, tid in enumerate(generated_ids):
            if tid == input_id:
                input_pos = i
            if tid == result_id:
                result_pos = i
                break
        return max(0, result_pos - input_pos - 1)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_generation(model, test_ds, device, max_length=200):
    """Run generation on test set and compute tier metrics."""
    generator = BasinGenerator(model, device=device, max_length=max_length)

    gen_results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        gen = generator.generate(ex['expression'])

        is_correct = (gen['parsed_answer'] == ex['answer'])
        osc = detect_oscillation(gen['state_vectors'])
        convergence = classify_convergence(gen, osc)

        gen_results.append({
            'tier': ex['tier'],
            'is_correct': is_correct,
            'convergence': convergence,
            'reasoning_tokens': gen['reasoning_tokens'],
            'total_tokens': gen['total_tokens'],
            'stop_reason': gen['stop_reason'],
            'state_vectors': gen['state_vectors'],
            'halt_confidences': gen['halt_confidences'],
            'state_entropies': gen['state_entropies'],
            'generated_ids': gen['generated_ids'],
            'generated_text': gen.get('generated_text', '')[:200],
            'parsed_answer': gen['parsed_answer'],
            'expression': ex['expression'],
            'ground_truth': ex['answer'],
            'effective_ops': ex['effective_ops'],
            'num_ops': ex['num_ops'],
            'confusion_scores': [],
        })

    # Per-tier metrics
    tier_metrics = {}
    for tier in [0, 1, 2]:
        tier_results = [r for r in gen_results if r['tier'] == tier]
        if not tier_results:
            continue
        tier_metrics[tier] = {
            'n': len(tier_results),
            'accuracy': float(np.mean([r['is_correct'] for r in tier_results])),
            'mean_tokens': float(np.mean([r['reasoning_tokens'] for r in tier_results])),
            'true_convergence': sum(
                1 for r in tier_results if r['convergence'] == 'true_convergence'
            ),
            'false_convergence': sum(
                1 for r in tier_results if r['convergence'] == 'false_convergence'
            ),
        }

    overall_acc = float(np.mean([r['is_correct'] for r in gen_results]))
    mean_tokens = float(np.mean([r['reasoning_tokens'] for r in gen_results]))

    # Stop reason distribution
    stop_reasons = defaultdict(int)
    for r in gen_results:
        stop_reasons[r['stop_reason']] += 1

    return {
        'gen_results': gen_results,
        'tier_metrics': tier_metrics,
        'overall_accuracy': overall_acc,
        'mean_tokens': mean_tokens,
        'stop_reasons': dict(stop_reasons),
        'n': len(gen_results),
    }


# ============================================================================
# Figures
# ============================================================================

def plot_training_curves(history, fig_dir):
    """Plot training loss curves with phase annotations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = [h['epoch'] for h in history]

    # Total loss
    ax = axes[0]
    ax.plot(epochs, [h['total_loss'] for h in history], 'b-', label='Total')
    ax.plot(epochs, [h['ce_loss'] for h in history], 'g--', label='CE')
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses", fontweight='bold')
    ax.legend()

    # Component losses
    ax = axes[1]
    ax.plot(epochs, [h['constraint_loss'] for h in history], 'r-', label='Constraint')
    ax.plot(epochs, [h['terminal_loss'] for h in history], 'm-', label='Terminal')
    ax.plot(epochs, [h['approach_loss'] for h in history], 'c-', label='Approach')
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Component Losses", fontweight='bold')
    ax.legend()

    # Validation accuracy
    ax = axes[2]
    ax.plot(epochs, [h['val_acc'] for h in history], 'k-')
    ax.axvline(x=10, color='gray', linestyle=':', alpha=0.5, label='Phase boundary')
    ax.axvline(x=30, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy", fontweight='bold')
    ax.legend()

    # Phase labels
    for ax in axes:
        ax.text(5, ax.get_ylim()[1] * 0.95, "CE\nonly", ha='center',
                fontsize=8, color='gray')
        ax.text(20, ax.get_ylim()[1] * 0.95, "+Constr", ha='center',
                fontsize=8, color='gray')
        ax.text(40, ax.get_ylim()[1] * 0.95, "+Term\n+App", ha='center',
                fontsize=8, color='gray')
        ax.text(60, ax.get_ylim()[1] * 0.95, "Full", ha='center',
                fontsize=8, color='gray')

    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig32_basin_training.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_generation_results(eval_results, fig_dir):
    """Plot generation accuracy by tier and stop reason distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tm = eval_results['tier_metrics']
    tiers = sorted(tm.keys())
    tier_labels = [TIER_NAMES.get(t, f"T{t}") for t in tiers]
    accs = [tm[t]['accuracy'] for t in tiers]
    colors_t = [TIER_COLORS.get(t, '#333') for t in tiers]

    ax1.bar(tier_labels, accs, color=colors_t)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Generation Accuracy by Tier", fontweight='bold')
    ax1.set_ylim(0, 1)
    for i, (label, acc) in enumerate(zip(tier_labels, accs)):
        ax1.text(i, acc + 0.02, f"{acc:.1%}", ha='center')

    # Stop reasons
    sr = eval_results['stop_reasons']
    reasons = list(sr.keys())
    counts = [sr[r] for r in reasons]
    ax2.bar(reasons, counts, color='#3498db')
    ax2.set_ylabel("Count")
    ax2.set_title("Stop Reason Distribution", fontweight='bold')

    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig33_basin_generation.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 14F: Basin Deepening")
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--val-n', type=int, default=1000)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 14F: Basin Deepening — Fresh SSM Without Halt Head")
    print(f"{'=' * 60}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Create Datasets (matching Phase 12/14) ----
    print("\n--- Creating Datasets ---")
    train_ds = CompressibleArithmeticDataset(
        num_samples=args.train_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=42,
    )
    val_ds = CompressibleArithmeticDataset(
        num_samples=args.val_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=123,
    )
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456,
    )
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Val:   {len(val_ds)} examples")
    print(f"  Test:  {len(test_ds)} examples")

    # Wrap with operation metadata
    rim_train = RIMDatasetWrapper(train_ds)
    rim_val = RIMDatasetWrapper(val_ds)

    train_loader = DataLoader(rim_train, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(rim_val, batch_size=args.batch_size, shuffle=False)

    # ---- Create Model ----
    print("\n--- Creating Model ---")
    model = PNA_SSM_Basin(
        VOCAB_SIZE, d_model=512, n_layers=4, d_state=16, max_seq_len=64,
    ).to(device)
    n_params = count_parameters(model)
    print(f"  Architecture: PNA_SSM_Basin (no halt head)")
    print(f"  Layers: 4, d_model: 512, d_state: 16")
    print(f"  Parameters: {n_params:,}")

    # ---- Loss Functions ----
    constraint_loss_fn = RuleConstraintLoss()
    terminal_loss_fn = TerminalLoss(eos_id=VOCAB['<EOS>'], ramp_steps=3)
    approach_loss_fn = ApproachLoss()

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.999), weight_decay=0.01,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    # ---- Training Loop ----
    print(f"\n--- Training ({args.epochs} epochs) ---")
    print("  Schedule:")
    print("    Epochs  1-10:  L_CE only")
    print("    Epochs 10-30:  + L_constraint (annealed)")
    print("    Epochs 30-50:  + L_terminal + L_approach (annealed)")
    print("    Epochs 50-70:  All at full weight")

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        delta, lam = get_loss_weights(epoch)

        train_metrics = train_one_epoch(
            model, train_loader,
            ce_loss_fn=None,  # CE computed inline
            constraint_loss_fn=constraint_loss_fn,
            terminal_loss_fn=terminal_loss_fn,
            approach_loss_fn=approach_loss_fn,
            optimizer=optimizer, scheduler=scheduler,
            device=device, delta=delta, lam=lam,
        )

        val_metrics = evaluate_model(model, val_loader, device)
        elapsed = time.time() - t0

        history.append({
            'epoch': epoch,
            'total_loss': train_metrics['total_loss'],
            'ce_loss': train_metrics['ce_loss'],
            'constraint_loss': train_metrics['constraint_loss'],
            'terminal_loss': train_metrics['terminal_loss'],
            'approach_loss': train_metrics['approach_loss'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'delta': delta,
            'lambda': lam,
            'time': elapsed,
        })

        phase = ("CE" if epoch < 10
                 else "+Constr" if epoch < 30
                 else "+Term+App" if epoch < 50
                 else "Full")

        print(f"  Epoch {epoch:3d} [{phase:10s}] | "
              f"loss={train_metrics['total_loss']:.4f} "
              f"ce={train_metrics['ce_loss']:.4f} "
              f"constr={train_metrics['constraint_loss']:.4f} "
              f"term={train_metrics['terminal_loss']:.4f} "
              f"app={train_metrics['approach_loss']:.4f} "
              f"(d={delta:.2f} l={lam:.2f}) | "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_acc={val_metrics['accuracy']:.1%} | "
              f"{elapsed:.1f}s")

        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save checkpoint
    ckpt_path = os.path.join(args.results_dir, 'basin_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n  Saved checkpoint: {ckpt_path}")

    # ---- Generation Evaluation ----
    print(f"\n{'=' * 60}")
    print("Generation Evaluation")
    print(f"{'=' * 60}")

    eval_results = evaluate_generation(model, test_ds, device)

    tm = eval_results['tier_metrics']
    print(f"\n  Per-tier accuracy:")
    for tier in sorted(tm.keys()):
        t = tm[tier]
        print(f"    T{tier}: acc={t['accuracy']:.1%} tokens={t['mean_tokens']:.1f} "
              f"(n={t['n']})")
    print(f"  Overall: {eval_results['overall_accuracy']:.1%}")
    print(f"  Mean tokens: {eval_results['mean_tokens']:.1f}")
    print(f"  Stop reasons: {eval_results['stop_reasons']}")

    # ---- Purity Evaluation ----
    print(f"\n{'=' * 60}")
    print("Geodesic Purity Evaluation")
    print(f"{'=' * 60}")

    purity_eval = GeodesicPurity()
    purities = []
    for r in tqdm(eval_results['gen_results'], desc="Purity"):
        if not r.get('generated_ids'):
            continue
        purity_result = evaluate_purity_corrected(
            model, r['generated_ids'], r['expression'], device, purity_eval
        )
        if purity_result['n_constraints'] > 0:
            purities.append(purity_result['purity'])

    mean_purity = float(np.mean(purities)) if purities else 0.0
    print(f"  Mean corrected purity: {mean_purity:.3f} (n={len(purities)})")

    # ---- Figures ----
    print("\n--- Generating Figures ---")
    plot_training_curves(history, args.fig_dir)
    plot_generation_results(eval_results, args.fig_dir)

    # ---- Save Results ----
    all_results = {
        'model': {
            'architecture': 'PNA_SSM_Basin',
            'n_layers': 4, 'd_model': 512, 'd_state': 16,
            'n_params': n_params,
        },
        'training': {
            'epochs_completed': len(history),
            'best_val_loss': best_val_loss,
            'final_val_acc': history[-1]['val_acc'] if history else 0,
            'history': history,
        },
        'generation': {
            'tier_metrics': {str(k): v for k, v in tm.items()},
            'overall_accuracy': eval_results['overall_accuracy'],
            'mean_tokens': eval_results['mean_tokens'],
            'stop_reasons': eval_results['stop_reasons'],
            'n': eval_results['n'],
        },
        'purity': {
            'mean': mean_purity,
            'n': len(purities),
        },
    }

    results_path = os.path.join(args.results_dir, 'phase14f_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    # ---- Success Criteria ----
    print(f"\n{'=' * 60}")
    print("Phase 14F: Summary")
    print(f"{'=' * 60}")

    t2_acc = tm.get(2, {}).get('accuracy', 0)
    overall = eval_results['overall_accuracy']
    eos_pct = eval_results['stop_reasons'].get('eos', 0) / max(eval_results['n'], 1)

    print(f"  T2 Accuracy:    {t2_acc:.1%}")
    print(f"  Overall:        {overall:.1%}")
    print(f"  Mean tokens:    {eval_results['mean_tokens']:.1f}")
    print(f"  EOS stop rate:  {eos_pct:.1%}")
    print(f"  Purity:         {mean_purity:.3f}")


if __name__ == '__main__':
    main()
