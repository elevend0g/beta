"""
Phase 11: Compressible Task Experiment — Variable Depth Reasoning

Tests whether PNA-SSM exploits algebraic shortcuts to produce shorter reasoning
chains. When tasks allow compression (identity ops, cancellations, *0 collapse),
the halt head should fire at the true point of entropy collapse.

Compressibility tiers:
  Tier 0 (incompressible): No shortcuts, all ops are real
  Tier 1 (lightly compressible): 1 identity/cancel op
  Tier 2 (heavily compressible): 2+ cancel ops or *0 collapse

Metrics:
  Token Economy (η): accuracy / mean_reasoning_tokens
  Entropy Bifurcation: H(h_t) trajectories per tier
  Confusion Head: oscillation detection via lagged autocorrelation
"""

import os
import sys
import re
import json
import time
import copy
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from tqdm import tqdm

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import create_model, count_parameters
from losses import ThermodynamicLoss
from train import get_device, get_cosine_schedule, train_one_epoch, evaluate, compute_halt_f1
from ssm_state_entropy_collapse import compute_state_entropy

TIER_NAMES = {0: 'Incompressible', 1: 'Light', 2: 'Heavy'}
TIER_COLORS = {0: '#e74c3c', 1: '#f39c12', 2: '#2ecc71'}


# ============================================================================
# Compressible Arithmetic Dataset
# ============================================================================

class CompressibleArithmeticDataset(Dataset):
    """
    Multi-step arithmetic with controlled compressibility.

    Training data uses compressed reasoning chains — shortcuts already applied.
    The model learns that compressible inputs need fewer reasoning steps,
    so Result: (halt target) naturally moves earlier for compressible chains.
    """

    def __init__(self, num_samples=10000, min_ops=3, max_ops=6,
                 max_seq_len=64, seed=42, tier_weights=None):
        self.max_seq_len = max_seq_len
        self.examples = []
        self.tier_weights = tier_weights or {0: 1/3, 1: 1/3, 2: 1/3}

        rng = np.random.RandomState(seed)

        for _ in range(num_samples):
            # Pick tier according to weights
            tier = rng.choice([0, 1, 2], p=[
                self.tier_weights[0], self.tier_weights[1], self.tier_weights[2]
            ])
            ex = self._generate_example(min_ops, max_ops, tier, rng)
            self.examples.append(ex)

    def _generate_example(self, min_ops, max_ops, tier, rng):
        """Generate a single arithmetic chain with controlled compressibility."""
        num_ops = rng.randint(min_ops, max_ops + 1)

        # Generate raw operation sequence based on tier
        ops, operands = self._generate_ops(num_ops, tier, rng)

        # Start value
        start = int(rng.randint(1, 10))

        # Build expression string: e.g. "3+5-2*0+4"
        expr_parts = [str(start)]
        for op, val in zip(ops, operands):
            expr_parts.append(op)
            expr_parts.append(str(val))
        expression = ''.join(expr_parts)

        # Compute full step-by-step result
        full_steps, final_result = self._compute_full_steps(start, ops, operands)

        # Apply compression — remove redundant steps
        compressed_steps, effective_ops = self._compress_steps(
            start, ops, operands, full_steps
        )

        # Build token sequence with compressed reasoning
        input_str = f"Input:{expression}"
        if compressed_steps:
            reasoning_str = ' '.join(compressed_steps)
            full_text = f"{input_str} {reasoning_str} Result:{final_result}"
        else:
            # Fully compressed — immediate answer
            full_text = f"{input_str} Result:{final_result}"

        # Tokenize
        tokens = [VOCAB['<BOS>']] + tokenize(full_text) + [VOCAB['<HALT>'], VOCAB['<EOS>']]

        # Find Result: position
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
            'answer': final_result,
            'expression': expression,
            'tier': tier,
            'num_ops': num_ops,
            'effective_ops': effective_ops,
            'full_steps': full_steps,
            'compressed_steps': compressed_steps,
        }

    def _generate_ops(self, num_ops, tier, rng):
        """Generate operation sequence with tier-appropriate compressibility."""
        ops = []
        operands = []

        if tier == 0:
            # Incompressible: all ops are real, no cancellations or identities
            for i in range(num_ops):
                op = rng.choice(['+', '-'])
                val = int(rng.randint(1, 10))  # 1-9, never 0
                # Prevent immediate cancellation: if last op was +N, don't do -N
                if i > 0 and ops[-1] != op and operands[-1] == val:
                    val = int((val % 9) + 1)  # shift to avoid cancel
                ops.append(op)
                operands.append(val)

        elif tier == 1:
            # Lightly compressible: inject 1 identity or cancel
            compress_pos = int(rng.randint(0, num_ops))
            compress_type = rng.choice(['identity', 'cancel'])

            for i in range(num_ops):
                if i == compress_pos:
                    if compress_type == 'identity':
                        # +0 or -0 or *1
                        choice = rng.choice(['+0', '-0', '*1'])
                        if choice == '+0':
                            ops.append('+')
                            operands.append(0)
                        elif choice == '-0':
                            ops.append('-')
                            operands.append(0)
                        else:
                            ops.append('*')
                            operands.append(1)
                    else:
                        # Cancel: +N then -N (takes 2 positions)
                        if i + 1 < num_ops:
                            val = int(rng.randint(1, 10))
                            ops.append('+')
                            operands.append(val)
                            # The next iteration will add the cancel
                            # Mark it so we handle in next step
                            ops.append('-')
                            operands.append(val)
                            # Skip the extra op since we added 2
                            # Fill remaining normally
                            for j in range(i + 2, num_ops):
                                op = rng.choice(['+', '-'])
                                v = int(rng.randint(1, 10))
                                ops.append(op)
                                operands.append(v)
                            break
                        else:
                            # Last position, just do identity
                            ops.append('+')
                            operands.append(0)
                else:
                    op = rng.choice(['+', '-'])
                    val = int(rng.randint(1, 10))
                    ops.append(op)
                    operands.append(val)

        elif tier == 2:
            # Heavily compressible: multiple cancels, identities, or *0
            compress_type = rng.choice(['multi_cancel', 'star_zero'])

            if compress_type == 'star_zero':
                # *0 at some point collapses everything before to 0
                star_pos = int(rng.randint(0, max(1, num_ops - 1)))
                for i in range(num_ops):
                    if i == star_pos:
                        ops.append('*')
                        operands.append(0)
                    else:
                        op = rng.choice(['+', '-'])
                        val = int(rng.randint(1, 10))
                        ops.append(op)
                        operands.append(val)
            else:
                # Multiple cancellation pairs: +a-a +b-b ...
                max_pairs = max(2, num_ops // 2)
                n_cancel_pairs = min(num_ops // 2, int(rng.randint(1, max_pairs + 1)))
                remaining = num_ops - n_cancel_pairs * 2
                # Build cancel pairs
                for _ in range(n_cancel_pairs):
                    val = int(rng.randint(1, 10))
                    ops.append('+')
                    operands.append(val)
                    ops.append('-')
                    operands.append(val)
                # Fill remaining with real ops
                for _ in range(remaining):
                    op = rng.choice(['+', '-'])
                    val = int(rng.randint(1, 10))
                    ops.append(op)
                    operands.append(val)

        return ops, operands

    def _compute_full_steps(self, start, ops, operands):
        """Compute all steps, returning list of step strings and final result."""
        current = start
        steps = []
        for op, val in zip(ops, operands):
            if op == '+':
                next_val = current + val
            elif op == '-':
                next_val = current - val
            elif op == '*':
                next_val = current * val
            else:
                next_val = current

            # Clamp to [0, 99]
            next_val = max(0, min(99, next_val))
            steps.append(f"{current}{op}{val}={next_val}")
            current = next_val

        return steps, current

    def _compress_steps(self, start, ops, operands, full_steps):
        """
        Remove redundant steps from the reasoning chain.
        Returns compressed step list and count of effective (non-trivial) ops.
        """
        compressed = []
        current = start
        effective_ops = 0

        i = 0
        while i < len(ops):
            op, val = ops[i], operands[i]

            # Identity: +0, -0, *1 — skip entirely
            if (op in ('+', '-') and val == 0) or (op == '*' and val == 1):
                # Result doesn't change, skip this step
                i += 1
                continue

            # *0: collapse — result is 0, skip prior context
            if op == '*' and val == 0:
                current = 0
                # Don't add a step, just update current
                i += 1
                continue

            # Cancellation: +N followed by -N (or vice versa)
            if i + 1 < len(ops):
                next_op, next_val = ops[i + 1], operands[i + 1]
                if val == next_val and op in ('+', '-') and next_op in ('+', '-') and op != next_op:
                    # They cancel — skip both
                    i += 2
                    continue

            # Real step — compute and add
            if op == '+':
                next_val = current + val
            elif op == '-':
                next_val = current - val
            elif op == '*':
                next_val = current * val

            next_val = max(0, min(99, next_val))
            compressed.append(f"{current}{op}{val}={next_val}")
            current = next_val
            effective_ops += 1
            i += 1

        return compressed, effective_ops

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = list(ex['tokens'])
        mask = list(ex['reasoning_mask'])

        # Pad or truncate
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
# Autoregressive Generator with State Tracking
# ============================================================================

class CompressibleGenerator:
    """
    Autoregressive generation that tracks state entropy at each step.
    Adapted from FreeGenerator for arithmetic (multi-digit results).
    """

    def __init__(self, model, device='cpu', halt_threshold=0.95, max_length=200):
        self.model = model
        self.device = device
        self.halt_threshold = halt_threshold
        self.max_length = max_length
        self.halt_id = VOCAB['<HALT>']
        self.eos_id = VOCAB['<EOS>']
        self.result_id = VOCAB['Result:']

    def generate(self, expression):
        """
        Generate reasoning chain for an arithmetic expression.
        Returns dict with tokens, halt trajectory, state entropy trajectory.
        """
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

                # Clamp to max sequence length
                max_pos = getattr(self.model, 'max_seq_len', 256)
                if hasattr(self.model, 'pos_encoding'):
                    max_pos = self.model.pos_encoding.num_embeddings
                if input_tensor.size(1) > max_pos:
                    input_tensor = input_tensor[:, -max_pos:]

                outputs = self.model(input_tensor)
                logits = outputs["logits"][:, -1, :]

                # Halt confidence
                halt_conf = 0.0
                if outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                # State entropy (SSM only)
                if outputs.get("states_sequence") is not None:
                    last_state = outputs["states_sequence"][:, -1, :]  # [1, d_state]
                    state_vec = last_state.squeeze(0).cpu()
                    state_vectors.append(state_vec)

                    # Compute energy entropy of this state vector
                    energy = last_state ** 2
                    probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
                    ent = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).item()
                    state_entropies.append(ent)
                else:
                    state_entropies.append(0.0)

                # Greedy decoding
                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                # Stop conditions
                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif halt_conf > self.halt_threshold:
                    stop_reason = "halt_confidence"
                    break

        # Parse output
        text = detokenize(generated_ids)
        parsed_answer = self._parse_arithmetic_result(text)
        reasoning_tokens = self._count_reasoning_tokens(generated_ids)

        return {
            'generated_text': text,
            'generated_ids': generated_ids,
            'parsed_answer': parsed_answer,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': len(generated_ids),
            'halt_confidences': halt_confidences,
            'state_entropies': state_entropies,
            'state_vectors': state_vectors,
            'stop_reason': stop_reason,
        }

    def _parse_arithmetic_result(self, text):
        """Extract numeric result from generated text."""
        match = re.search(r'Result:(\d+)', text)
        if match:
            return int(match.group(1))
        # Fallback: last number after '='
        eq_matches = re.findall(r'=(\d+)', text)
        if eq_matches:
            return int(eq_matches[-1])
        return None

    def _count_reasoning_tokens(self, token_ids):
        """Count tokens in the reasoning region."""
        input_tok = VOCAB['Input:']
        result_tok = VOCAB['Result:']
        halt_tok = VOCAB['<HALT>']
        space_tok = VOCAB[' ']

        # Find end of input section
        start = None
        for i, t in enumerate(token_ids):
            if t == input_tok:
                j = i + 1
                # Skip past expression tokens until space
                while j < len(token_ids) and token_ids[j] != space_tok:
                    j += 1
                if j < len(token_ids) and token_ids[j] == space_tok:
                    j += 1
                start = j
                break

        if start is None:
            return 0

        # Find end of reasoning
        end = len(token_ids)
        for i in range(start, len(token_ids)):
            if token_ids[i] in (result_tok, halt_tok):
                end = i
                break

        return max(0, end - start)


# ============================================================================
# Confusion Head — Oscillation Detector
# ============================================================================

def detect_oscillation(state_vectors, lags=(1, 2, 3), threshold=0.95):
    """
    Detect state vector cycling via lagged Pearson autocorrelation.

    Args:
        state_vectors: list of tensors [d_state], one per generation step
        lags: tuple of lag values to check
        threshold: autocorrelation threshold for flagging oscillation

    Returns:
        dict with oscillation flags, per-lag max rho, and classification
    """
    if len(state_vectors) < 3:
        return {
            'has_oscillation': False,
            'max_rho': {},
            'oscillation_positions': [],
            'classification': 'too_short',
        }

    states = torch.stack(state_vectors)  # [L, d_state]
    L = states.size(0)

    oscillation_positions = []
    max_rho_per_lag = {}

    for k in lags:
        if k >= L:
            continue
        rhos = []
        for t in range(k, L):
            s_t = states[t].float()
            s_prev = states[t - k].float()

            # Pearson correlation
            s_t_centered = s_t - s_t.mean()
            s_prev_centered = s_prev - s_prev.mean()
            numer = (s_t_centered * s_prev_centered).sum()
            denom = (s_t_centered.norm() * s_prev_centered.norm()) + 1e-9
            rho = (numer / denom).item()

            rhos.append(rho)
            if rho > threshold:
                oscillation_positions.append({'step': t, 'lag': k, 'rho': rho})

        max_rho_per_lag[k] = max(rhos) if rhos else 0.0

    has_oscillation = len(oscillation_positions) > 0

    return {
        'has_oscillation': has_oscillation,
        'max_rho': max_rho_per_lag,
        'oscillation_positions': oscillation_positions,
        'classification': 'oscillation' if has_oscillation else 'stable',
    }


def classify_convergence(gen_result, oscillation_result):
    """
    Classify each generation into convergence categories.

    Returns one of:
      'true_convergence': halt fired + no oscillation (stable fixed point)
      'false_convergence': halt fired + oscillation detected (state cycling)
      'continued_reasoning': halt didn't fire, state still evolving
    """
    halted = gen_result['stop_reason'] in ('halt_confidence', 'halt_token')

    if halted and not oscillation_result['has_oscillation']:
        return 'true_convergence'
    elif halted and oscillation_result['has_oscillation']:
        return 'false_convergence'
    else:
        return 'continued_reasoning'


# ============================================================================
# Training
# ============================================================================

def train_compressible_model(train_ds, val_ds, device, results_dir,
                             epochs=40, batch_size=32, lr=1e-3, patience=5):
    """Train fresh E_ssm on compressible arithmetic."""
    print("\n" + "=" * 60)
    print("Training E_ssm on Compressible Arithmetic")
    print("=" * 60)

    model = create_model('C', VOCAB_SIZE, device=device)
    n_params = count_parameters(model)
    print(f"  Architecture: PNA_SSM, Params: {n_params:,}")
    print(f"  Loss: L_ce + 0.1*L_halt (halt-only)")
    print(f"  Data: {len(train_ds)} train, {len(val_ds)} val")

    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, None, optimizer, scheduler, device, 'C'
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        val_loss = val_metrics.get('total_loss', val_metrics.get('ce_loss', 0))
        val_acc = val_metrics.get('accuracy', 0)

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics.get('total_loss', 0),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': elapsed,
        })

        print(f"  Epoch {epoch:3d} | train_loss={train_metrics.get('total_loss', 0):.4f} "
              f"| val_loss={val_loss:.4f} | val_acc={val_acc:.1%} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save
    ckpt_path = os.path.join(results_dir, 'compressible_E_ssm_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return model, history


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def run_generation_experiment(model, test_ds, device):
    """Run autoregressive generation on all test examples with state tracking."""
    generator = CompressibleGenerator(model, device=device)

    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        expression = ex['expression']
        ground_truth = ex['answer']
        tier = ex['tier']

        gen_result = generator.generate(expression)

        # Confusion head analysis
        osc_result = detect_oscillation(gen_result['state_vectors'])
        convergence = classify_convergence(gen_result, osc_result)

        is_correct = (gen_result['parsed_answer'] == ground_truth)

        results.append({
            'expression': expression,
            'tier': tier,
            'ground_truth': ground_truth,
            'parsed_answer': gen_result['parsed_answer'],
            'is_correct': is_correct,
            'reasoning_tokens': gen_result['reasoning_tokens'],
            'total_tokens': gen_result['total_tokens'],
            'stop_reason': gen_result['stop_reason'],
            'halt_confidences': gen_result['halt_confidences'],
            'state_entropies': gen_result['state_entropies'],
            'convergence': convergence,
            'has_oscillation': osc_result['has_oscillation'],
            'max_rho': {str(k): v for k, v in osc_result['max_rho'].items()},
            'effective_ops': ex['effective_ops'],
            'num_ops': ex['num_ops'],
            'generated_text': gen_result['generated_text'][:200],
        })

    return results


def compute_tier_metrics(results):
    """Compute per-tier aggregate metrics."""
    tier_results = defaultdict(list)
    for r in results:
        tier_results[r['tier']].append(r)

    metrics = {}
    for tier in sorted(tier_results.keys()):
        items = tier_results[tier]
        n = len(items)
        correct = [r['is_correct'] for r in items]
        reasoning = [r['reasoning_tokens'] for r in items]
        accuracy = np.mean(correct)
        mean_tokens = np.mean(reasoning)

        # Token economy: accuracy / mean_tokens (higher = more efficient)
        eta = accuracy / max(mean_tokens, 1)

        # Halt timing: mean step where halt_confidence > 0.5
        halt_times = []
        for r in items:
            for t, h in enumerate(r['halt_confidences']):
                if h > 0.5:
                    halt_times.append(t)
                    break
            else:
                halt_times.append(len(r['halt_confidences']))

        # Convergence classification
        convergence_counts = defaultdict(int)
        for r in items:
            convergence_counts[r['convergence']] += 1

        # Stop reason distribution
        stop_counts = defaultdict(int)
        for r in items:
            stop_counts[r['stop_reason']] += 1

        # Effective ops distribution
        eff_ops = [r['effective_ops'] for r in items]

        metrics[tier] = {
            'n': n,
            'accuracy': float(accuracy),
            'mean_reasoning_tokens': float(mean_tokens),
            'std_reasoning_tokens': float(np.std(reasoning)),
            'median_reasoning_tokens': float(np.median(reasoning)),
            'token_economy_eta': float(eta),
            'mean_halt_time': float(np.mean(halt_times)),
            'std_halt_time': float(np.std(halt_times)),
            'mean_effective_ops': float(np.mean(eff_ops)),
            'convergence': dict(convergence_counts),
            'stop_reasons': dict(stop_counts),
        }

    return metrics


# ============================================================================
# Visualization
# ============================================================================

def plot_entropy_trajectories(results, fig_dir):
    """
    Fig 15: State entropy trajectories per tier.
    1x3 grid showing entropy bifurcation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for tier_idx, tier in enumerate([0, 1, 2]):
        ax = axes[tier_idx]
        tier_items = [r for r in results if r['tier'] == tier]

        # Plot up to 50 individual trajectories
        max_len = 0
        halt_positions = []
        all_trajs = []
        for r in tier_items[:50]:
            traj = r['state_entropies']
            if len(traj) > 2:
                # Normalize to [0, 1]
                traj_arr = np.array(traj)
                if traj_arr.max() > 0:
                    traj_norm = traj_arr / traj_arr.max()
                else:
                    traj_norm = traj_arr
                ax.plot(traj_norm, color=TIER_COLORS[tier], alpha=0.15, linewidth=0.5)
                all_trajs.append(traj_norm)
                max_len = max(max_len, len(traj_norm))

            # Halt position
            for t, h in enumerate(r['halt_confidences']):
                if h > 0.5:
                    halt_positions.append(t)
                    break

        # Mean trajectory
        if all_trajs:
            padded = np.zeros((len(all_trajs), max_len))
            for i, tr in enumerate(all_trajs):
                padded[i, :len(tr)] = tr
                padded[i, len(tr):] = tr[-1] if len(tr) > 0 else 0
            mean_traj = padded.mean(axis=0)
            std_traj = padded.std(axis=0)
            x = np.arange(max_len)
            ax.plot(x, mean_traj, color=TIER_COLORS[tier], linewidth=2.5, label='Mean')
            ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj,
                            color=TIER_COLORS[tier], alpha=0.2)

        # Mean halt position
        if halt_positions:
            mean_halt = np.mean(halt_positions)
            ax.axvline(mean_halt, color='black', linestyle='--', alpha=0.6,
                        label=f'Mean halt: {mean_halt:.1f}')

        ax.set_title(f"Tier {tier} ({TIER_NAMES[tier]})", fontsize=13)
        ax.set_xlabel("Generation Step")
        if tier_idx == 0:
            ax.set_ylabel("Normalized State Entropy")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.1)

    fig.suptitle("State Entropy Trajectories by Compressibility Tier",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig15_compressible_entropy_trajectories.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_token_economy(tier_metrics, fig_dir):
    """
    Fig 16: Token economy per tier.
    Left: box plot of reasoning tokens. Right: bar chart of η.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    tiers = sorted(tier_metrics.keys())
    labels = [f"T{t} ({TIER_NAMES[t]})" for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    # Left: reasoning token counts
    means = [tier_metrics[t]['mean_reasoning_tokens'] for t in tiers]
    stds = [tier_metrics[t]['std_reasoning_tokens'] for t in tiers]
    medians = [tier_metrics[t]['median_reasoning_tokens'] for t in tiers]

    bars = ax1.bar(labels, means, color=colors, alpha=0.8, edgecolor='black')
    ax1.errorbar(labels, means, yerr=stds, fmt='none', color='black', capsize=5)
    for i, (m, med) in enumerate(zip(means, medians)):
        ax1.annotate(f'μ={m:.1f}\nmed={med:.0f}', (i, m + stds[i] + 0.5),
                     ha='center', fontsize=9)
    ax1.set_ylabel("Reasoning Tokens")
    ax1.set_title("Reasoning Chain Length by Tier")

    # Right: token economy η
    etas = [tier_metrics[t]['token_economy_eta'] for t in tiers]
    accs = [tier_metrics[t]['accuracy'] for t in tiers]
    bars2 = ax2.bar(labels, etas, color=colors, alpha=0.8, edgecolor='black')
    for i, (eta, acc) in enumerate(zip(etas, accs)):
        ax2.annotate(f'η={eta:.3f}\nacc={acc:.1%}', (i, eta + 0.002),
                     ha='center', fontsize=9)
    ax2.set_ylabel("Token Economy (η = acc / tokens)")
    ax2.set_title("Token Economy by Tier")

    fig.suptitle("Variable Depth Reasoning: Token Efficiency",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig16_token_economy.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_head(results, tier_metrics, fig_dir):
    """
    Fig 17: Confusion head analysis.
    Left: example trajectories (stable vs oscillating).
    Right: stacked bar of convergence classification per tier.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: example state entropy trajectories — one stable, one oscillating (if found)
    stable_ex = None
    osc_ex = None
    for r in results:
        if r['convergence'] == 'true_convergence' and stable_ex is None:
            if len(r['state_entropies']) > 5:
                stable_ex = r
        if r['convergence'] == 'false_convergence' and osc_ex is None:
            if len(r['state_entropies']) > 5:
                osc_ex = r
        if stable_ex and osc_ex:
            break

    if stable_ex:
        ax1.plot(stable_ex['state_entropies'], color='#2ecc71', linewidth=2,
                 label=f"Stable: {stable_ex['expression'][:15]}...")
    if osc_ex:
        ax1.plot(osc_ex['state_entropies'], color='#e74c3c', linewidth=2,
                 linestyle='--', label=f"Oscillating: {osc_ex['expression'][:15]}...")
    if not stable_ex and not osc_ex:
        ax1.text(0.5, 0.5, "No examples found", ha='center', va='center',
                 transform=ax1.transAxes)

    ax1.set_xlabel("Generation Step")
    ax1.set_ylabel("State Entropy (bits)")
    ax1.set_title("Convergence Examples")
    ax1.legend(fontsize=9)

    # Right: stacked bar of convergence per tier
    tiers = sorted(tier_metrics.keys())
    labels = [f"T{t} ({TIER_NAMES[t]})" for t in tiers]

    true_conv = []
    false_conv = []
    cont_reasoning = []

    for t in tiers:
        conv = tier_metrics[t].get('convergence', {})
        total = tier_metrics[t]['n']
        true_conv.append(conv.get('true_convergence', 0) / max(total, 1) * 100)
        false_conv.append(conv.get('false_convergence', 0) / max(total, 1) * 100)
        cont_reasoning.append(conv.get('continued_reasoning', 0) / max(total, 1) * 100)

    x = np.arange(len(tiers))
    width = 0.5
    ax2.bar(x, true_conv, width, color='#2ecc71', label='True Convergence')
    ax2.bar(x, false_conv, width, bottom=true_conv, color='#e74c3c',
            label='False Convergence')
    ax2.bar(x, cont_reasoning, width,
            bottom=[a + b for a, b in zip(true_conv, false_conv)],
            color='#95a5a6', label='Continued Reasoning')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("% of Examples")
    ax2.set_title("Convergence Classification by Tier")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 105)

    fig.suptitle("Confusion Head: Oscillation Detection",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig17_confusion_head.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Phase 11b: Convergence vs Halt Analysis (Basin vs Fixed Point)
# ============================================================================

def analyze_convergence_vs_halt(results):
    """
    Compare halt accuracy between true convergence (stable fixed point)
    and state cycling (oscillation within basin) cases.

    If USS is robust: accuracy ~equal in both → halt detects basin entry
    If USS is spurious: accuracy higher in cycling → halt fires on cycling pattern
    """
    convergence_results = [r for r in results if r['convergence'] == 'true_convergence']
    cycling_results = [r for r in results if r['convergence'] == 'false_convergence']

    def compute_group_metrics(items, label):
        if not items:
            return {'label': label, 'n': 0, 'accuracy': 0, 'mean_tokens': 0,
                    'mean_entropy_at_halt': 0, 'mean_halt_conf_at_stop': 0,
                    'mean_final_entropy': 0, 'entropy_variance': 0}

        acc = np.mean([r['is_correct'] for r in items])
        tokens = np.mean([r['reasoning_tokens'] for r in items])

        # Entropy at halt: what was state entropy when halt fired?
        entropy_at_halt = []
        halt_conf_at_stop = []
        final_entropies = []
        entropy_variances = []

        for r in items:
            ents = r['state_entropies']
            halts = r['halt_confidences']

            if ents:
                final_entropies.append(ents[-1])
                if len(ents) > 2:
                    entropy_variances.append(float(np.var(ents[-5:])))  # variance of last 5 steps

            # Find halt step (first step where halt_conf > 0.5)
            for t, h in enumerate(halts):
                if h > 0.5 and t < len(ents):
                    entropy_at_halt.append(ents[t])
                    halt_conf_at_stop.append(h)
                    break

        return {
            'label': label,
            'n': len(items),
            'accuracy': float(acc),
            'mean_tokens': float(tokens),
            'mean_entropy_at_halt': float(np.mean(entropy_at_halt)) if entropy_at_halt else 0,
            'mean_halt_conf_at_stop': float(np.mean(halt_conf_at_stop)) if halt_conf_at_stop else 0,
            'mean_final_entropy': float(np.mean(final_entropies)) if final_entropies else 0,
            'entropy_variance_last5': float(np.mean(entropy_variances)) if entropy_variances else 0,
        }

    conv_metrics = compute_group_metrics(convergence_results, 'True Convergence')
    cyc_metrics = compute_group_metrics(cycling_results, 'State Cycling')

    # Compute gap
    gap = cyc_metrics['accuracy'] - conv_metrics['accuracy']

    # Determine interpretation
    if conv_metrics['n'] < 5:
        interpretation = 'insufficient_convergence_samples'
    elif abs(gap) < 0.10:
        interpretation = 'USS_robust_basin_detection'
    elif gap > 0.10:
        interpretation = 'USS_spurious_cycling_dependent'
    else:
        interpretation = 'USS_robust_convergence_advantage'

    # Per-tier breakdown
    tier_breakdown = {}
    for tier in [0, 1, 2]:
        tier_conv = [r for r in convergence_results if r['tier'] == tier]
        tier_cyc = [r for r in cycling_results if r['tier'] == tier]
        tier_breakdown[tier] = {
            'convergence': compute_group_metrics(tier_conv, f'T{tier} Converged'),
            'cycling': compute_group_metrics(tier_cyc, f'T{tier} Cycling'),
        }

    return {
        'convergence': conv_metrics,
        'cycling': cyc_metrics,
        'accuracy_gap': float(gap),
        'interpretation': interpretation,
        'tier_breakdown': tier_breakdown,
    }


def plot_basin_analysis(results, basin_metrics, fig_dir):
    """
    Fig 18: Basin vs Fixed Point analysis.
    3-panel figure:
      Left: Accuracy comparison (convergent vs cycling, per tier)
      Center: Entropy at halt (convergent vs cycling)
      Right: Example basin trajectories — entropy oscillation within answer-determined region
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left panel: Accuracy by convergence type per tier ---
    tiers = [0, 1, 2]
    tier_labels = [f"T{t}" for t in tiers]
    x = np.arange(len(tiers))
    width = 0.35

    conv_accs = []
    cyc_accs = []
    conv_ns = []
    cyc_ns = []

    for t in tiers:
        tb = basin_metrics['tier_breakdown'][t]
        conv_accs.append(tb['convergence']['accuracy'] * 100)
        cyc_accs.append(tb['cycling']['accuracy'] * 100)
        conv_ns.append(tb['convergence']['n'])
        cyc_ns.append(tb['cycling']['n'])

    bars1 = ax1.bar(x - width/2, conv_accs, width, color='#2ecc71',
                     alpha=0.8, edgecolor='black', label='True Convergence')
    bars2 = ax1.bar(x + width/2, cyc_accs, width, color='#e74c3c',
                     alpha=0.8, edgecolor='black', label='State Cycling')

    for i, (ca, cn, ya, yn) in enumerate(zip(conv_accs, conv_ns, cyc_accs, cyc_ns)):
        if cn > 0:
            ax1.annotate(f'{ca:.0f}%\nn={cn}', (i - width/2, ca + 1),
                         ha='center', fontsize=8)
        if yn > 0:
            ax1.annotate(f'{ya:.0f}%\nn={yn}', (i + width/2, ya + 1),
                         ha='center', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(tier_labels)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy: Convergent vs Cycling")
    ax1.legend(fontsize=9)

    # Overall gap annotation
    gap = basin_metrics['accuracy_gap']
    interp = basin_metrics['interpretation']
    interp_short = interp.replace('USS_', '').replace('_', ' ').title()
    ax1.annotate(f'Gap: {gap:+.1%}\n{interp_short}',
                 xy=(0.5, 0.02), xycoords='axes fraction',
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Center panel: Entropy at halt ---
    conv_ent = basin_metrics['convergence']['mean_entropy_at_halt']
    cyc_ent = basin_metrics['cycling']['mean_entropy_at_halt']
    conv_var = basin_metrics['convergence']['entropy_variance_last5']
    cyc_var = basin_metrics['cycling']['entropy_variance_last5']

    categories = ['Entropy\nat Halt', 'Entropy\nVariance\n(last 5)']
    conv_vals = [conv_ent, conv_var]
    cyc_vals = [cyc_ent, cyc_var]

    x2 = np.arange(len(categories))
    ax2.bar(x2 - width/2, conv_vals, width, color='#2ecc71', alpha=0.8,
            edgecolor='black', label='True Convergence')
    ax2.bar(x2 + width/2, cyc_vals, width, color='#e74c3c', alpha=0.8,
            edgecolor='black', label='State Cycling')

    for i, (cv, yv) in enumerate(zip(conv_vals, cyc_vals)):
        ax2.annotate(f'{cv:.2f}', (i - width/2, cv + 0.01), ha='center', fontsize=9)
        ax2.annotate(f'{yv:.2f}', (i + width/2, yv + 0.01), ha='center', fontsize=9)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("State Entropy at Halt")
    ax2.legend(fontsize=9)

    # --- Right panel: Basin trajectory examples ---
    # Show 3 examples: one true convergence, one cycling (correct), one cycling (wrong)
    conv_correct = None
    cyc_correct = None
    cyc_wrong = None

    for r in results:
        if r['convergence'] == 'true_convergence' and r['is_correct'] and conv_correct is None:
            if len(r['state_entropies']) > 3:
                conv_correct = r
        if r['convergence'] == 'false_convergence' and r['is_correct'] and cyc_correct is None:
            if len(r['state_entropies']) > 5:
                cyc_correct = r
        if r['convergence'] == 'false_convergence' and not r['is_correct'] and cyc_wrong is None:
            if len(r['state_entropies']) > 5:
                cyc_wrong = r
        if conv_correct and cyc_correct and cyc_wrong:
            break

    if conv_correct:
        ax3.plot(conv_correct['state_entropies'], color='#2ecc71', linewidth=2,
                 label=f"Converged (correct)")
    if cyc_correct:
        ax3.plot(cyc_correct['state_entropies'], color='#3498db', linewidth=2,
                 linestyle='--', label=f"Cycling (correct)")
    if cyc_wrong:
        ax3.plot(cyc_wrong['state_entropies'], color='#e74c3c', linewidth=1.5,
                 linestyle=':', label=f"Cycling (wrong)")

    ax3.set_xlabel("Generation Step")
    ax3.set_ylabel("State Entropy (bits)")
    ax3.set_title("Basin Trajectories")
    ax3.legend(fontsize=9)

    # Add annotation explaining basin concept
    ax3.annotate('Basin = answer determined,\nstate still oscillating',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=8, fontstyle='italic',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle("USS Robustness: Basin Entry vs Fixed-Point Convergence",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig18_basin_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 11: Compressible Task Experiment")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--val-n', type=int, default=1000)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--device', default=None)
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--fig-dir', default='figures')
    parser.add_argument('--skip-training', action='store_true',
                        help='Load existing checkpoint, skip training')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Create datasets ----
    print("\n--- Creating Compressible Arithmetic Datasets ---")
    train_ds = CompressibleArithmeticDataset(
        num_samples=args.train_n, min_ops=3, max_ops=6,
        max_seq_len=64, seed=42
    )
    val_ds = CompressibleArithmeticDataset(
        num_samples=args.val_n, min_ops=3, max_ops=6,
        max_seq_len=64, seed=123
    )
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456
    )

    # Print tier distribution
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        tier_counts = defaultdict(int)
        for ex in ds.examples:
            tier_counts[ex['tier']] += 1
        dist = {f"T{k}": v for k, v in sorted(tier_counts.items())}
        print(f"  {name}: {len(ds)} examples, tiers={dist}")

    # Print examples per tier
    print("\n--- Example per Tier ---")
    for tier in [0, 1, 2]:
        for ex in train_ds.examples:
            if ex['tier'] == tier:
                print(f"  Tier {tier} ({TIER_NAMES[tier]}): {ex['text']}")
                print(f"    Expression: {ex['expression']}, Effective ops: {ex['effective_ops']}")
                print(f"    Full steps: {ex['full_steps']}")
                print(f"    Compressed: {ex['compressed_steps']}")
                break

    # ---- Train ----
    ckpt_path = os.path.join(args.results_dir, 'compressible_E_ssm_model.pt')

    if args.skip_training and os.path.exists(ckpt_path):
        print(f"\n--- Loading existing checkpoint: {ckpt_path} ---")
        model = create_model('C', VOCAB_SIZE, device=device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        history = []
    else:
        model, history = train_compressible_model(
            train_ds, val_ds, device, args.results_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience
        )

    # ---- Teacher-forced evaluation ----
    print("\n--- Teacher-Forced Evaluation ---")
    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    halt_f1 = compute_halt_f1(model, test_loader, device)
    print(f"  Test accuracy: {test_metrics.get('accuracy', 0):.1%}")
    print(f"  Halt F1: {halt_f1:.1%}")

    # ---- Autoregressive Generation ----
    print("\n--- Autoregressive Generation with State Tracking ---")
    gen_results = run_generation_experiment(model, test_ds, device)

    # ---- Compute Metrics ----
    print("\n--- Per-Tier Metrics ---")
    tier_metrics = compute_tier_metrics(gen_results)

    for tier in sorted(tier_metrics.keys()):
        m = tier_metrics[tier]
        print(f"\n  Tier {tier} ({TIER_NAMES[tier]}):")
        print(f"    N={m['n']}, Accuracy={m['accuracy']:.1%}")
        print(f"    Reasoning tokens: {m['mean_reasoning_tokens']:.1f} ± {m['std_reasoning_tokens']:.1f}")
        print(f"    Token economy η: {m['token_economy_eta']:.4f}")
        print(f"    Mean halt time: {m['mean_halt_time']:.1f} ± {m['std_halt_time']:.1f}")
        print(f"    Effective ops: {m['mean_effective_ops']:.1f}")
        print(f"    Convergence: {dict(m['convergence'])}")
        print(f"    Stop reasons: {dict(m['stop_reasons'])}")

    # ---- Convergence vs Halt (Basin Analysis) ----
    print("\n--- Convergence vs Halt Analysis (Basin vs Fixed Point) ---")
    basin_metrics = analyze_convergence_vs_halt(gen_results)

    conv = basin_metrics['convergence']
    cyc = basin_metrics['cycling']
    print(f"\n  True Convergence (n={conv['n']}):")
    print(f"    Accuracy: {conv['accuracy']:.1%}")
    print(f"    Mean tokens: {conv['mean_tokens']:.1f}")
    print(f"    Entropy at halt: {conv['mean_entropy_at_halt']:.3f} bits")
    print(f"    Entropy variance (last 5): {conv['entropy_variance_last5']:.4f}")

    print(f"\n  State Cycling (n={cyc['n']}):")
    print(f"    Accuracy: {cyc['accuracy']:.1%}")
    print(f"    Mean tokens: {cyc['mean_tokens']:.1f}")
    print(f"    Entropy at halt: {cyc['mean_entropy_at_halt']:.3f} bits")
    print(f"    Entropy variance (last 5): {cyc['entropy_variance_last5']:.4f}")

    print(f"\n  Accuracy gap: {basin_metrics['accuracy_gap']:+.1%}")
    print(f"  Interpretation: {basin_metrics['interpretation']}")

    # Per-tier breakdown
    print("\n  Per-tier breakdown:")
    for tier in [0, 1, 2]:
        tb = basin_metrics['tier_breakdown'][tier]
        tc = tb['convergence']
        ty = tb['cycling']
        print(f"    Tier {tier}: converged={tc['n']} ({tc['accuracy']:.0%}), "
              f"cycling={ty['n']} ({ty['accuracy']:.0%})")

    # ---- Save Results ----
    output = {
        'teacher_forced': {
            'accuracy': test_metrics.get('accuracy', 0),
            'halt_f1': halt_f1,
        },
        'tier_metrics': {str(k): v for k, v in tier_metrics.items()},
        'basin_analysis': basin_metrics,
        'training_history': history,
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'train_n': args.train_n,
            'val_n': args.val_n,
            'test_n': args.test_n,
        },
    }

    results_path = os.path.join(args.results_dir, 'compressible_task_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved results: {results_path}")

    # ---- Figures ----
    print("\n--- Generating Figures ---")
    plot_entropy_trajectories(gen_results, args.fig_dir)
    plot_token_economy(tier_metrics, args.fig_dir)
    plot_confusion_head(gen_results, tier_metrics, args.fig_dir)
    plot_basin_analysis(gen_results, basin_metrics, args.fig_dir)

    print("\n" + "=" * 60)
    print("Phase 11: Compressible Task Experiment COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
