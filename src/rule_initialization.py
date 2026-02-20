"""
Phase 12: Rule-Initialized Models (RIM) Experiment

Adds algebraic rule constraints to the SSM state space to convert false convergence
(limit cycles) into true convergence (fixed points). Builds on Phase 11's compressible
task experiment.

Rules operate on state dynamics — they check whether the state *behaves correctly*
in response to algebraic patterns:
  - Identity ops (+0, -0, *1): state should not change
  - Cancellation pairs (+N-N): state should return to pre-cancel value
  - Multiplicative collapse (*0): state entropy should drop sharply

The constraint loss is annealed: 0 for first 10 epochs, ramps to 1.0 by epoch 30.
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
from tqdm import tqdm

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, count_parameters
from losses import ThermodynamicLoss
from train import get_device, get_cosine_schedule, evaluate, compute_halt_f1
from compressible_task import (
    CompressibleArithmeticDataset, CompressibleGenerator,
    detect_oscillation, classify_convergence, analyze_convergence_vs_halt,
    compute_tier_metrics, TIER_NAMES, TIER_COLORS,
)
from ssm_state_entropy_collapse import compute_state_entropy

# ============================================================================
# Constants
# ============================================================================

OP_PAD = 0
OP_REAL = 1
OP_IDENTITY = 2
OP_CANCEL_START = 3
OP_CANCEL_END = 4
OP_STAR_ZERO = 5

OP_NAMES = {
    OP_PAD: 'pad', OP_REAL: 'real', OP_IDENTITY: 'identity',
    OP_CANCEL_START: 'cancel_start', OP_CANCEL_END: 'cancel_end',
    OP_STAR_ZERO: 'star_zero',
}

MAX_OPS = 8  # max operations per expression


# ============================================================================
# Operation Classification
# ============================================================================

def classify_ops(expression):
    """
    Parse expression and classify each operation.

    Args:
        expression: str like "3+0+5-2" or "7*0+4"

    Returns:
        list of (op_char, operand, op_type) tuples
    """
    # Parse start value
    i = 0
    while i < len(expression) and expression[i].isdigit():
        i += 1

    ops = []
    while i < len(expression):
        op_char = expression[i]
        i += 1
        operand_str = ''
        while i < len(expression) and expression[i].isdigit():
            operand_str += expression[i]
            i += 1
        operand = int(operand_str) if operand_str else 0
        ops.append((op_char, operand))

    # Classify each op
    classified = []
    j = 0
    while j < len(ops):
        op_char, operand = ops[j]

        # Identity: +0, -0, *1
        if (op_char in ('+', '-') and operand == 0) or (op_char == '*' and operand == 1):
            classified.append((op_char, operand, OP_IDENTITY))
            j += 1
            continue

        # Star zero: *0
        if op_char == '*' and operand == 0:
            classified.append((op_char, operand, OP_STAR_ZERO))
            j += 1
            continue

        # Check for cancellation: +N-N or -N+N
        if j + 1 < len(ops):
            next_op, next_val = ops[j + 1]
            if (operand == next_val and op_char in ('+', '-') and
                    next_op in ('+', '-') and op_char != next_op):
                classified.append((op_char, operand, OP_CANCEL_START))
                classified.append((next_op, next_val, OP_CANCEL_END))
                j += 2
                continue

        # Real op
        classified.append((op_char, operand, OP_REAL))
        j += 1

    return classified


def find_op_token_positions(tokens, expression):
    """
    Find token positions of each operation in the expression.

    Returns list of (op_type, before_pos, after_pos) where:
      before_pos = token index of state BEFORE the op
      after_pos  = token index of state AFTER the op's operand
    """
    # Find expression start: position after Input: token
    input_tok = VOCAB['Input:']
    expr_start = None
    for i, t in enumerate(tokens):
        if t == input_tok:
            expr_start = i + 1
            break
    if expr_start is None:
        return []

    # Parse expression to get character offsets
    ops = classify_ops(expression)

    # Walk through expression chars to find token positions
    # Expression chars map 1:1 to tokens (single-digit operands)
    char_idx = 0
    # Skip start value digits
    while char_idx < len(expression) and expression[char_idx].isdigit():
        char_idx += 1

    results = []
    op_idx = 0
    while char_idx < len(expression) and op_idx < len(ops):
        op_char, operand, op_type = ops[op_idx]

        # Current position: the operator token
        op_token_pos = expr_start + char_idx
        char_idx += 1  # skip operator char

        # Operand position(s)
        operand_str = str(operand)
        operand_end_pos = expr_start + char_idx + len(operand_str) - 1
        char_idx += len(operand_str)

        # before_pos: token just before the operator
        before_pos = op_token_pos - 1
        # after_pos: last token of the operand
        after_pos = operand_end_pos

        results.append((op_type, before_pos, after_pos))
        op_idx += 1

    return results


# ============================================================================
# RIM Dataset Wrapper
# ============================================================================

class RIMDatasetWrapper(Dataset):
    """
    Wraps CompressibleArithmeticDataset to add operation metadata.
    Returns additional fields: tier, op_types, op_before_pos, op_after_pos.
    """

    def __init__(self, base_dataset):
        self.base = base_dataset
        self.max_seq_len = base_dataset.max_seq_len
        self._precompute_metadata()

    def _precompute_metadata(self):
        """Precompute op positions and types for all examples."""
        self.metadata = []
        for ex in self.base.examples:
            op_info = find_op_token_positions(ex['tokens'], ex['expression'])

            op_types = []
            before_positions = []
            after_positions = []
            for op_type, before_pos, after_pos in op_info:
                op_types.append(op_type)
                before_positions.append(min(before_pos, self.max_seq_len - 2))
                after_positions.append(min(after_pos, self.max_seq_len - 2))

            # Pad to MAX_OPS
            n_ops = len(op_types)
            while len(op_types) < MAX_OPS:
                op_types.append(OP_PAD)
                before_positions.append(0)
                after_positions.append(0)

            self.metadata.append({
                'tier': ex['tier'],
                'n_ops': n_ops,
                'op_types': op_types[:MAX_OPS],
                'before_pos': before_positions[:MAX_OPS],
                'after_pos': after_positions[:MAX_OPS],
                'expression': ex['expression'],
            })

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        meta = self.metadata[idx]

        item['tier'] = torch.tensor(meta['tier'], dtype=torch.long)
        item['n_ops'] = torch.tensor(meta['n_ops'], dtype=torch.long)
        item['op_types'] = torch.tensor(meta['op_types'], dtype=torch.long)
        item['op_before_pos'] = torch.tensor(meta['before_pos'], dtype=torch.long)
        item['op_after_pos'] = torch.tensor(meta['after_pos'], dtype=torch.long)

        return item


# ============================================================================
# PNA_SSM_RIM Model
# ============================================================================

class PNA_SSM_RIM(PNA_SSM):
    """
    PNA_SSM with learned state correction for rule satisfaction.

    Adds a small residual corrector that nudges SSM states toward
    rule-satisfying regions. The corrected states are fed to the halt head.
    """

    def __init__(self, vocab_size, d_model=512, n_layers=6, d_state=16,
                 max_seq_len=256, correction_scale=0.1):
        super().__init__(vocab_size, d_model, n_layers, d_state, max_seq_len)

        self.correction_scale = correction_scale

        # Learned state correction: small residual network
        self.state_corrector = nn.Sequential(
            nn.Linear(d_state, d_state * 2),
            nn.ReLU(),
            nn.Linear(d_state * 2, d_state),
            nn.Tanh()  # bounded correction
        )

        # Value decoder probe: extracts numerical belief from state
        # 20 bins for values 0-19 (covers most arithmetic results)
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

        # Apply state correction (soft projection)
        correction = self.state_corrector(all_states)
        corrected_states = all_states + self.correction_scale * correction

        # Halt confidence from corrected states
        halt_confidence = self.halt_head(corrected_states)

        # Value decoder
        value_logits = self.value_decoder(corrected_states)

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
            "states_sequence": corrected_states,
            "final_state": corrected_states[:, -1, :],
            "raw_states": all_states,
            "value_logits": value_logits,
        }


# ============================================================================
# Rule Constraint Loss
# ============================================================================

class RuleConstraintLoss(nn.Module):
    """
    Soft constraint loss for algebraic rules.

    Operates on raw SSM states and penalizes violations:
      - Identity ops: ||s_after - s_before||² should be small
      - Cancel pairs: ||s_after_pair - s_before_pair||² should be small
      - Star zero: entropy at collapse point should be low
    """

    def __init__(self, identity_weight=1.0, cancel_weight=1.0, collapse_weight=0.5):
        super().__init__()
        self.identity_weight = identity_weight
        self.cancel_weight = cancel_weight
        self.collapse_weight = collapse_weight

    def forward(self, states, op_types, op_before_pos, op_after_pos, n_ops):
        """
        Args:
            states: [B, L, d_state] — raw SSM states
            op_types: [B, MAX_OPS] — operation type codes
            op_before_pos: [B, MAX_OPS] — token position before each op
            op_after_pos: [B, MAX_OPS] — token position after each op's operand
            n_ops: [B] — number of valid ops per example

        Returns:
            dict with total constraint loss and per-rule breakdown
        """
        B, L, D = states.shape
        device = states.device

        identity_loss = torch.tensor(0.0, device=device)
        cancel_loss = torch.tensor(0.0, device=device)
        collapse_loss = torch.tensor(0.0, device=device)
        n_identity = 0
        n_cancel = 0
        n_collapse = 0

        for b in range(B):
            n = n_ops[b].item()
            i = 0
            while i < n:
                ot = op_types[b, i].item()
                bp = op_before_pos[b, i].item()
                ap = op_after_pos[b, i].item()

                # Bounds check
                if bp >= L or ap >= L or bp < 0 or ap < 0:
                    i += 1
                    continue

                s_before = states[b, bp]
                s_after = states[b, ap]

                if ot == OP_IDENTITY:
                    # State should not change
                    identity_loss = identity_loss + F.mse_loss(s_after, s_before)
                    n_identity += 1

                elif ot == OP_CANCEL_START:
                    # Find the cancel_end partner
                    if i + 1 < n and op_types[b, i + 1].item() == OP_CANCEL_END:
                        ap_end = op_after_pos[b, i + 1].item()
                        if ap_end < L:
                            s_after_pair = states[b, ap_end]
                            cancel_loss = cancel_loss + F.mse_loss(s_after_pair, s_before)
                            n_cancel += 1
                        i += 2  # skip the cancel_end
                        continue

                elif ot == OP_STAR_ZERO:
                    # State entropy should drop after *0
                    energy = s_after ** 2
                    probs = energy / (energy.sum() + 1e-9)
                    ent_after = -(probs * torch.log2(probs + 1e-9)).sum()

                    energy_b = s_before ** 2
                    probs_b = energy_b / (energy_b.sum() + 1e-9)
                    ent_before = -(probs_b * torch.log2(probs_b + 1e-9)).sum()

                    # Penalize if entropy doesn't drop
                    collapse_loss = collapse_loss + F.relu(ent_after - ent_before + 0.5)
                    n_collapse += 1

                i += 1

        # Normalize
        if n_identity > 0:
            identity_loss = identity_loss / n_identity
        if n_cancel > 0:
            cancel_loss = cancel_loss / n_cancel
        if n_collapse > 0:
            collapse_loss = collapse_loss / n_collapse

        total = (self.identity_weight * identity_loss +
                 self.cancel_weight * cancel_loss +
                 self.collapse_weight * collapse_loss)

        return {
            'total': total,
            'identity_loss': identity_loss.item(),
            'cancel_loss': cancel_loss.item(),
            'collapse_loss': collapse_loss.item(),
            'n_identity': n_identity,
            'n_cancel': n_cancel,
            'n_collapse': n_collapse,
        }


# ============================================================================
# Geodesic Purity Evaluator
# ============================================================================

class GeodesicPurity:
    """
    Measures what fraction of a generation trajectory satisfies algebraic rules.

    For each step in the input processing region, checks whether the state
    transition respects the rule that should apply at that step.
    """

    def __init__(self, identity_threshold=0.1, cancel_threshold=0.15):
        self.identity_threshold = identity_threshold
        self.cancel_threshold = cancel_threshold

    def evaluate_trajectory(self, state_vectors, expression, tokens):
        """
        Evaluate geodesic purity of a single generation.

        Args:
            state_vectors: list of [d_state] tensors, one per generation step
            expression: str, the input expression
            tokens: list of token ids (full generated sequence)

        Returns:
            dict with purity score and violation details
        """
        if len(state_vectors) < 3:
            return {'purity': 1.0, 'violations': [], 'n_constraints': 0, 'n_satisfied': 0}

        states = torch.stack(state_vectors)  # [L, d_state]
        op_info = find_op_token_positions(tokens, expression)

        n_constraints = 0
        n_satisfied = 0
        violations = []

        i = 0
        while i < len(op_info):
            op_type, before_pos, after_pos = op_info[i]

            if before_pos >= len(states) or after_pos >= len(states):
                i += 1
                continue

            s_before = states[before_pos]
            s_after = states[after_pos]

            if op_type == OP_IDENTITY:
                n_constraints += 1
                delta = (s_after - s_before).norm().item()
                if delta < self.identity_threshold:
                    n_satisfied += 1
                else:
                    violations.append(('identity', before_pos, after_pos, delta))

            elif op_type == OP_CANCEL_START:
                if i + 1 < len(op_info) and op_info[i + 1][0] == OP_CANCEL_END:
                    _, _, after_pair = op_info[i + 1]
                    if after_pair < len(states):
                        n_constraints += 1
                        s_after_pair = states[after_pair]
                        delta = (s_after_pair - s_before).norm().item()
                        if delta < self.cancel_threshold:
                            n_satisfied += 1
                        else:
                            violations.append(('cancel', before_pos, after_pair, delta))
                    i += 2
                    continue

            elif op_type == OP_STAR_ZERO:
                n_constraints += 1
                # Check entropy drop
                energy_a = s_after ** 2
                probs_a = energy_a / (energy_a.sum() + 1e-9)
                ent_a = -(probs_a * torch.log2(probs_a + 1e-9)).sum().item()

                energy_b = s_before ** 2
                probs_b = energy_b / (energy_b.sum() + 1e-9)
                ent_b = -(probs_b * torch.log2(probs_b + 1e-9)).sum().item()

                if ent_a < ent_b:
                    n_satisfied += 1
                else:
                    violations.append(('star_zero', before_pos, after_pos, ent_a - ent_b))

            i += 1

        purity = n_satisfied / max(1, n_constraints)
        return {
            'purity': purity,
            'n_constraints': n_constraints,
            'n_satisfied': n_satisfied,
            'violations': violations,
        }


# ============================================================================
# Training
# ============================================================================

def get_constraint_weight(epoch, warmup_epochs=10, ramp_epochs=20):
    """Annealing schedule for constraint loss weight δ."""
    if epoch < warmup_epochs:
        return 0.0
    elif epoch < warmup_epochs + ramp_epochs:
        progress = (epoch - warmup_epochs) / ramp_epochs
        return progress
    else:
        return 1.0


def train_rim_one_epoch(model, dataloader, loss_fn, constraint_loss_fn,
                        optimizer, scheduler, device, constraint_weight):
    """Train RIM model for one epoch."""
    model.train()
    metrics_accum = defaultdict(float)
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

        # Standard thermodynamic loss
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=targets,
            halt_confidence=outputs['halt_confidence'],
            states_sequence=outputs.get('states_sequence'),
            reasoning_mask=reasoning_mask,
            result_token_positions=result_pos,
        )

        # Rule constraint loss on raw states
        raw_states = outputs.get('raw_states', outputs['states_sequence'])
        constraint_dict = constraint_loss_fn(
            raw_states, op_types, op_before_pos, op_after_pos, n_ops
        )

        # Combined loss
        total_loss = loss_dict['total'] + constraint_weight * constraint_dict['total']

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Accumulate metrics
        metrics_accum['total_loss'] += total_loss.item()
        metrics_accum['ce_loss'] += loss_dict['ce_loss']
        metrics_accum['halt_loss'] += loss_dict['halt_loss']
        metrics_accum['constraint_loss'] += constraint_dict['total'].item()
        metrics_accum['identity_loss'] += constraint_dict['identity_loss']
        metrics_accum['cancel_loss'] += constraint_dict['cancel_loss']
        metrics_accum['collapse_loss'] += constraint_dict['collapse_loss']
        n_batches += 1

    return {k: v / n_batches for k, v in metrics_accum.items()}


def train_rim_model(train_ds, val_ds, device, results_dir,
                    epochs=40, batch_size=32, lr=1e-3, patience=5):
    """Train PNA_SSM_RIM on compressible arithmetic."""
    print("\n" + "=" * 60)
    print("Training PNA_SSM_RIM on Compressible Arithmetic")
    print("=" * 60)

    model = PNA_SSM_RIM(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                         max_seq_len=64).to(device)
    n_params = count_parameters(model)
    print(f"  Architecture: PNA_SSM_RIM, Params: {n_params:,}")
    print(f"  Loss: L_ce + 0.1*L_halt + δ*L_constraint (δ annealed)")

    # Wrap datasets
    rim_train = RIMDatasetWrapper(train_ds)
    rim_val = RIMDatasetWrapper(val_ds)
    print(f"  Data: {len(rim_train)} train, {len(rim_val)} val")

    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])
    constraint_loss_fn = RuleConstraintLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    train_loader = DataLoader(rim_train, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(rim_val, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        delta = get_constraint_weight(epoch)

        train_metrics = train_rim_one_epoch(
            model, train_loader, loss_fn, constraint_loss_fn,
            optimizer, scheduler, device, delta
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        val_loss = val_metrics.get('total_loss', val_metrics.get('ce_loss', 0))
        val_acc = val_metrics.get('accuracy', 0)

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'ce_loss': train_metrics['ce_loss'],
            'halt_loss': train_metrics['halt_loss'],
            'constraint_loss': train_metrics['constraint_loss'],
            'identity_loss': train_metrics['identity_loss'],
            'cancel_loss': train_metrics['cancel_loss'],
            'collapse_loss': train_metrics['collapse_loss'],
            'constraint_weight': delta,
            'time': elapsed,
        })

        print(f"  Epoch {epoch:3d} | loss={train_metrics['total_loss']:.4f} "
              f"ce={train_metrics['ce_loss']:.4f} halt={train_metrics['halt_loss']:.4f} "
              f"constr={train_metrics['constraint_loss']:.4f} (δ={delta:.2f}) "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.1%} | {elapsed:.1f}s")

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
    ckpt_path = os.path.join(results_dir, 'rim_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return model, history


# ============================================================================
# Generation + Evaluation
# ============================================================================

def evaluate_purity_corrected(model, generated_ids, expression, device, purity_eval):
    """
    Correct purity evaluation: run one final forward pass on the complete
    generated sequence to get states aligned with token positions.

    During autoregressive generation, state_vectors[i] = state at generation
    step i, NOT at token position i. find_op_token_positions returns token
    positions. This mismatch caused the false 16.6% purity reading.

    Fix: feed the complete generated sequence through the model in one pass.
    Now states_sequence[0, pos, :] = state at token position pos.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

        max_pos = getattr(model, 'max_seq_len', 256)
        if hasattr(model, 'pos_encoding'):
            max_pos = model.pos_encoding.num_embeddings
        if input_tensor.size(1) > max_pos:
            input_tensor = input_tensor[:, :max_pos]

        outputs = model(input_tensor)
        # states_sequence[0, pos, :] = state at token position pos
        full_states = outputs['states_sequence'][0].cpu()  # [L, d_state]

    # Now indexing is correct: full_states[pos] = state at token pos
    position_states = [full_states[i] for i in range(full_states.size(0))]

    purity_result = purity_eval.evaluate_trajectory(
        position_states, expression, generated_ids[:full_states.size(0)]
    )

    return purity_result


def run_rim_generation(model, test_ds, device):
    """
    Run autoregressive generation with corrected purity tracking.
    Reuses CompressibleGenerator and adds GeodesicPurity evaluation.

    Purity fix: uses a single forward pass on the complete generated sequence
    to get states aligned with token positions, rather than using incremental
    generation-step states which have mismatched indices.
    """
    generator = CompressibleGenerator(model, device=device)
    purity_eval = GeodesicPurity()

    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        expression = ex['expression']
        ground_truth = ex['answer']
        tier = ex['tier']

        gen_result = generator.generate(expression)

        # Confusion head analysis (uses generation-step states — correct for this)
        osc_result = detect_oscillation(gen_result['state_vectors'])
        convergence = classify_convergence(gen_result, osc_result)

        # CORRECTED purity: single forward pass on complete sequence
        purity_result = evaluate_purity_corrected(
            model, gen_result['generated_ids'], expression, device, purity_eval
        )

        is_correct = (gen_result['parsed_answer'] == ground_truth)

        # Constraint violation at halt: check state at last step
        constraint_at_halt = 0.0
        if gen_result['state_vectors']:
            last_state = gen_result['state_vectors'][-1]
            if len(gen_result['state_vectors']) > 1:
                prev_state = gen_result['state_vectors'][-2]
                constraint_at_halt = (last_state - prev_state).norm().item()

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
            'geodesic_purity': purity_result['purity'],
            'n_constraints': purity_result['n_constraints'],
            'n_satisfied': purity_result.get('n_satisfied', 0),
            'constraint_at_halt': constraint_at_halt,
        })

    return results


# ============================================================================
# Figures
# ============================================================================

def plot_geodesic_purity(results, fig_dir):
    """
    Fig 19: Geodesic purity analysis.
    Left: purity distribution per tier (box plots).
    Right: purity vs accuracy scatter.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: Purity by tier ---
    tier_purities = defaultdict(list)
    for r in results:
        if r['n_constraints'] > 0:
            tier_purities[r['tier']].append(r['geodesic_purity'])

    tiers = sorted(tier_purities.keys())
    data_for_box = [tier_purities.get(t, [0]) for t in tiers]
    labels = [f"T{t} ({TIER_NAMES[t]})" for t in tiers]
    colors = [TIER_COLORS[t] for t in tiers]

    bp = ax1.boxplot(data_for_box, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean annotations
    for i, t in enumerate(tiers):
        vals = tier_purities.get(t, [0])
        mean_p = np.mean(vals) if vals else 0
        ax1.annotate(f'μ={mean_p:.2f}\nn={len(vals)}',
                     (i + 1, max(vals) + 0.05 if vals else 0.5),
                     ha='center', fontsize=9)

    ax1.set_ylabel("Geodesic Purity")
    ax1.set_title("Geodesic Purity by Tier")
    ax1.set_ylim(-0.05, 1.15)

    # --- Right: Purity vs accuracy scatter ---
    constrained = [r for r in results if r['n_constraints'] > 0]
    purities = [r['geodesic_purity'] for r in constrained]
    correct = [1 if r['is_correct'] else 0 for r in constrained]
    tier_labels = [r['tier'] for r in constrained]

    for t in tiers:
        mask = [i for i, tl in enumerate(tier_labels) if tl == t]
        p = [purities[i] for i in mask]
        c = [correct[i] for i in mask]
        # Jitter for visibility
        jitter_c = [ci + np.random.uniform(-0.05, 0.05) for ci in c]
        jitter_p = [pi + np.random.uniform(-0.02, 0.02) for pi in p]
        ax2.scatter(jitter_p, jitter_c, alpha=0.3, s=15, color=TIER_COLORS[t],
                    label=f'T{t}')

    ax2.set_xlabel("Geodesic Purity")
    ax2.set_ylabel("Correct (1) / Wrong (0)")
    ax2.set_title("Purity vs Correctness")
    ax2.legend(fontsize=9)

    fig.suptitle("Rule-Initialized Model: Geodesic Purity Analysis",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig19_rim_geodesic_purity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_convergence_improvement(results, baseline_results_path, fig_dir):
    """
    Fig 20: Convergence improvement vs Phase 11 baseline.
    Left: stacked bar of convergence types per tier (baseline vs RIM).
    Right: accuracy by convergence type.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Load baseline results
    baseline_data = None
    if os.path.exists(baseline_results_path):
        with open(baseline_results_path) as f:
            baseline_data = json.load(f)

    # --- Left: Convergence type distribution ---
    tiers = [0, 1, 2]
    x = np.arange(len(tiers))
    width = 0.35

    # RIM convergence counts
    rim_conv = defaultdict(lambda: defaultdict(int))
    for r in results:
        rim_conv[r['tier']][r['convergence']] += 1

    rim_true_rates = []
    rim_false_rates = []
    for t in tiers:
        total = sum(rim_conv[t].values()) or 1
        rim_true_rates.append(rim_conv[t].get('true_convergence', 0) / total * 100)
        rim_false_rates.append(rim_conv[t].get('false_convergence', 0) / total * 100)

    # Baseline rates from saved results
    base_true_rates = [0, 0.9, 9.7]  # Phase 11 defaults
    base_false_rates = [100, 99.1, 90.3]
    if baseline_data and 'tier_metrics' in baseline_data:
        for i, t in enumerate(tiers):
            tm = baseline_data['tier_metrics'].get(str(t), {})
            conv = tm.get('convergence', {})
            total = tm.get('n', 1)
            base_true_rates[i] = conv.get('true_convergence', 0) / total * 100
            base_false_rates[i] = conv.get('false_convergence', 0) / total * 100

    # Plot grouped bars
    bars1 = ax1.bar(x - width/2, base_true_rates, width, color='#95a5a6',
                     alpha=0.7, edgecolor='black', label='Baseline True Conv')
    bars2 = ax1.bar(x + width/2, rim_true_rates, width, color='#2ecc71',
                     alpha=0.8, edgecolor='black', label='RIM True Conv')

    for i in range(len(tiers)):
        ax1.annotate(f'{base_true_rates[i]:.1f}%', (i - width/2, base_true_rates[i] + 1),
                     ha='center', fontsize=8)
        ax1.annotate(f'{rim_true_rates[i]:.1f}%', (i + width/2, rim_true_rates[i] + 1),
                     ha='center', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{t}' for t in tiers])
    ax1.set_ylabel("True Convergence Rate (%)")
    ax1.set_title("True Convergence: Baseline vs RIM")
    ax1.legend(fontsize=9)

    # --- Right: Accuracy by convergence type ---
    categories = ['True Conv', 'False Conv']
    rim_conv_results = [r for r in results if r['convergence'] == 'true_convergence']
    rim_cycle_results = [r for r in results if r['convergence'] == 'false_convergence']

    rim_acc_conv = np.mean([r['is_correct'] for r in rim_conv_results]) * 100 if rim_conv_results else 0
    rim_acc_cycle = np.mean([r['is_correct'] for r in rim_cycle_results]) * 100 if rim_cycle_results else 0

    base_acc_conv = 94.4  # Phase 11 baseline
    base_acc_cycle = 47.4
    if baseline_data and 'basin_analysis' in baseline_data:
        ba = baseline_data['basin_analysis']
        base_acc_conv = ba.get('convergence', {}).get('accuracy', 0.944) * 100
        base_acc_cycle = ba.get('cycling', {}).get('accuracy', 0.474) * 100

    x2 = np.arange(len(categories))
    ax2.bar(x2 - width/2, [base_acc_conv, base_acc_cycle], width,
            color='#95a5a6', alpha=0.7, edgecolor='black', label='Baseline')
    ax2.bar(x2 + width/2, [rim_acc_conv, rim_acc_cycle], width,
            color='#3498db', alpha=0.8, edgecolor='black', label='RIM')

    for i, (bv, rv) in enumerate(zip([base_acc_conv, base_acc_cycle],
                                       [rim_acc_conv, rim_acc_cycle])):
        ax2.annotate(f'{bv:.1f}%', (i - width/2, bv + 1), ha='center', fontsize=9)
        ax2.annotate(f'{rv:.1f}%', (i + width/2, rv + 1), ha='center', fontsize=9)

    n_conv = len(rim_conv_results)
    n_cycle = len(rim_cycle_results)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'{c}\n(n={n})' for c, n in zip(categories, [n_conv, n_cycle])])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy by Convergence Type")
    ax2.legend(fontsize=9)

    fig.suptitle("RIM: Convergence Improvement over Phase 11 Baseline",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig20_rim_convergence_improvement.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_constraint_annealing(history, fig_dir):
    """
    Fig 21: Training curves with constraint annealing.
    Left: loss components over epochs.
    Right: val accuracy and constraint loss over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = [h['epoch'] for h in history]
    ce_losses = [h['ce_loss'] for h in history]
    halt_losses = [h['halt_loss'] for h in history]
    constr_losses = [h['constraint_loss'] for h in history]
    delta_weights = [h['constraint_weight'] for h in history]
    val_accs = [h['val_acc'] for h in history]

    # --- Left: Loss components ---
    ax1.plot(epochs, ce_losses, 'b-', linewidth=2, label='CE Loss')
    ax1.plot(epochs, halt_losses, 'r-', linewidth=2, label='Halt Loss')
    ax1.plot(epochs, constr_losses, 'g-', linewidth=2, label='Constraint Loss')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, delta_weights, 'k--', linewidth=1.5, alpha=0.5, label='δ weight')
    ax1_twin.set_ylabel("Constraint Weight (δ)", color='gray')

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Components")
    ax1.legend(loc='upper right', fontsize=9)
    ax1_twin.legend(loc='center right', fontsize=9)

    # --- Right: Val accuracy ---
    ax2.plot(epochs, [a * 100 for a in val_accs], 'b-', linewidth=2, label='Val Accuracy')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy During Training")
    ax2.legend(fontsize=9)

    # Add constraint loss on secondary axis
    ax2_twin = ax2.twinx()
    weighted_constr = [c * d for c, d in zip(constr_losses, delta_weights)]
    ax2_twin.plot(epochs, weighted_constr, 'g--', linewidth=1.5, alpha=0.6,
                  label='δ × Constraint')
    ax2_twin.set_ylabel("Weighted Constraint Loss", color='green')
    ax2_twin.legend(loc='center right', fontsize=9)

    fig.suptitle("RIM Training: Constraint Annealing",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig21_rim_constraint_annealing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 12: Rule-Initialized Models")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--val-n', type=int, default=1000)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 12: Rule-Initialized Models (RIM) Experiment")
    print(f"{'=' * 60}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Create Datasets ----
    print("\n--- Creating Datasets ---")
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

    # Tier distribution check
    for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
        tier_counts = defaultdict(int)
        for ex in ds.examples:
            tier_counts[ex['tier']] += 1
        print(f"  {name}: {len(ds)} examples, tiers: {dict(sorted(tier_counts.items()))}")

    # ---- Train or Load Model ----
    ckpt_path = os.path.join(args.results_dir, 'rim_model.pt')
    if args.skip_training and os.path.exists(ckpt_path):
        print(f"\n--- Loading pretrained model from {ckpt_path} ---")
        model = PNA_SSM_RIM(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                             max_seq_len=64).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        history = []
    else:
        model, history = train_rim_model(
            train_ds, val_ds, device, args.results_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience
        )

    # ---- Teacher-forced Evaluation ----
    print("\n--- Teacher-forced Evaluation ---")
    rim_val = RIMDatasetWrapper(val_ds)
    val_loader = DataLoader(rim_val, batch_size=args.batch_size, shuffle=False)
    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])
    test_loader_rim = DataLoader(RIMDatasetWrapper(test_ds),
                                  batch_size=args.batch_size, shuffle=False)
    # Use base test_ds for standard evaluate (which expects standard batch keys)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    halt_f1 = compute_halt_f1(model, test_loader, device)
    print(f"  Test accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  Halt F1: {halt_f1:.4f}")

    # ---- Autoregressive Generation ----
    print("\n--- Autoregressive Generation ---")
    gen_results = run_rim_generation(model, test_ds, device)

    # ---- Tier Metrics ----
    print("\n--- Per-Tier Metrics ---")
    tier_metrics = compute_tier_metrics(gen_results)
    for tier in sorted(tier_metrics.keys()):
        m = tier_metrics[tier]
        print(f"\n  Tier {tier} ({TIER_NAMES.get(tier, '?')}):")
        print(f"    N={m['n']}, Accuracy={m['accuracy']:.1%}")
        print(f"    Reasoning tokens: {m['mean_reasoning_tokens']:.1f} ± {m['std_reasoning_tokens']:.1f}")
        print(f"    Token economy η: {m['token_economy_eta']:.4f}")
        print(f"    Mean halt time: {m['mean_halt_time']:.1f} ± {m['std_halt_time']:.1f}")
        print(f"    Convergence: {dict(m['convergence'])}")
        print(f"    Stop reasons: {dict(m['stop_reasons'])}")

    # ---- Geodesic Purity ----
    print("\n--- Geodesic Purity ---")
    purity_by_tier = defaultdict(list)
    for r in gen_results:
        if r['n_constraints'] > 0:
            purity_by_tier[r['tier']].append(r['geodesic_purity'])

    for tier in sorted(purity_by_tier.keys()):
        vals = purity_by_tier[tier]
        print(f"  Tier {tier}: mean purity = {np.mean(vals):.3f} ± {np.std(vals):.3f} "
              f"(n={len(vals)})")

    # Overall purity
    all_purities = [r['geodesic_purity'] for r in gen_results if r['n_constraints'] > 0]
    if all_purities:
        print(f"  Overall: mean purity = {np.mean(all_purities):.3f}")

    # ---- Basin Analysis ----
    print("\n--- Convergence vs Halt Analysis (Basin vs Fixed Point) ---")
    basin_metrics = analyze_convergence_vs_halt(gen_results)

    conv = basin_metrics['convergence']
    cyc = basin_metrics['cycling']
    print(f"\n  True Convergence (n={conv['n']}):")
    print(f"    Accuracy: {conv['accuracy']:.1%}")
    print(f"    Mean tokens: {conv['mean_tokens']:.1f}")
    print(f"    Entropy at halt: {conv['mean_entropy_at_halt']:.3f} bits")

    print(f"\n  State Cycling (n={cyc['n']}):")
    print(f"    Accuracy: {cyc['accuracy']:.1%}")
    print(f"    Mean tokens: {cyc['mean_tokens']:.1f}")
    print(f"    Entropy at halt: {cyc['mean_entropy_at_halt']:.3f} bits")

    print(f"\n  Accuracy gap: {basin_metrics['accuracy_gap']:+.1%}")
    print(f"  Interpretation: {basin_metrics['interpretation']}")

    # ---- Save Results ----
    purity_summary = {}
    for tier in sorted(purity_by_tier.keys()):
        vals = purity_by_tier[tier]
        purity_summary[str(tier)] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals),
        }

    output = {
        'teacher_forced': {
            'accuracy': test_metrics.get('accuracy', 0),
            'halt_f1': halt_f1,
        },
        'tier_metrics': {str(k): v for k, v in tier_metrics.items()},
        'geodesic_purity': purity_summary,
        'overall_purity': float(np.mean(all_purities)) if all_purities else 0.0,
        'basin_analysis': basin_metrics,
        'training_history': history,
        'config': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'train_n': args.train_n,
            'val_n': args.val_n,
            'test_n': args.test_n,
            'model': 'PNA_SSM_RIM',
            'constraint_annealing': 'warmup=10, ramp=20',
        },
    }

    results_path = os.path.join(args.results_dir, 'rim_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved results: {results_path}")

    # ---- Figures ----
    print("\n--- Generating Figures ---")
    plot_geodesic_purity(gen_results, args.fig_dir)

    baseline_path = os.path.join(args.results_dir, 'compressible_task_results.json')
    plot_convergence_improvement(gen_results, baseline_path, args.fig_dir)

    if history:
        plot_constraint_annealing(history, args.fig_dir)

    # ---- Success Criteria Check ----
    print("\n--- Success Criteria Check ---")
    t2_metrics = tier_metrics.get(2, {})
    t2_accuracy = t2_metrics.get('accuracy', 0)
    t2_tokens = t2_metrics.get('mean_reasoning_tokens', 999)

    false_conv_rate = 0
    total_gen = len(gen_results)
    false_conv_count = sum(1 for r in gen_results if r['convergence'] == 'false_convergence')
    false_conv_rate = false_conv_count / max(1, total_gen) * 100

    overall_purity = float(np.mean(all_purities)) if all_purities else 0

    constraint_at_halt = np.mean([r['constraint_at_halt'] for r in gen_results])

    print(f"  Tier 2 Accuracy: {t2_accuracy:.1%} (target: >85%) — "
          f"{'PASS' if t2_accuracy > 0.85 else 'FAIL'}")
    print(f"  False Convergence: {false_conv_rate:.1f}% (target: <30%) — "
          f"{'PASS' if false_conv_rate < 30 else 'FAIL'}")
    print(f"  Geodesic Purity: {overall_purity:.1%} (target: >80%) — "
          f"{'PASS' if overall_purity > 0.80 else 'FAIL'}")
    print(f"  Tier 2 Tokens: {t2_tokens:.1f} (target: <20) — "
          f"{'PASS' if t2_tokens < 20 else 'FAIL'}")
    print(f"  Constraint at Halt: {constraint_at_halt:.3f} (target: <0.1) — "
          f"{'PASS' if constraint_at_halt < 0.1 else 'FAIL'}")

    print("\n" + "=" * 60)
    print("Phase 12: Rule-Initialized Models Experiment COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
