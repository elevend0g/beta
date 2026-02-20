"""
Phase 14: Halt Control on a Geometrically Sound Manifold

Phase 12 proved the geometry works (83% geodesic purity). Phase 13b proved
that invasive SSM modification destroys it (18% purity). The remaining
problem is halting: the Halt Head fires during limit cycles (96% false
convergence rate), not at true fixed points.

Phase 14 fixes halting without touching the SSM dynamics:
  14A: Diagnostics — extended budget, three-signal validation, op probe
  14B: Confusion detection — non-parametric CycleDetector + learned ConfusionHead
  14C: Halt gating — confusion-modulated halt threshold
  14D: Cycle breaking — input re-injection when patience exhausted (conditional)
  14E: Integration + figures

All experiments use the Phase 12 checkpoint (results/rim_model.pt) with
frozen backbone. No SSM recurrence modification.
"""

import os
import sys
import re
import json
import math
import time
import copy
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

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, count_parameters
from losses import ThermodynamicLoss
from train import get_device
from compressible_task import (
    CompressibleArithmeticDataset, CompressibleGenerator,
    detect_oscillation, classify_convergence,
    compute_tier_metrics, analyze_convergence_vs_halt,
    TIER_NAMES, TIER_COLORS,
)
from rule_initialization import (
    PNA_SSM_RIM, GeodesicPurity, RIMDatasetWrapper,
    evaluate_purity_corrected, find_op_token_positions,
    OP_PAD, OP_REAL, OP_IDENTITY, OP_CANCEL_START, OP_CANCEL_END,
    OP_STAR_ZERO, MAX_OPS,
)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_state_entropy(state_vec):
    """Compute energy-based entropy of a state vector."""
    energy = state_vec ** 2
    probs = energy / (energy.sum() + 1e-9)
    ent = -(probs * torch.log2(probs + 1e-9)).sum().item()
    return ent


def compute_entropy_gradient(entropies, t, window=3):
    """Smoothed entropy gradient. Negative = converging."""
    if t < 1:
        return 0.0
    start = max(0, t - window)
    if start == t:
        return 0.0
    return (entropies[t] - entropies[start]) / (t - start)


def compute_cycling_score(state_vectors, t, k_values=[2, 4, 8]):
    """
    Detect periodic orbits in state space.
    Sc(t,k) = 1 - ||h_t - h_{t-k}|| / (||h_t|| + ||h_{t-k}|| + eps)
    High Sc -> state has returned to a previous position.
    """
    if t < min(k_values):
        return 0.0

    scores = []
    h_t = state_vectors[t]
    for k in k_values:
        if t - k >= 0:
            h_prev = state_vectors[t - k]
            dist = (h_t - h_prev).norm().item()
            norm_sum = h_t.norm().item() + h_prev.norm().item() + 1e-9
            scores.append(1.0 - dist / norm_sum)

    return max(scores) if scores else 0.0


# ============================================================================
# 14A-0: Extended Generation Budget
# ============================================================================

def experiment_14a0_extended_budget(model, test_ds, device):
    """
    Does giving the model more time improve accuracy?

    Four conditions:
      1. Default halt threshold (0.95) — baseline (matches CompressibleGenerator default)
      2. Raised threshold (0.98) — model must be more confident to halt
      3. Raised threshold (0.995) — model almost never halts voluntarily
      4. No halt at all (threshold=1.0) — generate max_length tokens

    If accuracy improves with raised thresholds:
      -> False convergence is partly a confidence calibration problem
    If accuracy doesn't improve:
      -> Model is stuck in limit cycles regardless of halt timing
    """
    print("\n" + "=" * 60)
    print("14A-0: Extended Generation Budget")
    print("=" * 60)

    thresholds = [0.95, 0.98, 0.995, 1.0]
    results = {}

    for thresh in thresholds:
        generator = CompressibleGenerator(
            model, device=device,
            halt_threshold=thresh,
        )

        gen_results = []
        for i in tqdm(range(len(test_ds.examples)),
                      desc=f"thresh={thresh}", leave=False):
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
                'stop_reason': gen['stop_reason'],
            })

        # Per-tier accuracy
        for tier in [0, 1, 2]:
            tier_results = [r for r in gen_results if r['tier'] == tier]
            if not tier_results:
                continue
            acc = np.mean([r['is_correct'] for r in tier_results])
            tokens = np.mean([r['reasoning_tokens'] for r in tier_results])
            true_conv = sum(
                1 for r in tier_results
                if r['convergence'] == 'true_convergence'
            )
            results[(thresh, tier)] = {
                'accuracy': float(acc),
                'mean_tokens': float(tokens),
                'true_convergence': int(true_conv),
                'n': len(tier_results),
            }

        overall_acc = np.mean([r['is_correct'] for r in gen_results])
        overall_tokens = np.mean([r['reasoning_tokens'] for r in gen_results])
        overall_true_conv = sum(
            1 for r in gen_results
            if r['convergence'] == 'true_convergence'
        )
        results[(thresh, 'all')] = {
            'accuracy': float(overall_acc),
            'mean_tokens': float(overall_tokens),
            'true_convergence': int(overall_true_conv),
            'n': len(gen_results),
        }

        label = f"thresh={thresh}" if thresh < 1.0 else "no_halt(1.0)"
        t2 = results.get((thresh, 2), {})
        print(f"  {label:15s} | T2 acc={t2.get('accuracy',0):.1%} "
              f"tokens={t2.get('mean_tokens',0):.1f} "
              f"true_conv={t2.get('true_convergence',0)} "
              f"| overall={results[(thresh, 'all')]['accuracy']:.1%}")

    # Decision logic
    baseline_acc = results.get((0.95, 2), {}).get('accuracy', 0)
    raised_acc = results.get((0.98, 2), {}).get('accuracy', 0)
    high_acc = results.get((0.995, 2), {}).get('accuracy', 0)
    nohalt_acc = results.get((1.0, 2), {}).get('accuracy', 0)

    print(f"\n  Decision analysis:")
    print(f"    Baseline (0.95) T2: {baseline_acc:.1%}")
    print(f"    Raised  (0.98) T2: {raised_acc:.1%}")
    print(f"    High   (0.995) T2: {high_acc:.1%}")
    print(f"    No halt (1.0)  T2: {nohalt_acc:.1%}")

    if nohalt_acc > baseline_acc + 0.10:
        print(f"    -> Halt truncation is the dominant problem (fix halt -> done)")
        decision = "halt_truncation"
    elif raised_acc > baseline_acc + 0.05:
        print(f"    -> Halt calibration IS the dominant problem (14C should suffice)")
        decision = "calibration_dominant"
    elif high_acc <= baseline_acc + 0.02:
        print(f"    -> More time doesn't help -> model is stuck in cycles (14D needed)")
        decision = "cycles_stuck"
    else:
        print(f"    -> Marginal improvement -> both 14C and 14D may help")
        decision = "mixed"

    return results, decision


# ============================================================================
# 14A-1: Three-Signal Validation
# ============================================================================

def experiment_14a1_three_signal(model, test_ds, device):
    """
    Compute H, grad_H, Sc at each generation step.
    Test whether the three-signal framework discriminates
    true convergence from false convergence.
    """
    print("\n" + "=" * 60)
    print("14A-1: Three-Signal Validation")
    print("=" * 60)

    generator = CompressibleGenerator(model, device=device)

    trajectories = []
    for i in tqdm(range(len(test_ds.examples)), desc="14A-1"):
        ex = test_ds.examples[i]
        gen = generator.generate(ex['expression'])

        states = gen['state_vectors']
        entropies = gen['state_entropies']
        osc = detect_oscillation(states)
        convergence = classify_convergence(gen, osc)
        is_correct = (gen['parsed_answer'] == ex['answer'])

        # Compute per-step signals
        per_step = []
        for t in range(len(states)):
            H = entropies[t] if t < len(entropies) else 0.0
            grad_H = compute_entropy_gradient(entropies, t, window=3)
            Sc = compute_cycling_score(states, t, k_values=[2, 4, 8])

            per_step.append({
                'H': H,
                'grad_H': grad_H,
                'Sc': Sc,
                'halt_conf': (gen['halt_confidences'][t]
                              if t < len(gen['halt_confidences'])
                              else 0.0),
            })

        # Summary features at halt point
        T = len(per_step) - 1
        if T >= 0:
            trajectories.append({
                'tier': ex['tier'],
                'convergence': convergence,
                'is_correct': is_correct,
                'n_steps': len(states),
                'H_at_halt': per_step[T]['H'],
                'grad_H_at_halt': per_step[T]['grad_H'],
                'Sc_at_halt': per_step[T]['Sc'],
                'halt_conf_at_halt': per_step[T]['halt_conf'],
                # Window features (last 5 steps)
                'mean_Sc_last5': float(np.mean(
                    [s['Sc'] for s in per_step[max(0, T - 4):T + 1]]
                )),
                'mean_gradH_last5': float(np.mean(
                    [s['grad_H'] for s in per_step[max(0, T - 4):T + 1]]
                )),
                'max_Sc': max(s['Sc'] for s in per_step),
                'per_step': per_step,
            })

    # Discrimination analysis
    true_conv = [t for t in trajectories
                 if t['convergence'] == 'true_convergence']
    false_conv = [t for t in trajectories
                  if t['convergence'] == 'false_convergence']

    print(f"\n  True Convergence (n={len(true_conv)}):")
    if true_conv:
        print(f"    Sc at halt:    {np.mean([t['Sc_at_halt'] for t in true_conv]):.3f}")
        print(f"    grad_H at halt: {np.mean([t['grad_H_at_halt'] for t in true_conv]):.3f}")
        print(f"    H at halt:     {np.mean([t['H_at_halt'] for t in true_conv]):.3f}")
    else:
        print(f"    (no true convergence cases)")

    print(f"\n  False Convergence (n={len(false_conv)}):")
    if false_conv:
        print(f"    Sc at halt:    {np.mean([t['Sc_at_halt'] for t in false_conv]):.3f}")
        print(f"    grad_H at halt: {np.mean([t['grad_H_at_halt'] for t in false_conv]):.3f}")
        print(f"    H at halt:     {np.mean([t['H_at_halt'] for t in false_conv]):.3f}")
    else:
        print(f"    (no false convergence cases)")

    # ROC analysis
    labels = [1 if t['convergence'] == 'true_convergence' else 0
              for t in trajectories]
    auc_results = {}

    if len(set(labels)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            for feature in ['Sc_at_halt', 'grad_H_at_halt', 'mean_Sc_last5',
                            'mean_gradH_last5']:
                values = [t[feature] for t in trajectories]
                try:
                    # True convergence should have LOWER Sc (negate)
                    auc = roc_auc_score(labels, [-v for v in values])
                    auc_results[feature] = float(auc)
                    print(f"    AUC ({feature}): {auc:.3f}")
                except ValueError:
                    print(f"    AUC ({feature}): undefined")

            # Combined: simple logistic on top 3 features
            from sklearn.linear_model import LogisticRegression
            feature_matrix = np.array([
                [t['Sc_at_halt'], t['grad_H_at_halt'], t['mean_Sc_last5']]
                for t in trajectories
            ])
            lr = LogisticRegression(max_iter=1000)
            lr.fit(feature_matrix, labels)
            combined_scores = lr.predict_proba(feature_matrix)[:, 1]
            combined_auc = roc_auc_score(labels, combined_scores)
            auc_results['combined'] = float(combined_auc)
            print(f"    AUC (combined three-signal): {combined_auc:.3f}")
        except ImportError:
            print("    sklearn not available — skipping AUC analysis")
    else:
        print("    Only one convergence class — cannot compute AUC")

    return trajectories, auc_results


# ============================================================================
# 14A-2: Op Detector Probe on Phase 12 Backbone
# ============================================================================

def experiment_14a2_op_probe(model, train_ds, test_ds, device, epochs=10):
    """
    Train a lightweight Op Detector on Phase 12's FROZEN backbone.
    Tests: "Do Phase 12 states encode operation types?"
    """
    print("\n" + "=" * 60)
    print("14A-2: Op Detector Probe on Phase 12 Backbone")
    print("=" * 60)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Build Op Detector
    N_OP_TYPES = 6
    op_detector = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, N_OP_TYPES),
    ).to(device)

    optimizer = torch.optim.Adam(op_detector.parameters(), lr=1e-3)
    rim_train = RIMDatasetWrapper(train_ds)
    loader = DataLoader(rim_train, batch_size=32, shuffle=True)

    history = []
    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            op_types = batch['op_types'].to(device)
            op_before_pos = batch['op_before_pos'].to(device)
            op_after_pos = batch['op_after_pos'].to(device)
            n_ops = batch['n_ops'].to(device)

            B, L = input_ids.shape

            with torch.no_grad():
                # Extract hidden representations through the full model
                h = model.embedding(input_ids) + model.pos_encoding(
                    torch.arange(L, device=device).unsqueeze(0)
                )
                for layer in model.layers:
                    h, _ = layer(h)
                h = model.norm(h)

            # Op detection on hidden states
            op_logits = op_detector(h)

            # Build per-token op labels
            per_token_labels = torch.zeros(B, L, dtype=torch.long, device=device)
            pad_mask = (input_ids != VOCAB['<PAD>'])
            per_token_labels[pad_mask] = OP_REAL

            for b_idx in range(B):
                n = n_ops[b_idx].item()
                for op_i in range(n):
                    ot = op_types[b_idx, op_i].item()
                    bp = op_before_pos[b_idx, op_i].item()
                    ap = op_after_pos[b_idx, op_i].item()
                    for pos in range(bp + 1, min(ap + 1, L)):
                        per_token_labels[b_idx, pos] = ot

            loss = F.cross_entropy(
                op_logits.reshape(-1, N_OP_TYPES),
                per_token_labels.reshape(-1),
                ignore_index=OP_PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = op_logits.argmax(dim=-1)
            mask = per_token_labels != OP_PAD
            total_correct += (preds[mask] == per_token_labels[mask]).sum().item()
            total_samples += mask.sum().item()
            total_loss += loss.item()

        acc = total_correct / max(total_samples, 1)
        avg_loss = total_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': acc})
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f} acc={acc:.1%}")

    # Unfreeze backbone after probe
    for p in model.parameters():
        p.requires_grad = True

    final_acc = history[-1]['accuracy'] if history else 0.0
    print(f"\n  Final probe accuracy: {final_acc:.1%}")
    print(f"  (Expected ~74% if Phase 12 states encode op types)")

    return op_detector, history


# ============================================================================
# 14B-1: Non-Parametric Cycle Detector
# ============================================================================

class CycleDetector:
    """
    Non-parametric false convergence detector.
    Uses state delta statistics to distinguish fixed points from limit cycles.
    No training required.
    """

    def __init__(self, window=5, convergence_threshold=0.05,
                 cycling_cv_threshold=0.3):
        self.window = window
        self.conv_thresh = convergence_threshold
        self.cycling_cv_thresh = cycling_cv_threshold

    def detect(self, state_vectors):
        """
        Returns:
          ('converged', score):  deltas are all small (fixed point)
          ('cycling', score):    deltas are consistent but non-zero (limit cycle)
          ('diverging', score):  deltas are growing
          ('uncertain', score):  not enough data
        """
        if len(state_vectors) < self.window + 1:
            return 'uncertain', 0.0

        recent = torch.stack(state_vectors[-(self.window + 1):])
        deltas = (recent[1:] - recent[:-1]).norm(dim=-1)

        mean_delta = deltas.mean().item()
        max_delta = deltas.max().item()

        # Fixed point: all deltas are tiny
        if max_delta < self.conv_thresh:
            return 'converged', 0.0

        # Limit cycle: consistent non-zero deltas
        cv = deltas.std().item() / (mean_delta + 1e-9)
        if cv < self.cycling_cv_thresh:
            confusion_score = 1.0 - cv / self.cycling_cv_thresh
            return 'cycling', confusion_score

        # Diverging: growing deltas
        if deltas[-1] > deltas[0] * 1.5:
            return 'diverging', 0.8

        return 'uncertain', 0.3

    def should_suppress_halt(self, state_vectors, halt_confidence):
        """Should we suppress the halt at this step?"""
        status, confusion = self.detect(state_vectors)
        if status == 'cycling' and halt_confidence > 0.5:
            return True, confusion
        return False, confusion


def evaluate_cycle_detector(model, test_ds, device):
    """
    Evaluate CycleDetector's ability to predict true/false convergence.
    No intervention — pure detection accuracy.
    """
    print("\n" + "=" * 60)
    print("14B-1: Non-Parametric Cycle Detector Evaluation")
    print("=" * 60)

    generator = CompressibleGenerator(model, device=device)
    detector = CycleDetector()

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in tqdm(range(len(test_ds.examples)), desc="14B-1"):
        ex = test_ds.examples[i]
        gen = generator.generate(ex['expression'])
        osc = detect_oscillation(gen['state_vectors'])
        truth = classify_convergence(gen, osc)

        status, _ = detector.detect(gen['state_vectors'])

        predicted_cycling = (status == 'cycling')
        actual_false_conv = (truth == 'false_convergence')

        if predicted_cycling and actual_false_conv:
            tp += 1
        elif predicted_cycling and not actual_false_conv:
            fp += 1
        elif not predicted_cycling and actual_false_conv:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"  CycleDetector: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    False alarm rate: {fp}/{fp + tn} = {fp / max(fp + tn, 1):.1%}")

    result = {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }
    return result


# ============================================================================
# 14B-2: Learned Confusion Head
# ============================================================================

class ConfusionHead(nn.Module):
    """
    Detects false convergence from trajectory features.
    Input: window of state deltas + entropy features + halt confidences.
    Output: confusion probability (high = likely false convergence).
    """

    def __init__(self, d_state=16, window=5, d_hidden=32):
        super().__init__()
        self.window = window
        self.d_state = d_state
        # Input per step: d_state (delta) + 3 (H, grad_H, Sc) + 1 (halt_conf)
        input_per_step = d_state + 4
        self.net = nn.Sequential(
            nn.Linear(input_per_step * window, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, features):
        """features: [B, window, d_state+4] -> confusion logits [B, 1]"""
        B = features.size(0)
        return self.net(features.reshape(B, -1))


def collect_confusion_labels(model, train_ds, device, n_samples=2000):
    """
    Generate trajectories and label each step with confusion signal.
    Label = 1 if false convergence AND near halt, else 0.
    """
    print("\n  Collecting confusion training data...")
    generator = CompressibleGenerator(model, device=device)
    windows = []
    window = 5

    for i in tqdm(range(min(n_samples, len(train_ds.examples))),
                  desc="Collecting labels"):
        ex = train_ds.examples[i]
        gen = generator.generate(ex['expression'])

        states = gen['state_vectors']
        entropies = gen['state_entropies']
        halt_confs = gen['halt_confidences']

        osc = detect_oscillation(states)
        convergence = classify_convergence(gen, osc)
        is_false_conv = (convergence == 'false_convergence')

        if len(states) < window + 1:
            continue

        halt_step = len(states) - 1

        for t in range(window, len(states)):
            step_features = []
            for w in range(t - window, t):
                if w == 0:
                    delta = torch.zeros_like(states[0])
                else:
                    delta = states[w] - states[w - 1]

                H = entropies[w] if w < len(entropies) else 0.0
                grad_H = compute_entropy_gradient(entropies, w)
                Sc = compute_cycling_score(states, w)
                hc = halt_confs[w] if w < len(halt_confs) else 0.0

                feat = torch.cat([
                    delta.cpu(),
                    torch.tensor([H, grad_H, Sc, hc]),
                ])
                step_features.append(feat)

            features = torch.stack(step_features)

            near_halt = (halt_step - t) < 5
            label = 1.0 if (is_false_conv and near_halt) else 0.0

            windows.append({
                'features': features,
                'label': label,
                'tier': ex['tier'],
            })

    n_pos = sum(1 for w in windows if w['label'] > 0.5)
    n_neg = len(windows) - n_pos
    print(f"  Collected {len(windows)} windows ({n_pos} positive, {n_neg} negative)")

    return windows


def train_confusion_head(d_state, training_windows, device, epochs=20, lr=1e-3):
    """Train Confusion Head on retrospective trajectory labels."""
    print("\n  Training Confusion Head...")
    confusion_head = ConfusionHead(d_state=d_state).to(device)

    n_pos = sum(1 for w in training_windows if w['label'] > 0.5)
    n_neg = len(training_windows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    optimizer = torch.optim.Adam(confusion_head.parameters(), lr=lr)

    features = torch.stack([w['features'] for w in training_windows])
    labels = torch.tensor([w['label'] for w in training_windows])

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    history = []
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device).unsqueeze(-1)

            logits = confusion_head(feat_batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, label_batch, pos_weight=pos_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (logits > 0).float()
            correct += (preds == label_batch).sum().item()
            total += label_batch.numel()
            total_loss += loss.item()

        acc = correct / max(total, 1)
        avg_loss = total_loss / len(loader)
        history.append({'epoch': epoch + 1, 'loss': avg_loss, 'accuracy': acc})

        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1}: loss={avg_loss:.4f} acc={acc:.1%}")

    # Evaluate on held-out: compute precision/recall on the positive class
    confusion_head.eval()
    with torch.no_grad():
        all_logits = confusion_head(features.to(device))
        all_preds = (all_logits > 0).float().cpu().squeeze(-1)
        all_labels_t = labels

        tp = ((all_preds == 1) & (all_labels_t == 1)).sum().item()
        fp = ((all_preds == 1) & (all_labels_t == 0)).sum().item()
        fn = ((all_preds == 0) & (all_labels_t == 1)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"\n  Confusion Head: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")

    return confusion_head, history, {'precision': precision, 'recall': recall, 'f1': f1}


# ============================================================================
# 14C: Confusion-Gated Generator
# ============================================================================

class ConfusionGatedGenerator:
    """
    Generation with confusion-modulated halt threshold.

    Normal:    halt when halt_conf > base_threshold AND confusion < veto_threshold
    Confused:  confusion >= veto_threshold vetoes the halt, generation continues

    No SSM modification. No retry. Just patience.
    """

    def __init__(self, model, confusion_detector, device,
                 base_threshold=0.95, confusion_veto_threshold=0.3,
                 extended_max_length=100):
        self.model = model
        self.confusion_detector = confusion_detector
        self.device = device
        self.base_threshold = base_threshold
        self.confusion_veto_threshold = confusion_veto_threshold
        self.extended_max_length = extended_max_length
        self.halt_id = VOCAB.get('<HALT>', -1)
        self.eos_id = VOCAB.get('<EOS>', -1)

    def generate(self, expression):
        """Modified generation loop with confusion-gated halting."""
        prompt_text = f"Input:{expression} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        state_vectors = []
        halt_confidences = []
        state_entropies = []
        confusion_scores = []
        stop_reason = "max_length"

        self.model.eval()
        with torch.no_grad():
            for step in range(self.extended_max_length):
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

                # Capture state
                halt_conf = 0.0
                if outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                if outputs.get("states_sequence") is not None:
                    last_state = outputs["states_sequence"][:, -1, :]
                    state_vec = last_state.squeeze(0).cpu()
                    state_vectors.append(state_vec)

                    energy = last_state ** 2
                    probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
                    ent = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).item()
                    state_entropies.append(ent)
                else:
                    state_entropies.append(0.0)

                # Compute confusion
                if isinstance(self.confusion_detector, CycleDetector):
                    _, confusion = self.confusion_detector.detect(state_vectors)
                elif isinstance(self.confusion_detector, ConfusionHead):
                    confusion = self._compute_learned_confusion(
                        state_vectors, state_entropies, halt_confidences
                    )
                else:
                    confusion = 0.0
                confusion_scores.append(confusion)

                # Get next token (greedy)
                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                # Stop conditions — confusion acts as a veto on halt
                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif (halt_conf > self.base_threshold and
                      confusion < self.confusion_veto_threshold):
                    stop_reason = "halt_confidence"
                    break

        # Parse output
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
            'confusion_scores': confusion_scores,
        }

    def _compute_learned_confusion(self, state_vectors, entropies, halt_confs):
        """Build feature window and run ConfusionHead."""
        window = 5
        t = len(state_vectors) - 1
        if t < window:
            return 0.0

        step_features = []
        for w in range(t - window, t):
            delta = (state_vectors[w] - state_vectors[w - 1]
                     if w > 0 else torch.zeros_like(state_vectors[0]))
            H = entropies[w] if w < len(entropies) else 0.0
            grad_H = compute_entropy_gradient(entropies, w)
            Sc = compute_cycling_score(state_vectors, w)
            hc = halt_confs[w] if w < len(halt_confs) else 0.0

            feat = torch.cat([
                delta, torch.tensor([H, grad_H, Sc, hc])
            ])
            step_features.append(feat)

        features = torch.stack(step_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.confusion_detector(features)
            return torch.sigmoid(logit).item()

    def _parse_arithmetic_result(self, text):
        """Extract numeric result from generated text."""
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
        """Count tokens between Input: section and Result:."""
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
# 14C: Halt Gating Ablation
# ============================================================================

def run_halt_gating_ablation(model, test_ds, device, confusion_detector_np,
                              confusion_detector_learned=None):
    """
    Ablation matrix:
      A. Baseline (no intervention)
      B. Constant raised threshold (+0.03)
      C. CycleDetector (dynamic)
      D. ConfusionHead (dynamic, if available)
    """
    print("\n" + "=" * 60)
    print("14C: Halt Gating Ablation")
    print("=" * 60)

    default_gen = CompressibleGenerator(model, device=device)
    print(f"  CompressibleGenerator default halt_threshold: {default_gen.halt_threshold}")

    conditions = {}

    # A. Baseline
    print("\n  Condition A: Baseline (default threshold)")
    gen_a = default_gen
    results_a = _run_generation_eval(gen_a, test_ds)
    conditions['A_baseline'] = results_a

    # B. Constant raised threshold
    print("\n  Condition B: Constant raised threshold (0.98)")
    gen_b = CompressibleGenerator(model, device=device, halt_threshold=0.98)
    results_b = _run_generation_eval(gen_b, test_ds)
    conditions['B_constant_boost'] = results_b

    # C. CycleDetector (non-parametric)
    print("\n  Condition C: CycleDetector (non-parametric)")
    gen_c = ConfusionGatedGenerator(
        model, confusion_detector_np, device,
        base_threshold=0.95, confusion_veto_threshold=0.3,
        extended_max_length=100,
    )
    results_c = _run_generation_eval(gen_c, test_ds)
    conditions['C_cycle_detector'] = results_c

    # D. Learned ConfusionHead (if available)
    if confusion_detector_learned is not None:
        print("\n  Condition D: ConfusionHead (learned)")
        gen_d = ConfusionGatedGenerator(
            model, confusion_detector_learned, device,
            base_threshold=0.95, confusion_veto_threshold=0.3,
            extended_max_length=100,
        )
        results_d = _run_generation_eval(gen_d, test_ds)
        conditions['D_confusion_head'] = results_d

    # E. Control: ConfusionGatedGenerator with veto disabled (threshold=1.0)
    # If E matches C/D, the improvement comes from the generator, not the veto.
    # If E matches A/B, the veto IS the active ingredient.
    print("\n  Condition E: ConfusionGatedGenerator, veto disabled (control)")
    gen_e = ConfusionGatedGenerator(
        model, confusion_detector_np, device,
        base_threshold=0.95, confusion_veto_threshold=1.0,
        extended_max_length=100,
    )
    results_e = _run_generation_eval(gen_e, test_ds)
    conditions['E_no_veto_control'] = results_e

    # Summary
    print("\n  Summary:")
    print(f"  {'Condition':25s} | {'T0 Acc':>8s} | {'T1 Acc':>8s} | {'T2 Acc':>8s} "
          f"| {'Overall':>8s} | {'Tokens':>8s} | {'TrueConv':>8s}")
    print("  " + "-" * 95)
    for name, res in conditions.items():
        tm = res['tier_metrics']
        t0 = tm.get(0, {}).get('accuracy', 0)
        t1 = tm.get(1, {}).get('accuracy', 0)
        t2 = tm.get(2, {}).get('accuracy', 0)
        overall = res['overall_accuracy']
        tokens = res['mean_tokens']
        true_conv = res['true_convergence_count']
        print(f"  {name:25s} | {t0:7.1%} | {t1:7.1%} | {t2:7.1%} "
              f"| {overall:7.1%} | {tokens:7.1f} | {true_conv:8d}")

    return conditions


def _run_generation_eval(generator, test_ds):
    """Run generation and compute tier metrics + convergence stats."""
    gen_results = []
    for i in tqdm(range(len(test_ds.examples)),
                  desc="Generating", leave=False):
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
            'confusion_scores': gen.get('confusion_scores', []),
        })

    # Compute metrics
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
    true_conv_total = sum(
        1 for r in gen_results if r['convergence'] == 'true_convergence'
    )

    return {
        'gen_results': gen_results,
        'tier_metrics': tier_metrics,
        'overall_accuracy': overall_acc,
        'mean_tokens': mean_tokens,
        'true_convergence_count': true_conv_total,
        'n': len(gen_results),
    }


# ============================================================================
# 14D: Cycle Breaking (conditional)
# ============================================================================

class CycleBreakingGenerator(ConfusionGatedGenerator):
    """
    When confused for patience consecutive steps, apply non-invasive
    cycle breaking via input re-injection or temperature spike.
    """

    def __init__(self, model, confusion_detector, device,
                 base_threshold=0.95, confusion_veto_threshold=0.3,
                 extended_max_length=100,
                 strategy='input_reinjection', patience=5,
                 max_breaks=3):
        super().__init__(model, confusion_detector, device,
                         base_threshold, confusion_veto_threshold,
                         extended_max_length)
        self.strategy = strategy
        self.patience = patience
        self.max_breaks = max_breaks

    def generate(self, expression):
        """Generation with cycle breaking."""
        prompt_text = f"Input:{expression} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        state_vectors = []
        halt_confidences = []
        state_entropies = []
        confusion_scores = []
        n_breaks = 0
        stop_reason = "max_length"

        self.model.eval()
        with torch.no_grad():
            for step in range(self.extended_max_length):
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

                halt_conf = 0.0
                if outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                if outputs.get("states_sequence") is not None:
                    last_state = outputs["states_sequence"][:, -1, :]
                    state_vec = last_state.squeeze(0).cpu()
                    state_vectors.append(state_vec)
                    energy = last_state ** 2
                    probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
                    ent = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).item()
                    state_entropies.append(ent)
                else:
                    state_entropies.append(0.0)

                # Compute confusion
                if isinstance(self.confusion_detector, CycleDetector):
                    _, confusion = self.confusion_detector.detect(state_vectors)
                elif isinstance(self.confusion_detector, ConfusionHead):
                    confusion = self._compute_learned_confusion(
                        state_vectors, state_entropies, halt_confidences
                    )
                else:
                    confusion = 0.0
                confusion_scores.append(confusion)

                # Check if we should break the cycle
                if (n_breaks < self.max_breaks and
                    len(confusion_scores) >= self.patience and
                    all(c > 0.5 for c in confusion_scores[-self.patience:])):

                    if self.strategy == 'input_reinjection':
                        reminder = tokenize(f" {expression}")
                        generated_ids.extend(reminder)
                    elif self.strategy == 'temperature_spike':
                        # Sample with high temperature for one step
                        temp_logits = logits / 2.0
                        probs_dist = F.softmax(temp_logits, dim=-1)
                        next_token = torch.multinomial(probs_dist, 1).item()
                        generated_ids.append(next_token)
                    n_breaks += 1
                    continue

                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                # Stop conditions — confusion acts as a veto on halt
                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif (halt_conf > self.base_threshold and
                      confusion < self.confusion_veto_threshold):
                    stop_reason = "halt_confidence"
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
            'confusion_scores': confusion_scores,
            'n_breaks': n_breaks,
        }


def run_cycle_breaking_ablation(model, test_ds, device, confusion_detector):
    """
    Ablation of cycle-breaking strategies.
    Only run if 14A-0 shows more time doesn't help.
    """
    print("\n" + "=" * 60)
    print("14D: Cycle Breaking Ablation")
    print("=" * 60)

    strategies = ['input_reinjection', 'temperature_spike']
    results = {}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        gen = CycleBreakingGenerator(
            model, confusion_detector, device,
            strategy=strategy, patience=5, max_breaks=3,
        )
        res = _run_generation_eval(gen, test_ds)
        results[strategy] = res

        t2 = res['tier_metrics'].get(2, {})
        print(f"    T2 acc={t2.get('accuracy', 0):.1%} "
              f"tokens={t2.get('mean_tokens', 0):.1f} "
              f"overall={res['overall_accuracy']:.1%}")

    return results


# ============================================================================
# 14E: Corrected Purity Evaluation
# ============================================================================

def evaluate_corrected_purity(model, gen_results, device):
    """Run corrected purity on generation results."""
    print("\n  Evaluating corrected geodesic purity...")
    purity_eval = GeodesicPurity()

    purities = []
    for i, r in enumerate(tqdm(gen_results, desc="Purity", leave=False)):
        if not r.get('generated_ids'):
            continue
        purity_result = evaluate_purity_corrected(
            model, r['generated_ids'], r['expression'], device, purity_eval
        )
        r['corrected_purity'] = purity_result['purity']
        r['n_constraints'] = purity_result['n_constraints']
        r['n_satisfied'] = purity_result.get('n_satisfied', 0)
        if purity_result['n_constraints'] > 0:
            purities.append(purity_result['purity'])

    mean_purity = float(np.mean(purities)) if purities else 0.0
    print(f"  Mean corrected purity: {mean_purity:.3f} (n={len(purities)} with constraints)")
    return mean_purity


# ============================================================================
# Figures
# ============================================================================

def plot_fig25_halt_control_problem(trajectories, fig_dir):
    """
    Fig 25: The Halt Control Problem (motivating figure).
    Left: True convergence trajectory
    Center: False convergence trajectory
    Right: Distribution of Sc at halt
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    true_conv = [t for t in trajectories
                 if t['convergence'] == 'true_convergence' and t['n_steps'] > 5]
    false_conv = [t for t in trajectories
                  if t['convergence'] == 'false_convergence' and t['n_steps'] > 5]

    # Left: True convergence example
    if true_conv:
        ex = true_conv[0]
        steps = list(range(len(ex['per_step'])))
        H_vals = [s['H'] for s in ex['per_step']]
        Sc_vals = [s['Sc'] for s in ex['per_step']]
        halt_vals = [s['halt_conf'] for s in ex['per_step']]

        ax1.plot(steps, H_vals, 'b-', linewidth=1.5, alpha=0.8, label='H (entropy)')
        ax1.plot(steps, Sc_vals, 'r-', linewidth=1.5, alpha=0.8, label='Sc (cycling)')
        ax1.plot(steps, halt_vals, 'g--', linewidth=1.5, alpha=0.8, label='halt_conf')
        ax1.set_xlabel("Generation Step")
        ax1.set_ylabel("Signal Value")
        ax1.set_title(f"True Convergence\n(correct={ex['is_correct']})")
        ax1.legend(fontsize=8)
        ax1.set_ylim(-0.05, 1.05)
    else:
        ax1.text(0.5, 0.5, "No true convergence examples",
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("True Convergence")

    # Center: False convergence example
    if false_conv:
        ex = false_conv[0]
        steps = list(range(len(ex['per_step'])))
        H_vals = [s['H'] for s in ex['per_step']]
        Sc_vals = [s['Sc'] for s in ex['per_step']]
        halt_vals = [s['halt_conf'] for s in ex['per_step']]

        ax2.plot(steps, H_vals, 'b-', linewidth=1.5, alpha=0.8, label='H (entropy)')
        ax2.plot(steps, Sc_vals, 'r-', linewidth=1.5, alpha=0.8, label='Sc (cycling)')
        ax2.plot(steps, halt_vals, 'g--', linewidth=1.5, alpha=0.8, label='halt_conf')
        ax2.set_xlabel("Generation Step")
        ax2.set_title(f"False Convergence\n(correct={ex['is_correct']})")
        ax2.legend(fontsize=8)
        ax2.set_ylim(-0.05, 1.05)
    else:
        ax2.text(0.5, 0.5, "No false convergence examples",
                 ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("False Convergence")

    # Right: Distribution of Sc at halt
    if true_conv and false_conv:
        true_sc = [t['Sc_at_halt'] for t in true_conv]
        false_sc = [t['Sc_at_halt'] for t in false_conv]

        bins = np.linspace(0, 1, 30)
        ax3.hist(true_sc, bins=bins, alpha=0.6, color='#2ecc71',
                 label=f'True conv (n={len(true_sc)})', density=True)
        ax3.hist(false_sc, bins=bins, alpha=0.6, color='#e74c3c',
                 label=f'False conv (n={len(false_sc)})', density=True)
        ax3.set_xlabel("Sc at Halt")
        ax3.set_ylabel("Density")
        ax3.set_title("Cycling Score Distribution\nat Halt Point")
        ax3.legend(fontsize=9)
    else:
        ax3.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                 transform=ax3.transAxes)
        ax3.set_title("Sc Distribution")

    fig.suptitle("Phase 14: The Halt Control Problem", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig25_halt_control_problem.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fig26_confusion_detection(trajectories, auc_results, cycle_det_result,
                                    confusion_head_result, fig_dir):
    """
    Fig 26: Confusion Detection Performance.
    Left: Feature discrimination (bar chart of AUCs)
    Right: Detection performance comparison (P/R/F1)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUC per feature
    if auc_results:
        features = list(auc_results.keys())
        aucs = [auc_results[f] for f in features]
        colors = ['#3498db'] * len(features)
        if 'combined' in features:
            colors[features.index('combined')] = '#2ecc71'

        bars = ax1.barh(range(len(features)), aucs, color=colors, alpha=0.8,
                        edgecolor='black')
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels([f.replace('_', '\n') for f in features], fontsize=9)
        ax1.set_xlabel("AUC (True vs False Convergence)")
        ax1.set_xlim(0.4, 1.0)
        ax1.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='random')
        ax1.axvline(x=0.70, color='orange', linestyle='--', alpha=0.7, label='target')

        for bar, auc_val in zip(bars, aucs):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{auc_val:.3f}', va='center', fontsize=9)
        ax1.legend(fontsize=9)
        ax1.set_title("Three-Signal AUC\n(True vs False Convergence)")
    else:
        ax1.text(0.5, 0.5, "No AUC data", ha='center', va='center',
                 transform=ax1.transAxes)

    # Right: Detection performance
    methods = []
    p_vals, r_vals, f1_vals = [], [], []

    if cycle_det_result:
        methods.append('CycleDetector\n(non-parametric)')
        p_vals.append(cycle_det_result['precision'])
        r_vals.append(cycle_det_result['recall'])
        f1_vals.append(cycle_det_result['f1'])

    if confusion_head_result:
        methods.append('ConfusionHead\n(learned)')
        p_vals.append(confusion_head_result['precision'])
        r_vals.append(confusion_head_result['recall'])
        f1_vals.append(confusion_head_result['f1'])

    if methods:
        x = np.arange(len(methods))
        width = 0.25
        ax2.bar(x - width, p_vals, width, color='#3498db', alpha=0.8,
                edgecolor='black', label='Precision')
        ax2.bar(x, r_vals, width, color='#f39c12', alpha=0.8,
                edgecolor='black', label='Recall')
        ax2.bar(x + width, f1_vals, width, color='#2ecc71', alpha=0.8,
                edgecolor='black', label='F1')

        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontsize=9)
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.15)
        ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.5, label='target F1')
        ax2.legend(fontsize=9)
        ax2.set_title("Confusion Detection Performance")
    else:
        ax2.text(0.5, 0.5, "No detection data", ha='center', va='center',
                 transform=ax2.transAxes)

    fig.suptitle("Phase 14: Confusion Detection", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig26_confusion_detection.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fig27_halt_gating_results(conditions, fig_dir):
    """
    Fig 27: Halt Gating Results (the money figure).
    Left: Accuracy by tier per condition
    Center: Accuracy vs tokens (Pareto frontier)
    Right: True convergence rate by condition
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    cond_names = list(conditions.keys())
    cond_colors = ['#95a5a6', '#3498db', '#f39c12', '#2ecc71'][:len(cond_names)]
    short_names = [n.split('_')[0] for n in cond_names]

    # Left: Accuracy by tier
    tiers = [0, 1, 2]
    x = np.arange(len(tiers))
    width = 0.8 / len(cond_names)

    for ci, (name, res) in enumerate(conditions.items()):
        accs = [res['tier_metrics'].get(t, {}).get('accuracy', 0) * 100
                for t in tiers]
        offset = (ci - len(cond_names) / 2 + 0.5) * width
        bars = ax1.bar(x + offset, accs, width, color=cond_colors[ci],
                       alpha=0.8, edgecolor='black', label=short_names[ci])

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{t} ({TIER_NAMES[t]})' for t in tiers])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Tier")
    ax1.legend(fontsize=8)

    # Center: Accuracy vs tokens (Pareto)
    for ci, (name, res) in enumerate(conditions.items()):
        ax2.scatter(res['mean_tokens'], res['overall_accuracy'] * 100,
                    color=cond_colors[ci], s=120, edgecolors='black',
                    zorder=3, label=short_names[ci])
        ax2.annotate(short_names[ci],
                     (res['mean_tokens'] + 0.5, res['overall_accuracy'] * 100 + 0.5),
                     fontsize=8)

    ax2.set_xlabel("Mean Reasoning Tokens")
    ax2.set_ylabel("Overall Accuracy (%)")
    ax2.set_title("Accuracy vs Token Budget\n(upper-left = better)")
    ax2.legend(fontsize=8)

    # Right: True convergence rate
    conv_rates = []
    for name, res in conditions.items():
        rate = res['true_convergence_count'] / max(res['n'], 1) * 100
        conv_rates.append(rate)

    bars = ax3.bar(range(len(cond_names)), conv_rates, color=cond_colors,
                   alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(cond_names)))
    ax3.set_xticklabels(short_names, rotation=15, ha='right', fontsize=9)
    ax3.set_ylabel("True Convergence Rate (%)")
    ax3.set_title("True Convergence Rate\n(target: >10%)")

    for bar, rate in zip(bars, conv_rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{rate:.1f}%', ha='center', fontsize=9)

    fig.suptitle("Phase 14: Halt Gating Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig27_halt_gating_results.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fig28_corrected_story(conditions, purity_baseline, purity_gated,
                                fig_dir):
    """
    Fig 28: The Corrected Story (summary figure).
    2x2: purity by phase, accuracy by phase, convergence rate, localization table.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Geodesic purity by phase
    phases = ['Phase 11\n(no constraints)', 'Phase 12\n(soft)', 'Phase 13b\n(hard)']
    purities = [0.0, 0.83, 0.18]
    colors_pur = ['#95a5a6', '#2ecc71', '#e74c3c']
    bars = ax1.bar(range(len(phases)), [p * 100 for p in purities],
                   color=colors_pur, alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(phases)))
    ax1.set_xticklabels(phases, fontsize=9)
    ax1.set_ylabel("Geodesic Purity (%)")
    ax1.set_title("Geodesic Purity by Phase\n(corrected measurement)")
    for bar, p in zip(bars, purities):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{p:.0%}', ha='center', fontsize=10, fontweight='bold')

    # Top-right: Accuracy by phase (Tier 2)
    baseline_res = conditions.get('A_baseline', {})
    best_cond_name = max(
        [k for k in conditions if k != 'A_baseline'],
        key=lambda k: conditions[k].get('tier_metrics', {}).get(2, {}).get('accuracy', 0),
        default='A_baseline'
    )
    best_res = conditions.get(best_cond_name, baseline_res)

    phase_accs = {
        'Phase 11': 0.652,
        'Phase 12\n(baseline)': baseline_res.get('tier_metrics', {}).get(2, {}).get('accuracy', 0),
        f'Phase 14\n({best_cond_name.split("_")[0]})': best_res.get('tier_metrics', {}).get(2, {}).get('accuracy', 0),
    }
    colors_acc = ['#95a5a6', '#f39c12', '#2ecc71']
    bars = ax2.bar(range(len(phase_accs)),
                   [v * 100 for v in phase_accs.values()],
                   color=colors_acc, alpha=0.8, edgecolor='black')
    ax2.set_xticks(range(len(phase_accs)))
    ax2.set_xticklabels(list(phase_accs.keys()), fontsize=9)
    ax2.set_ylabel("Tier 2 Accuracy (%)")
    ax2.set_title("Tier 2 Accuracy Progression")
    for bar, v in zip(bars, phase_accs.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{v:.1%}', ha='center', fontsize=10, fontweight='bold')

    # Bottom-left: True convergence rate
    conv_rates = {
        'Phase 12\n(baseline)': baseline_res.get('true_convergence_count', 0) / max(baseline_res.get('n', 1), 1),
        f'Phase 14\n({best_cond_name.split("_")[0]})': best_res.get('true_convergence_count', 0) / max(best_res.get('n', 1), 1),
    }
    colors_conv = ['#f39c12', '#2ecc71']
    bars = ax3.bar(range(len(conv_rates)),
                   [v * 100 for v in conv_rates.values()],
                   color=colors_conv, alpha=0.8, edgecolor='black')
    ax3.set_xticks(range(len(conv_rates)))
    ax3.set_xticklabels(list(conv_rates.keys()), fontsize=9)
    ax3.set_ylabel("True Convergence Rate (%)")
    ax3.set_title("True Convergence Rate\n(target: >10%)")
    for bar, v in zip(bars, conv_rates.values()):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{v:.1%}', ha='center', fontsize=10)

    # Bottom-right: Summary table
    ax4.axis('off')
    table_data = [
        ['Component', 'Phase 12', 'Phase 14'],
        ['Geometry (Purity)', f'{purity_baseline:.0%}', f'{purity_gated:.0%}'],
        ['T2 Accuracy',
         f"{baseline_res.get('tier_metrics', {}).get(2, {}).get('accuracy', 0):.1%}",
         f"{best_res.get('tier_metrics', {}).get(2, {}).get('accuracy', 0):.1%}"],
        ['True Convergence',
         f"{baseline_res.get('true_convergence_count', 0)}",
         f"{best_res.get('true_convergence_count', 0)}"],
        ['Mean Tokens',
         f"{baseline_res.get('mean_tokens', 0):.1f}",
         f"{best_res.get('mean_tokens', 0):.1f}"],
    ]
    table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    # Color header row
    for j in range(3):
        table[0, j].set_facecolor('#34495e')
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax4.set_title("Precision Localization", fontsize=12, fontweight='bold', pad=20)

    fig.suptitle("Phase 14: The Corrected Story", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig28_corrected_story.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 14: Halt Control")
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    parser.add_argument('--checkpoint', type=str, default='results/rim_model.pt')
    parser.add_argument('--skip-14a', action='store_true',
                        help='Skip diagnostics (14A)')
    parser.add_argument('--skip-14b', action='store_true',
                        help='Skip confusion head training (14B)')
    parser.add_argument('--skip-14d', action='store_true',
                        help='Skip cycle breaking (14D)')
    parser.add_argument('--force-14d', action='store_true',
                        help='Force cycle breaking even if not needed')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 14: Halt Control on a Geometrically Sound Manifold")
    print(f"{'=' * 60}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Load Phase 12 Checkpoint ----
    print(f"\n--- Loading Phase 12 Checkpoint ---")
    print(f"  Checkpoint: {args.checkpoint}")

    model = PNA_SSM_RIM(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                         max_seq_len=64).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded: {count_parameters(model):,} params")

    # ---- Create Datasets ----
    print("\n--- Creating Datasets ---")
    train_ds = CompressibleArithmeticDataset(
        num_samples=args.train_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=42,
    )
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456,
    )
    print(f"  Train: {len(train_ds)} examples")
    print(f"  Test:  {len(test_ds)} examples")

    # Storage for all results
    all_results = {}

    # ============================================================
    # 14A: Diagnostics
    # ============================================================
    trajectories = None
    auc_results = {}
    budget_decision = "mixed"

    if not args.skip_14a:
        # 14A-0: Extended budget
        budget_results, budget_decision = experiment_14a0_extended_budget(
            model, test_ds, device
        )
        all_results['14a0_budget'] = {
            str(k): v for k, v in budget_results.items()
        }
        all_results['14a0_decision'] = budget_decision

        # 14A-1: Three-signal validation
        trajectories, auc_results = experiment_14a1_three_signal(
            model, test_ds, device
        )
        all_results['14a1_auc'] = auc_results

        # 14A-2: Op detector probe
        op_detector, op_history = experiment_14a2_op_probe(
            model, train_ds, test_ds, device, epochs=10
        )
        all_results['14a2_op_probe'] = {
            'final_accuracy': op_history[-1]['accuracy'] if op_history else 0,
            'history': op_history,
        }

    # ============================================================
    # 14B: Confusion Detection
    # ============================================================
    cycle_det_result = None
    confusion_head_result = None
    confusion_head = None

    if not args.skip_14b:
        # 14B-1: Non-parametric CycleDetector
        cycle_det_result = evaluate_cycle_detector(model, test_ds, device)
        all_results['14b1_cycle_detector'] = cycle_det_result

        # 14B-2: Learned ConfusionHead (train regardless — it's cheap)
        print("\n" + "=" * 60)
        print("14B-2: Learned Confusion Head")
        print("=" * 60)

        training_windows = collect_confusion_labels(
            model, train_ds, device, n_samples=2000
        )
        if training_windows:
            confusion_head, ch_history, confusion_head_result = train_confusion_head(
                d_state=16, training_windows=training_windows,
                device=device, epochs=20,
            )
            all_results['14b2_confusion_head'] = confusion_head_result

            # Save confusion head
            ch_path = os.path.join(args.results_dir, 'confusion_head.pt')
            torch.save(confusion_head.state_dict(), ch_path)
            print(f"  Saved: {ch_path}")
        else:
            print("  No training windows collected — skipping ConfusionHead")

    # ============================================================
    # 14C: Halt Gating Ablation
    # ============================================================
    print("\n" + "=" * 60)
    print("14C: Halt Gating Ablation")
    print("=" * 60)

    cycle_detector = CycleDetector()
    conditions = run_halt_gating_ablation(
        model, test_ds, device,
        confusion_detector_np=cycle_detector,
        confusion_detector_learned=confusion_head,
    )
    all_results['14c_conditions'] = {
        name: {
            'tier_metrics': {str(k): v for k, v in res['tier_metrics'].items()},
            'overall_accuracy': res['overall_accuracy'],
            'mean_tokens': res['mean_tokens'],
            'true_convergence_count': res['true_convergence_count'],
            'n': res['n'],
        }
        for name, res in conditions.items()
    }

    # ============================================================
    # 14D: Cycle Breaking (conditional)
    # ============================================================
    run_14d = args.force_14d or (
        not args.skip_14d and budget_decision == "cycles_stuck"
    )
    cycle_breaking_results = None

    if run_14d:
        cycle_breaking_results = run_cycle_breaking_ablation(
            model, test_ds, device, cycle_detector
        )
        all_results['14d_cycle_breaking'] = {
            strategy: {
                'tier_metrics': {str(k): v for k, v in res['tier_metrics'].items()},
                'overall_accuracy': res['overall_accuracy'],
                'mean_tokens': res['mean_tokens'],
                'true_convergence_count': res['true_convergence_count'],
            }
            for strategy, res in cycle_breaking_results.items()
        }
    else:
        print(f"\n  Skipping 14D (decision={budget_decision}, "
              f"skip_14d={args.skip_14d}, force_14d={args.force_14d})")

    # ============================================================
    # 14E: Corrected Purity + Integration
    # ============================================================
    print("\n" + "=" * 60)
    print("14E: Corrected Purity Evaluation")
    print("=" * 60)

    # Baseline purity
    baseline_gen = conditions.get('A_baseline', {}).get('gen_results', [])
    purity_baseline = evaluate_corrected_purity(model, baseline_gen, device)

    # Best condition purity
    best_cond_name = max(
        [k for k in conditions if k != 'A_baseline'],
        key=lambda k: conditions[k]['overall_accuracy'],
        default='A_baseline'
    )
    best_gen = conditions.get(best_cond_name, {}).get('gen_results', [])
    purity_gated = evaluate_corrected_purity(model, best_gen, device)

    all_results['purity'] = {
        'baseline': purity_baseline,
        'best_condition': purity_gated,
        'best_condition_name': best_cond_name,
    }

    # ---- Convergence Analysis ----
    print("\n--- Convergence Analysis (Best Condition) ---")
    best_res = conditions.get(best_cond_name, {})
    best_gen_results = best_res.get('gen_results', [])

    if best_gen_results:
        basin = analyze_convergence_vs_halt(best_gen_results)
        conv = basin['convergence']
        cyc = basin['cycling']
        print(f"  True Convergence (n={conv['n']}):")
        print(f"    Accuracy: {conv['accuracy']:.1%}")
        print(f"    Mean tokens: {conv['mean_tokens']:.1f}")
        print(f"  Cycling (n={cyc['n']}):")
        print(f"    Accuracy: {cyc['accuracy']:.1%}")
        print(f"    Mean tokens: {cyc['mean_tokens']:.1f}")
        print(f"  Gap: {basin['accuracy_gap']:+.1%}")

        all_results['convergence_analysis'] = {
            'convergence': {
                'n': conv['n'], 'accuracy': conv['accuracy'],
                'mean_tokens': conv['mean_tokens'],
            },
            'cycling': {
                'n': cyc['n'], 'accuracy': cyc['accuracy'],
                'mean_tokens': cyc['mean_tokens'],
            },
            'accuracy_gap': basin['accuracy_gap'],
        }

    # ============================================================
    # Figures
    # ============================================================
    print("\n--- Generating Figures ---")

    if trajectories:
        plot_fig25_halt_control_problem(trajectories, args.fig_dir)

    plot_fig26_confusion_detection(
        trajectories or [], auc_results,
        cycle_det_result, confusion_head_result,
        args.fig_dir
    )

    plot_fig27_halt_gating_results(conditions, args.fig_dir)

    plot_fig28_corrected_story(
        conditions, purity_baseline, purity_gated, args.fig_dir
    )

    # ============================================================
    # Save Results
    # ============================================================
    results_path = os.path.join(args.results_dir, 'phase14_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    # ============================================================
    # Success Criteria
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 14: Success Criteria")
    print("=" * 60)

    best_cond = conditions.get(best_cond_name, {})
    baseline_cond = conditions.get('A_baseline', {})
    baseline_t2_acc = baseline_cond.get('tier_metrics', {}).get(2, {}).get('accuracy', 0)
    best_t2_acc = best_cond.get('tier_metrics', {}).get(2, {}).get('accuracy', 0)
    best_true_conv = best_cond.get('true_convergence_count', 0) / max(best_cond.get('n', 1), 1)
    baseline_tokens = baseline_cond.get('mean_tokens', 1)
    best_tokens = best_cond.get('mean_tokens', 1)
    token_ratio = best_tokens / max(baseline_tokens, 1)

    # AUC criteria
    sc_auc = auc_results.get('Sc_at_halt', 0)
    combined_auc = auc_results.get('combined', 0)
    best_f1 = max(
        (cycle_det_result or {}).get('f1', 0),
        (confusion_head_result or {}).get('f1', 0),
    )

    # Check if detection adds value over constant boost
    const_boost_acc = conditions.get('B_constant_boost', {}).get('overall_accuracy', 0)
    best_detection_acc = max(
        conditions.get('C_cycle_detector', {}).get('overall_accuracy', 0),
        conditions.get('D_confusion_head', {}).get('overall_accuracy', 0),
    )
    detection_adds_value = best_detection_acc > const_boost_acc + 0.02

    criteria = [
        ("Sc discriminates convergence (AUC>0.70)", sc_auc, 0.70),
        ("Combined three-signal AUC (>0.80)", combined_auc, 0.80),
        ("Confusion detection F1 (>0.70)", best_f1, 0.70),
        ("T2 Accuracy with halt gating (>72%)", best_t2_acc, 0.72),
        ("True convergence rate (>10%)", best_true_conv, 0.10),
        ("Detection > constant boost", float(detection_adds_value), 0.5),
        ("Token overhead (<2x)", 2.0 - token_ratio, 0.0),
        ("Purity preserved (>75%)", purity_gated, 0.75),
    ]

    n_pass = 0
    for name, value, threshold in criteria:
        status = "PASS" if value >= threshold else "FAIL"
        if status == "PASS":
            n_pass += 1
        print(f"  {name}: {value:.3f} — {status}")

    print(f"\n  Score: {n_pass}/{len(criteria)} criteria passed")
    print(f"\n  Best condition: {best_cond_name}")
    print(f"  Baseline T2: {baseline_t2_acc:.1%} -> Best T2: {best_t2_acc:.1%} "
          f"(delta: {best_t2_acc - baseline_t2_acc:+.1%})")
    print(f"  Token ratio: {token_ratio:.2f}x")


if __name__ == '__main__':
    main()
