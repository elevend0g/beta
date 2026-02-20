"""
Phase 15: Epistemic Controller

Builds and tests a closed-loop inference controller that uses the three-signal
framework (Sc, grad_H, H) to monitor reasoning trajectory health and intervene
when the model is stuck. Target: break through the T2 ceiling at inference time,
without retraining.

Phase 14: Proved halt veto works (via confusion detection)
Phase 15: Proves whether PERTURBATION adds value beyond halt veto
          Target: the residual error that halt control can't touch

All experiments use the Phase 12 checkpoint (results/rim_model.pt) with
frozen backbone. No SSM recurrence modification. All interventions are token-level.
"""

import os
import sys
import re
import json
import math
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, count_parameters
from train import get_device
from compressible_task import (
    CompressibleArithmeticDataset, CompressibleGenerator,
    detect_oscillation, classify_convergence,
    TIER_NAMES, TIER_COLORS,
)
from rule_initialization import (
    PNA_SSM_RIM, GeodesicPurity, RIMDatasetWrapper,
    evaluate_purity_corrected, find_op_token_positions,
)
from phase14_halt_control import (
    compute_state_entropy, compute_entropy_gradient, compute_cycling_score,
    CycleDetector, ConfusionHead, ConfusionGatedGenerator,
    _run_generation_eval,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ============================================================================
# Regime Classifier
# ============================================================================

class RegimeClassifier:
    """
    Classifies each generation step into one of four regimes
    using validated Phase 14 signals.

    Thresholds calibrated from Phase 14A-1:
      True convergence:  Sc=0.025, grad_H=-0.082
      False convergence: Sc=0.333, grad_H=+0.008
    """
    SC_THRESHOLD = 0.15       # Midpoint between 0.025 and 0.333
    GRAD_H_THRESHOLD = -0.03  # Midpoint between -0.082 and +0.008
    H_HIGH = 3.2              # Above both true (3.135) and false (3.037)

    def classify(self, Sc, grad_H, H):
        if grad_H < self.GRAD_H_THRESHOLD and Sc < self.SC_THRESHOLD:
            return 'CONVERGING'   # Entropy falling, not cycling
        elif Sc > self.SC_THRESHOLD and abs(grad_H) < 0.02:
            return 'ORBITING'     # Cycling, entropy flat
        elif H > self.H_HIGH and abs(grad_H) < 0.02:
            return 'DIFFUSING'    # High entropy, flat
        else:
            return 'PROGRESSING'  # Making progress, not yet converged


# ============================================================================
# Epistemic Controller
# ============================================================================

class EpistemicController:
    """
    Closed-loop inference controller using three-signal regime classification.

    Supports:
      - Halt veto (suppress halt when not CONVERGING)
      - Perturbation strategies when ORBITING
    """

    def __init__(self, window=8, max_interventions=3,
                 perturbation_strategy=None, patience=5,
                 halt_threshold=0.95):
        self.classifier = RegimeClassifier()
        self.window = window
        self.max_interventions = max_interventions
        self.perturbation_strategy = perturbation_strategy
        self.patience = patience
        self.halt_threshold = halt_threshold

        # Per-generation state (reset in reset())
        self.reset()

    def reset(self):
        self.state_vectors = []
        self.entropies = []
        self.halt_confidences = []
        self.regimes = []
        self.signals_log = []
        self.orbit_count = 0
        self.diffuse_count = 0
        self.n_interventions = 0
        self.temp_steps_remaining = 0
        self.boost_steps_remaining = 0

    def _compute_signals(self):
        """Compute three signals from current trajectory."""
        t = len(self.state_vectors) - 1
        H = self.entropies[t] if t < len(self.entropies) else 0.0
        grad_H = compute_entropy_gradient(self.entropies, t, window=3)
        Sc = compute_cycling_score(self.state_vectors, t, k_values=[2, 4, 8])
        return {'H': H, 'grad_H': grad_H, 'Sc': Sc}

    def step(self, state_vec, halt_conf, entropy):
        """
        Process one generation step.

        Returns dict with:
          action: 'CONTINUE', 'HALT', or 'PERTURB'
          regime: current regime classification
          signals: {H, grad_H, Sc}
          perturbation: strategy name if action=='PERTURB', else None
        """
        self.state_vectors.append(state_vec)
        self.entropies.append(entropy)
        self.halt_confidences.append(halt_conf)

        signals = self._compute_signals()
        regime = self.classifier.classify(
            signals['Sc'], signals['grad_H'], signals['H']
        )
        self.regimes.append(regime)
        self.signals_log.append(signals)

        # Track consecutive regime counts
        if regime == 'ORBITING':
            self.orbit_count += 1
            self.diffuse_count = 0
        elif regime == 'DIFFUSING':
            self.diffuse_count += 1
            self.orbit_count = 0
        else:
            self.orbit_count = 0
            self.diffuse_count = 0

        result = {
            'action': 'CONTINUE',
            'regime': regime,
            'signals': signals,
            'perturbation': None,
        }

        # Halt decision
        if halt_conf > self.halt_threshold:
            if regime == 'CONVERGING':
                result['action'] = 'HALT'
            else:
                result['action'] = 'CONTINUE'  # Veto

        # Perturbation decision (only if not halting and strategy is set)
        if (result['action'] == 'CONTINUE' and
                self.perturbation_strategy is not None and
                self.n_interventions < self.max_interventions):

            if regime == 'ORBITING' and self.orbit_count >= self.patience:
                result['action'] = 'PERTURB'
                result['perturbation'] = self.perturbation_strategy
                self.orbit_count = 0
                self.n_interventions += 1

            elif regime == 'DIFFUSING' and self.diffuse_count >= self.patience:
                result['action'] = 'PERTURB'
                result['perturbation'] = self.perturbation_strategy
                self.diffuse_count = 0
                self.n_interventions += 1

        return result


# ============================================================================
# Controlled Generator
# ============================================================================

class ControlledGenerator:
    """
    Generation loop driven by the EpistemicController.

    Supports multiple perturbation strategies:
      - expression_reinjection: re-append the expression
      - temperature_spike: sample at T=2.0 for 2 steps
      - separator_injection: inject " Check: " marker
      - logit_boosting: boost digit and Result: logits
    """

    def __init__(self, model, controller, device, max_length=200):
        self.model = model
        self.controller = controller
        self.device = device
        self.max_length = max_length
        self.halt_id = VOCAB.get('<HALT>', -1)
        self.eos_id = VOCAB.get('<EOS>', -1)

    def generate(self, expression):
        """Generate with epistemic control."""
        self.controller.reset()

        prompt_text = f"Input:{expression} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        halt_confidences = []
        state_entropies = []
        state_vectors = []
        stop_reason = "max_length"
        confusion_scores = []  # compatibility with _run_generation_eval

        temp_steps_remaining = 0
        boost_steps_remaining = 0

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

                # Halt confidence
                halt_conf = 0.0
                if outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                # State entropy
                state_vec = None
                entropy = 0.0
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

                # Controller step
                if state_vec is not None:
                    ctrl = self.controller.step(state_vec, halt_conf, entropy)
                else:
                    ctrl = {'action': 'CONTINUE', 'regime': 'PROGRESSING',
                            'signals': {'H': 0, 'grad_H': 0, 'Sc': 0},
                            'perturbation': None}

                confusion_scores.append(ctrl['signals']['Sc'])

                # Handle active perturbation effects from previous steps
                if temp_steps_remaining > 0:
                    probs_dist = F.softmax(logits / 2.0, dim=-1)
                    next_token = torch.multinomial(probs_dist, 1).item()
                    generated_ids.append(next_token)
                    temp_steps_remaining -= 1
                    if next_token == self.halt_id:
                        stop_reason = "halt_token"
                        break
                    elif next_token == self.eos_id:
                        stop_reason = "eos"
                        break
                    continue

                if boost_steps_remaining > 0:
                    boosted = logits.clone()
                    digit_ids = [VOCAB.get(str(d), -1) for d in range(10)]
                    result_id = VOCAB.get('Result:', -1)
                    for did in digit_ids:
                        if did >= 0:
                            boosted[0, did] += 3.0
                    if result_id >= 0:
                        boosted[0, result_id] += 5.0
                    next_token = boosted.argmax(dim=-1).item()
                    generated_ids.append(next_token)
                    boost_steps_remaining -= 1
                    if next_token == self.halt_id:
                        stop_reason = "halt_token"
                        break
                    elif next_token == self.eos_id:
                        stop_reason = "eos"
                        break
                    continue

                # Process controller action
                if ctrl['action'] == 'HALT':
                    # Append the greedy token first (consistent with CompressibleGenerator)
                    next_token = logits.argmax(dim=-1).item()
                    generated_ids.append(next_token)
                    stop_reason = "halt_confidence"
                    break

                elif ctrl['action'] == 'PERTURB':
                    strategy = ctrl['perturbation']

                    if strategy == 'expression_reinjection':
                        reminder_tokens = tokenize(f" Input:{expression} Result:")
                        generated_ids.extend(reminder_tokens)

                    elif strategy == 'temperature_spike':
                        temp_steps_remaining = 2
                        probs_dist = F.softmax(logits / 2.0, dim=-1)
                        next_token = torch.multinomial(probs_dist, 1).item()
                        generated_ids.append(next_token)
                        temp_steps_remaining -= 1
                        if next_token == self.halt_id:
                            stop_reason = "halt_token"
                            break
                        elif next_token == self.eos_id:
                            stop_reason = "eos"
                            break
                        continue

                    elif strategy == 'separator_injection':
                        separator = tokenize(" Check: ")
                        generated_ids.extend(separator)

                    elif strategy == 'logit_boosting':
                        boost_steps_remaining = 3
                        boosted = logits.clone()
                        digit_ids = [VOCAB.get(str(d), -1) for d in range(10)]
                        result_id = VOCAB.get('Result:', -1)
                        for did in digit_ids:
                            if did >= 0:
                                boosted[0, did] += 3.0
                        if result_id >= 0:
                            boosted[0, result_id] += 5.0
                        next_token = boosted.argmax(dim=-1).item()
                        generated_ids.append(next_token)
                        boost_steps_remaining -= 1
                        if next_token == self.halt_id:
                            stop_reason = "halt_token"
                            break
                        elif next_token == self.eos_id:
                            stop_reason = "eos"
                            break
                        continue
                    # For reinjection/separator: fall through to normal token generation

                # Normal generation: greedy decode
                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
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
            'controller_log': {
                'regimes': list(self.controller.regimes),
                'signals': list(self.controller.signals_log),
                'n_interventions': self.controller.n_interventions,
            },
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
# Best-of-N Generator
# ============================================================================

class BestOfNGenerator:
    """Generate N completions and select the best trajectory by signal quality."""

    def __init__(self, model, controller_factory, device, N=5,
                 temperature=0.8, max_length=200):
        self.model = model
        self.controller_factory = controller_factory
        self.device = device
        self.N = N
        self.temperature = temperature
        self.max_length = max_length
        self.halt_id = VOCAB.get('<HALT>', -1)
        self.eos_id = VOCAB.get('<EOS>', -1)

    def generate(self, expression):
        """Generate N candidates and return the best one."""
        candidates = []

        for trial in range(self.N):
            controller = self.controller_factory()
            controller.reset()

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

                    halt_conf = 0.0
                    if outputs.get("halt_confidence") is not None:
                        halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                    halt_confidences.append(halt_conf)

                    state_vec = None
                    entropy = 0.0
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

                    if state_vec is not None:
                        ctrl = controller.step(state_vec, halt_conf, entropy)
                    else:
                        ctrl = {'action': 'CONTINUE', 'regime': 'PROGRESSING',
                                'signals': {'H': 0, 'grad_H': 0, 'Sc': 0},
                                'perturbation': None}

                    if ctrl['action'] == 'HALT':
                        next_token = logits.argmax(dim=-1).item()
                        generated_ids.append(next_token)
                        stop_reason = "halt_confidence"
                        break

                    # Sample with temperature for diversity
                    probs_dist = F.softmax(logits / self.temperature, dim=-1)
                    next_token = torch.multinomial(probs_dist, 1).item()
                    generated_ids.append(next_token)

                    if next_token == self.halt_id:
                        stop_reason = "halt_token"
                        break
                    elif next_token == self.eos_id:
                        stop_reason = "eos"
                        break

            # Score this candidate
            final_signals = controller.signals_log[-1] if controller.signals_log else {
                'Sc': 1.0, 'grad_H': 0.0, 'H': 5.0
            }
            # Prefer: low Sc (not cycling), negative grad_H (converging), low H
            score = -final_signals['Sc'] + final_signals['grad_H'] - 0.1 * final_signals['H']

            text = detokenize(generated_ids)
            parsed_answer = self._parse_arithmetic_result(text)
            reasoning_tokens = self._count_reasoning_tokens(generated_ids)

            candidates.append((score, {
                'parsed_answer': parsed_answer,
                'generated_text': text,
                'generated_ids': generated_ids,
                'reasoning_tokens': reasoning_tokens,
                'total_tokens': len(generated_ids),
                'stop_reason': stop_reason,
                'state_vectors': state_vectors,
                'halt_confidences': halt_confidences,
                'state_entropies': state_entropies,
                'confusion_scores': [s['Sc'] for s in controller.signals_log],
                'controller_log': {
                    'regimes': list(controller.regimes),
                    'signals': list(controller.signals_log),
                    'n_interventions': controller.n_interventions,
                },
                'best_of_n_score': score,
                'n_candidates': self.N,
            }))

        # Return highest-scoring candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

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
# 15A: Regime Classification Diagnostic
# ============================================================================

def experiment_15a_regime_diagnostic(model, test_ds, device):
    """
    Classify regimes at the halt point and at final step.
    Report regime distribution for correct vs incorrect examples.
    """
    print("\n" + "=" * 60)
    print("15A: Regime Classification Diagnostic")
    print("=" * 60)

    classifier = RegimeClassifier()

    # Generate with no halt (thresh=1.0) to get full trajectories
    generator = CompressibleGenerator(
        model, device=device, halt_threshold=1.0
    )
    # Also generate with default halt to know where halt WOULD fire
    generator_default = CompressibleGenerator(
        model, device=device, halt_threshold=0.95
    )

    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="15A"):
        ex = test_ds.examples[i]

        # Full trajectory (no halt)
        gen_full = generator.generate(ex['expression'])
        # Default halt trajectory (to find halt point)
        gen_halt = generator_default.generate(ex['expression'])

        halt_step = len(gen_halt['state_vectors']) - 1
        is_correct = (gen_full['parsed_answer'] == ex['answer'])
        is_correct_halt = (gen_halt['parsed_answer'] == ex['answer'])

        states = gen_full['state_vectors']
        entropies = gen_full['state_entropies']

        # Classify at each step
        per_step_regimes = []
        for t in range(len(states)):
            H = entropies[t] if t < len(entropies) else 0.0
            grad_H = compute_entropy_gradient(entropies, t, window=3)
            Sc = compute_cycling_score(states, t, k_values=[2, 4, 8])
            regime = classifier.classify(Sc, grad_H, H)
            per_step_regimes.append(regime)

        # Regime at halt point (clamped to trajectory length)
        halt_idx = min(halt_step, len(per_step_regimes) - 1)
        regime_at_halt = per_step_regimes[halt_idx] if per_step_regimes else 'UNKNOWN'

        # Final regime
        regime_final = per_step_regimes[-1] if per_step_regimes else 'UNKNOWN'

        results.append({
            'tier': ex['tier'],
            'is_correct_full': is_correct,
            'is_correct_halt': is_correct_halt,
            'halt_step': halt_step,
            'total_steps': len(states),
            'regime_at_halt': regime_at_halt,
            'regime_final': regime_final,
            'per_step_regimes': per_step_regimes,
        })

    # Analysis: regime at halt point vs correctness
    print("\n  Regime at halt point (all examples):")
    print(f"  {'Regime':15s} | {'n':>5s} | {'Acc(halt)':>10s} | {'Acc(full)':>10s} | {'Action':>25s}")
    print("  " + "-" * 75)

    regime_names = ['CONVERGING', 'ORBITING', 'DIFFUSING', 'PROGRESSING']
    regime_stats = {}
    for regime in regime_names:
        subset = [r for r in results if r['regime_at_halt'] == regime]
        n = len(subset)
        if n == 0:
            continue
        acc_halt = np.mean([r['is_correct_halt'] for r in subset])
        acc_full = np.mean([r['is_correct_full'] for r in subset])

        actions = {
            'CONVERGING': 'Allow halt',
            'ORBITING': 'Veto + perturb',
            'DIFFUSING': 'Veto + concentrate',
            'PROGRESSING': 'Veto + wait',
        }
        print(f"  {regime:15s} | {n:5d} | {acc_halt:10.1%} | {acc_full:10.1%} | {actions[regime]:>25s}")
        regime_stats[regime] = {
            'n': n, 'accuracy_halt': float(acc_halt),
            'accuracy_full': float(acc_full),
        }

    # Focus on incorrect examples at halt
    incorrect = [r for r in results if not r['is_correct_halt']]
    print(f"\n  Incorrect examples at halt (n={len(incorrect)}):")
    for regime in regime_names:
        n = sum(1 for r in incorrect if r['regime_at_halt'] == regime)
        pct = n / max(len(incorrect), 1)
        print(f"    {regime:15s}: {n:4d} ({pct:.1%})")

    # Decision logic
    n_incorrect = len(incorrect)
    n_orbiting = sum(1 for r in incorrect if r['regime_at_halt'] == 'ORBITING')
    n_converging = sum(1 for r in incorrect if r['regime_at_halt'] == 'CONVERGING')
    n_progressing = sum(1 for r in incorrect if r['regime_at_halt'] == 'PROGRESSING')

    if n_incorrect > 0:
        if n_orbiting / n_incorrect > 0.5:
            decision = "perturbation_high_potential"
            print(f"\n  Decision: >50% incorrect are ORBITING -> perturbation has high potential")
        elif n_converging / n_incorrect > 0.5:
            decision = "wrong_basin"
            print(f"\n  Decision: >50% incorrect are CONVERGING -> converging to wrong answers")
        elif n_progressing / n_incorrect > 0.5:
            decision = "needs_more_time"
            print(f"\n  Decision: >50% incorrect are PROGRESSING -> halt veto may suffice")
        else:
            decision = "mixed"
            print(f"\n  Decision: Mixed regime distribution")
    else:
        decision = "no_errors"
        print(f"\n  Decision: No incorrect examples")

    return results, regime_stats, decision


# ============================================================================
# 15B: Halt Veto Baseline
# ============================================================================

def experiment_15b_halt_veto_baseline(model, test_ds, device):
    """
    Halt veto using regime classifier (no perturbation).
    Validates EpistemicController integration; should match Phase 14 veto result.
    """
    print("\n" + "=" * 60)
    print("15B: Halt Veto Baseline (Regime-Based)")
    print("=" * 60)

    controller = EpistemicController(
        window=8, max_interventions=0,
        perturbation_strategy=None, patience=5,
        halt_threshold=0.95,
    )
    gen = ControlledGenerator(model, controller, device)
    results = _run_generation_eval(gen, test_ds)

    tm = results['tier_metrics']
    print(f"\n  Results:")
    for tier in [0, 1, 2]:
        t = tm.get(tier, {})
        print(f"    T{tier}: acc={t.get('accuracy', 0):.1%} "
              f"tokens={t.get('mean_tokens', 0):.1f}")
    print(f"    Overall: {results['overall_accuracy']:.1%}")

    return results


# ============================================================================
# 15C: Perturbation Ablation
# ============================================================================

def experiment_15c_perturbation_ablation(model, test_ds, device):
    """
    Test five perturbation strategies, each with halt veto active.
    Sweep patience and max_interventions.
    """
    print("\n" + "=" * 60)
    print("15C: Perturbation Ablation")
    print("=" * 60)

    strategies = [
        'expression_reinjection',
        'temperature_spike',
        'separator_injection',
        'logit_boosting',
    ]

    # Ablation: patience x max_interventions
    patience_values = [3, 5, 8]
    max_int_values = [1, 2, 3]

    all_conditions = {}

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")
        best_acc = 0
        best_config = None

        for pat in patience_values:
            for mi in max_int_values:
                controller = EpistemicController(
                    window=8, max_interventions=mi,
                    perturbation_strategy=strategy,
                    patience=pat, halt_threshold=0.95,
                )
                gen = ControlledGenerator(model, controller, device)
                res = _run_generation_eval(gen, test_ds)

                t2_acc = res['tier_metrics'].get(2, {}).get('accuracy', 0)
                t2_tok = res['tier_metrics'].get(2, {}).get('mean_tokens', 0)
                overall = res['overall_accuracy']

                key = f"{strategy}_p{pat}_m{mi}"
                all_conditions[key] = res

                print(f"    p={pat} m={mi}: T2={t2_acc:.1%} overall={overall:.1%} "
                      f"tokens={t2_tok:.1f}")

                if t2_acc > best_acc:
                    best_acc = t2_acc
                    best_config = (pat, mi)

        if best_config:
            print(f"    Best: patience={best_config[0]}, max_int={best_config[1]}, "
                  f"T2={best_acc:.1%}")

    # Best-of-N
    print(f"\n  Strategy: best_of_5")

    def make_controller():
        return EpistemicController(
            window=8, max_interventions=0,
            perturbation_strategy=None, patience=5,
            halt_threshold=0.95,
        )

    gen_bon = BestOfNGenerator(
        model, make_controller, device, N=5,
        temperature=0.8, max_length=200,
    )
    res_bon = _run_generation_eval(gen_bon, test_ds)
    t2_acc = res_bon['tier_metrics'].get(2, {}).get('accuracy', 0)
    overall = res_bon['overall_accuracy']
    t2_tok = res_bon['tier_metrics'].get(2, {}).get('mean_tokens', 0)
    print(f"    best_of_5: T2={t2_acc:.1%} overall={overall:.1%} tokens={t2_tok:.1f}")
    all_conditions['best_of_5'] = res_bon

    return all_conditions


# ============================================================================
# 15D: Full Controller
# ============================================================================

def experiment_15d_full_controller(model, test_ds, device, best_strategy,
                                    best_patience, best_max_int):
    """
    Combine the best perturbation strategy with the halt veto.
    """
    print("\n" + "=" * 60)
    print("15D: Full Controller")
    print("=" * 60)
    print(f"  Strategy: {best_strategy}")
    print(f"  Patience: {best_patience}")
    print(f"  Max interventions: {best_max_int}")

    controller = EpistemicController(
        window=8, max_interventions=best_max_int,
        perturbation_strategy=best_strategy,
        patience=best_patience, halt_threshold=0.95,
    )
    gen = ControlledGenerator(model, controller, device)
    results = _run_generation_eval(gen, test_ds)

    tm = results['tier_metrics']
    print(f"\n  Results:")
    for tier in [0, 1, 2]:
        t = tm.get(tier, {})
        print(f"    T{tier}: acc={t.get('accuracy', 0):.1%} "
              f"tokens={t.get('mean_tokens', 0):.1f}")
    print(f"    Overall: {results['overall_accuracy']:.1%}")

    return results


# ============================================================================
# 15E: Failure Analysis
# ============================================================================

def experiment_15e_failure_analysis(full_controller_results, max_interventions):
    """Classify failure modes of remaining incorrect examples."""
    print("\n" + "=" * 60)
    print("15E: Failure Analysis")
    print("=" * 60)

    gen_results = full_controller_results['gen_results']
    incorrect = [r for r in gen_results if not r['is_correct']]

    failure_modes = defaultdict(list)

    for r in incorrect:
        log = r.get('controller_log', {})
        regimes = log.get('regimes', [])
        signals = log.get('signals', [])
        n_int = log.get('n_interventions', 0)

        final_regime = regimes[-1] if regimes else 'UNKNOWN'
        final_Sc = signals[-1]['Sc'] if signals else 1.0

        if final_regime == 'CONVERGING' and final_Sc < 0.1:
            mode = 'WRONG_BASIN'
        elif final_regime == 'ORBITING':
            mode = 'PERSISTENT_ORBIT'
        elif n_int >= max_interventions:
            mode = 'EXHAUSTED_BUDGET'
        else:
            mode = 'COMPUTATION_ERROR'

        failure_modes[mode].append(r)

    print(f"\n  Total incorrect: {len(incorrect)}")
    print(f"  {'Failure Mode':25s} | {'n':>5s} | {'% errors':>10s} | {'Addressable by':>25s}")
    print("  " + "-" * 75)

    addressable_by = {
        'WRONG_BASIN': '14F (basin deepening)',
        'PERSISTENT_ORBIT': 'Stronger perturbation / more N',
        'EXHAUSTED_BUDGET': 'Higher max_interventions',
        'COMPUTATION_ERROR': 'More training / larger model',
    }

    failure_summary = {}
    for mode in ['WRONG_BASIN', 'PERSISTENT_ORBIT', 'EXHAUSTED_BUDGET', 'COMPUTATION_ERROR']:
        n = len(failure_modes[mode])
        pct = n / max(len(incorrect), 1)
        print(f"  {mode:25s} | {n:5d} | {pct:10.1%} | {addressable_by[mode]:>25s}")
        failure_summary[mode] = {'n': n, 'pct': float(pct)}

    return failure_summary


# ============================================================================
# Figures
# ============================================================================

def plot_fig29_regime_distribution(regime_results, fig_dir):
    """Regime distribution at halt point."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    regime_names = ['CONVERGING', 'ORBITING', 'DIFFUSING', 'PROGRESSING']
    colors = ['#27ae60', '#e74c3c', '#3498db', '#f39c12']

    # All examples
    counts = [sum(1 for r in regime_results if r['regime_at_halt'] == rn)
              for rn in regime_names]
    ax1.bar(regime_names, counts, color=colors)
    ax1.set_title("Regime at Halt Point (All)", fontweight='bold')
    ax1.set_ylabel("Count")
    ax1.tick_params(axis='x', rotation=30)

    # Incorrect only
    incorrect = [r for r in regime_results if not r['is_correct_halt']]
    counts_inc = [sum(1 for r in incorrect if r['regime_at_halt'] == rn)
                  for rn in regime_names]
    ax2.bar(regime_names, counts_inc, color=colors)
    ax2.set_title("Regime at Halt Point (Incorrect Only)", fontweight='bold')
    ax2.set_ylabel("Count")
    ax2.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig29_regime_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fig30_perturbation_comparison(conditions_15c, baseline_15b, fig_dir):
    """Compare perturbation strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    baseline_t2 = baseline_15b['tier_metrics'].get(2, {}).get('accuracy', 0)

    # Collect best config for each strategy
    strategy_best = {}
    for key, res in conditions_15c.items():
        parts = key.rsplit('_p', 1)
        if len(parts) == 2:
            strategy = parts[0]
        else:
            strategy = key
        t2_acc = res['tier_metrics'].get(2, {}).get('accuracy', 0)
        if strategy not in strategy_best or t2_acc > strategy_best[strategy][0]:
            strategy_best[strategy] = (t2_acc, res)

    names = []
    t2_accs = []
    token_counts = []
    for strategy, (acc, res) in sorted(strategy_best.items()):
        names.append(strategy.replace('_', '\n'))
        t2_accs.append(acc)
        token_counts.append(res['tier_metrics'].get(2, {}).get('mean_tokens', 0))

    # T2 accuracy
    bars = ax1.bar(names, t2_accs, color='#3498db', alpha=0.8)
    ax1.axhline(y=baseline_t2, color='red', linestyle='--', label=f'15B baseline ({baseline_t2:.1%})')
    ax1.set_title("T2 Accuracy by Strategy", fontweight='bold')
    ax1.set_ylabel("T2 Accuracy")
    ax1.legend()
    ax1.tick_params(axis='x', rotation=0)

    # Token cost
    baseline_tok = baseline_15b['tier_metrics'].get(2, {}).get('mean_tokens', 0)
    ax2.bar(names, token_counts, color='#e67e22', alpha=0.8)
    ax2.axhline(y=baseline_tok, color='red', linestyle='--', label=f'15B baseline ({baseline_tok:.0f})')
    ax2.set_title("Mean Reasoning Tokens by Strategy", fontweight='bold')
    ax2.set_ylabel("Tokens")
    ax2.legend()
    ax2.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig30_perturbation_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_fig31_failure_modes(failure_summary, fig_dir):
    """Failure mode breakdown."""
    fig, ax = plt.subplots(figsize=(8, 5))

    modes = list(failure_summary.keys())
    counts = [failure_summary[m]['n'] for m in modes]
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#95a5a6']

    ax.barh(modes, counts, color=colors[:len(modes)])
    ax.set_xlabel("Count")
    ax.set_title("Failure Mode Distribution (15E)", fontweight='bold')

    for i, (m, c) in enumerate(zip(modes, counts)):
        if c > 0:
            pct = failure_summary[m]['pct']
            ax.text(c + 0.5, i, f"{pct:.0%}", va='center')

    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig31_failure_modes.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 15: Epistemic Controller")
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    parser.add_argument('--checkpoint', type=str, default='results/rim_model.pt')
    parser.add_argument('--skip-15a', action='store_true')
    parser.add_argument('--skip-15c', action='store_true')
    parser.add_argument('--skip-best-of-n', action='store_true',
                        help='Skip best-of-N in 15C (saves 5x time)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 15: Epistemic Controller")
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

    # ---- Create Test Dataset (matching Phase 12/14) ----
    print("\n--- Creating Test Dataset ---")
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456,
    )
    print(f"  Test: {len(test_ds)} examples")

    all_results = {}

    # ============================================================
    # 15A: Regime Classification Diagnostic
    # ============================================================
    regime_results = None
    regime_stats = {}
    regime_decision = "mixed"

    if not args.skip_15a:
        regime_results, regime_stats, regime_decision = \
            experiment_15a_regime_diagnostic(model, test_ds, device)
        all_results['15a_regime'] = {
            'stats': regime_stats,
            'decision': regime_decision,
        }

    # ============================================================
    # 15B: Halt Veto Baseline
    # ============================================================
    results_15b = experiment_15b_halt_veto_baseline(model, test_ds, device)
    all_results['15b_veto_baseline'] = {
        'tier_metrics': {str(k): v for k, v in results_15b['tier_metrics'].items()},
        'overall_accuracy': results_15b['overall_accuracy'],
        'mean_tokens': results_15b['mean_tokens'],
    }

    baseline_t2 = results_15b['tier_metrics'].get(2, {}).get('accuracy', 0)
    baseline_overall = results_15b['overall_accuracy']

    # ============================================================
    # 15C: Perturbation Ablation
    # ============================================================
    conditions_15c = {}
    if not args.skip_15c:
        conditions_15c = experiment_15c_perturbation_ablation(model, test_ds, device)
        all_results['15c_perturbation'] = {
            key: {
                'tier_metrics': {str(k): v for k, v in res['tier_metrics'].items()},
                'overall_accuracy': res['overall_accuracy'],
                'mean_tokens': res['mean_tokens'],
            }
            for key, res in conditions_15c.items()
        }

    # ============================================================
    # Find best strategy from 15C
    # ============================================================
    best_strategy = None
    best_patience = 5
    best_max_int = 2
    best_t2 = baseline_t2

    for key, res in conditions_15c.items():
        t2_acc = res['tier_metrics'].get(2, {}).get('accuracy', 0)
        if t2_acc > best_t2:
            best_t2 = t2_acc
            # Parse key to get params
            if key == 'best_of_5':
                best_strategy = 'best_of_5'
                best_patience = 5
                best_max_int = 0
            else:
                parts = key.rsplit('_p', 1)
                if len(parts) == 2:
                    best_strategy = parts[0]
                    pm = parts[1].split('_m')
                    if len(pm) == 2:
                        best_patience = int(pm[0])
                        best_max_int = int(pm[1])

    if best_strategy is None:
        # Fallback: use expression_reinjection with default params
        best_strategy = 'expression_reinjection'
        best_patience = 5
        best_max_int = 2

    print(f"\n  Best strategy from 15C: {best_strategy} "
          f"(p={best_patience}, m={best_max_int}, T2={best_t2:.1%})")

    # ============================================================
    # 15D: Full Controller
    # ============================================================
    if best_strategy == 'best_of_5':
        # Best-of-N is already the full controller
        print("\n  15D: Using best-of-5 as full controller (already evaluated in 15C)")
        results_15d = conditions_15c.get('best_of_5', results_15b)
    else:
        results_15d = experiment_15d_full_controller(
            model, test_ds, device,
            best_strategy, best_patience, best_max_int,
        )

    all_results['15d_full_controller'] = {
        'strategy': best_strategy,
        'patience': best_patience,
        'max_interventions': best_max_int,
        'tier_metrics': {str(k): v for k, v in results_15d['tier_metrics'].items()},
        'overall_accuracy': results_15d['overall_accuracy'],
        'mean_tokens': results_15d['mean_tokens'],
    }

    # ============================================================
    # 15E: Failure Analysis
    # ============================================================
    failure_summary = experiment_15e_failure_analysis(
        results_15d, best_max_int
    )
    all_results['15e_failure'] = failure_summary

    # ============================================================
    # Comparison Table
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 15: Comparison Table")
    print("=" * 60)

    t2_15b = results_15b['tier_metrics'].get(2, {}).get('accuracy', 0)
    t2_15d = results_15d['tier_metrics'].get(2, {}).get('accuracy', 0)
    tok_15b = results_15b['mean_tokens']
    tok_15d = results_15d['mean_tokens']

    print(f"  {'Condition':30s} | {'T2 Acc':>8s} | {'Overall':>8s} | {'Tokens':>8s}")
    print("  " + "-" * 65)
    print(f"  {'15B veto only':30s} | {t2_15b:8.1%} | {baseline_overall:8.1%} | {tok_15b:8.1f}")
    print(f"  {'15D full controller':30s} | {t2_15d:8.1%} | "
          f"{results_15d['overall_accuracy']:8.1%} | {tok_15d:8.1f}")
    print(f"  {'Delta':30s} | {t2_15d - t2_15b:+8.1%} | "
          f"{results_15d['overall_accuracy'] - baseline_overall:+8.1%} | "
          f"{tok_15d - tok_15b:+8.1f}")

    # ============================================================
    # Figures
    # ============================================================
    print("\n--- Generating Figures ---")

    if regime_results:
        plot_fig29_regime_distribution(regime_results, args.fig_dir)

    if conditions_15c and results_15b:
        plot_fig30_perturbation_comparison(conditions_15c, results_15b, args.fig_dir)

    if failure_summary:
        plot_fig31_failure_modes(failure_summary, args.fig_dir)

    # ============================================================
    # Save Results
    # ============================================================
    results_path = os.path.join(args.results_dir, 'phase15_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    # ============================================================
    # Success Criteria
    # ============================================================
    print("\n" + "=" * 60)
    print("Phase 15: Success Criteria")
    print("=" * 60)

    any_perturb_beats_veto = any(
        res['tier_metrics'].get(2, {}).get('accuracy', 0) > t2_15b + 0.005
        for res in conditions_15c.values()
    ) if conditions_15c else False

    token_ratio = tok_15d / max(tok_15b, 1)
    best_of_n_ratio = 5.0 if best_strategy == 'best_of_5' else token_ratio

    # Regime classification correlation with accuracy
    regime_corr = 0.0
    if regime_results:
        # Binary: CONVERGING=1, else=0; check correlation with correctness
        conv_vals = [1 if r['regime_at_halt'] == 'CONVERGING' else 0 for r in regime_results]
        acc_vals = [1 if r['is_correct_halt'] else 0 for r in regime_results]
        if len(set(conv_vals)) > 1 and len(set(acc_vals)) > 1:
            from scipy.stats import pearsonr
            regime_corr = abs(pearsonr(conv_vals, acc_vals)[0])

    addressable_errors = sum(
        v['n'] for k, v in failure_summary.items()
        if k in ['PERSISTENT_ORBIT', 'EXHAUSTED_BUDGET', 'WRONG_BASIN']
    )
    total_errors = sum(v['n'] for v in failure_summary.values())
    addressable_pct = addressable_errors / max(total_errors, 1)

    criteria = [
        ("T2 accuracy (full controller >80%)", t2_15d, 0.80),
        ("Overall accuracy (>75%)", results_15d['overall_accuracy'], 0.75),
        ("Perturbation beats veto alone", float(any_perturb_beats_veto), 0.5),
        ("Token overhead (<2x, <5x for BoN)", 2.0 - min(token_ratio, best_of_n_ratio / 2.5), 0.0),
        ("Regime corr. with accuracy (r>0.3)", regime_corr, 0.30),
        ("Addressable errors >30%", addressable_pct, 0.30),
    ]

    n_pass = 0
    for name, value, threshold in criteria:
        status = "PASS" if value >= threshold else "FAIL"
        if status == "PASS":
            n_pass += 1
        print(f"  {name}: {value:.3f} -- {status}")

    print(f"\n  Score: {n_pass}/{len(criteria)} criteria passed")

    best_delta = t2_15d - t2_15b
    print(f"\n  15B baseline T2: {t2_15b:.1%}")
    print(f"  15D full controller T2: {t2_15d:.1%} (delta: {best_delta:+.1%})")


if __name__ == '__main__':
    main()
