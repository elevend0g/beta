"""
Phase 16: Failure Analysis on Basin Model

Loads the basin model (results/basin_model.pt), generates answers for 1000
test examples, and classifies every incorrect example by:
  a) Output completeness (INCOMPLETE_OUTPUT vs completed)
  b) Intermediate verification (WRONG_INTERMEDIATE, WRONG_COMPOSITION, UNPARSEABLE)
  c) Regime at final generation step (ORBITING, CONVERGING, PROGRESSING, DIFFUSING)
  d) Expression difficulty (num_ops, effective_ops correlation)

Prints detailed breakdown and representative error examples.
"""

import os
import sys
import re
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, count_parameters
from train import get_device
from compressible_task import CompressibleArithmeticDataset
from phase14f_basin_depth import PNA_SSM_Basin


# ============================================================================
# Generation (works for both models)
# ============================================================================

def generate_answer(model, expression, device, max_length=200, has_halt_head=True):
    """
    Autoregressive generation for either model.
    Both models run WITHOUT halt confidence check (generate to EOS or max_length).
    """
    prompt_text = f"Input:{expression} "
    prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
    generated_ids = list(prompt_ids)

    states_list = []
    stop_reason = "max_length"

    eos_id = VOCAB['<EOS>']
    halt_id = VOCAB.get('<HALT>', -1)

    model.eval()
    with torch.no_grad():
        for step in range(max_length):
            input_tensor = torch.tensor(
                [generated_ids], dtype=torch.long, device=device
            )

            max_pos = getattr(model, 'max_seq_len', 256)
            if hasattr(model, 'pos_encoding'):
                max_pos = model.pos_encoding.num_embeddings
            if input_tensor.size(1) > max_pos:
                input_tensor = input_tensor[:, -max_pos:]

            outputs = model(input_tensor)
            logits = outputs["logits"][:, -1, :]

            # Collect state vector
            if outputs.get("states_sequence") is not None:
                last_state = outputs["states_sequence"][:, -1, :]
                states_list.append(last_state.squeeze(0).cpu())

            # Greedy decode
            next_token = logits.argmax(dim=-1).item()
            generated_ids.append(next_token)

            if next_token == eos_id:
                stop_reason = "eos"
                break
            elif next_token == halt_id:
                stop_reason = "halt_token"
                break

    text = detokenize(generated_ids)
    parsed_answer = parse_answer(text)
    reasoning_tokens = count_reasoning_tokens(generated_ids)

    return {
        'parsed_answer': parsed_answer,
        'generated_text': text,
        'generated_ids': generated_ids,
        'reasoning_tokens': reasoning_tokens,
        'states': states_list,
        'stop_reason': stop_reason,
    }


def parse_answer(text):
    """Extract numeric answer from generated text. Supports negative numbers."""
    match = re.search(r'Result:(-?\d+)', text)
    if match:
        return int(match.group(1))
    eq_matches = re.findall(r'=(-?\d+)', text)
    if eq_matches:
        return int(eq_matches[-1])
    num_matches = re.findall(r'(-?\d+)', text)
    if num_matches:
        return int(num_matches[-1])
    return None


def count_reasoning_tokens(generated_ids):
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
# Three-Signal Computation
# ============================================================================

def compute_state_entropy(state_vec):
    """Energy-based entropy of a state vector."""
    energy = state_vec ** 2
    probs = energy / (energy.sum() + 1e-9)
    return -(probs * torch.log2(probs + 1e-9)).sum().item()


def compute_cycling_score(states, t, k_values=[2, 4, 8]):
    """Sc: max similarity to previous states at lags k."""
    if t < min(k_values):
        return 0.0
    scores = []
    h_t = states[t]
    for k in k_values:
        if t - k >= 0:
            h_prev = states[t - k]
            dist = (h_t - h_prev).norm().item()
            norm_sum = h_t.norm().item() + h_prev.norm().item() + 1e-9
            scores.append(1.0 - dist / norm_sum)
    return max(scores) if scores else 0.0


def compute_entropy_gradient(states, t):
    """Smoothed entropy gradient over last 3 steps."""
    if t < 1:
        return 0.0
    H_t = compute_state_entropy(states[t])
    if t >= 3:
        H_prev = compute_state_entropy(states[t - 3])
        return (H_t - H_prev) / 3.0
    else:
        H_prev = compute_state_entropy(states[t - 1])
        return H_t - H_prev


def classify_regime(states):
    """Classify regime at the final generation step."""
    if len(states) < 2:
        return 'OTHER', 0.0, 0.0, 0.0

    t = len(states) - 1
    Sc = compute_cycling_score(states, t)
    grad_H = compute_entropy_gradient(states, t)
    H = compute_state_entropy(states[t])

    SC_THRESHOLD = 0.15
    GRAD_H_THRESHOLD = -0.03
    H_HIGH = 3.2

    if grad_H < GRAD_H_THRESHOLD and Sc < SC_THRESHOLD:
        regime = 'CONVERGING'
    elif Sc > SC_THRESHOLD and abs(grad_H) < 0.02:
        regime = 'ORBITING'
    elif H > H_HIGH and abs(grad_H) < 0.02:
        regime = 'DIFFUSING'
    else:
        regime = 'PROGRESSING'

    return regime, Sc, grad_H, H


# ============================================================================
# Intermediate Verification
# ============================================================================

def verify_intermediates(generated_text):
    """
    Extract and verify all "X op Y = Z" patterns in the reasoning chain.

    Returns:
        (mode, details)
        mode: 'NO_RESULT', 'WRONG_INTERMEDIATE', 'WRONG_COMPOSITION', 'ALL_CORRECT', 'UNPARSEABLE'
        details: dict with step-level info
    """
    if 'Result:' not in generated_text:
        return 'NO_RESULT', {'reason': 'No Result: in output'}

    # Split into reasoning and answer
    parts = generated_text.split('Result:')
    reasoning = parts[0]

    # Extract all "A op B = C" patterns (supports multi-digit, negative)
    # Pattern: number, operator (+,-,*), number, =, number
    step_pattern = r'(-?\d+)\s*([+\-*])\s*(-?\d+)\s*=\s*(-?\d+)'
    steps = re.findall(step_pattern, reasoning)

    if not steps:
        return 'UNPARSEABLE', {'reason': 'No intermediate steps found', 'text': reasoning[:100]}

    wrong_steps = []
    for i, (a_str, op, b_str, c_str) in enumerate(steps):
        a, b, c = int(a_str), int(b_str), int(c_str)
        if op == '+':
            expected = a + b
        elif op == '-':
            expected = a - b
        elif op == '*':
            expected = a * b
        else:
            continue

        # Clamp to [0, 99] to match dataset generation
        expected = max(0, min(99, expected))

        if c != expected:
            wrong_steps.append({
                'step': i,
                'expression': f"{a}{op}{b}",
                'got': c,
                'expected': expected,
            })

    if wrong_steps:
        return 'WRONG_INTERMEDIATE', {
            'n_steps': len(steps),
            'n_wrong': len(wrong_steps),
            'wrong_steps': wrong_steps,
        }

    return 'ALL_CORRECT', {'n_steps': len(steps)}


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 16: Failure Analysis")
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default='results/basin_model.pt')
    parser.add_argument('--model-type', type=str, default='basin',
                        choices=['basin', 'pna_ssm'],
                        help='basin: PNA_SSM_Basin (4 layers, max_seq_len=64); '
                             'pna_ssm: PNA_SSM (6 layers, max_seq_len=256)')
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(42)
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 16: Failure Analysis — Basin Model")
    print(f"{'=' * 60}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Load Model ----
    print(f"\n--- Loading Model ({args.model_type}) ---")
    print(f"  Checkpoint: {args.checkpoint}")

    if args.model_type == 'pna_ssm':
        model = PNA_SSM(
            VOCAB_SIZE, d_model=512, n_layers=6, d_state=16, max_seq_len=256,
        ).to(device)
    else:
        model = PNA_SSM_Basin(
            VOCAB_SIZE, d_model=512, n_layers=4, d_state=16, max_seq_len=64,
        ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"  Model loaded: {count_parameters(model):,} params")
    print(f"  Has halt head: {hasattr(model, 'halt_head')}")

    # ---- Create Test Dataset (matching Phase 12/14) ----
    print("\n--- Creating Test Dataset ---")
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456,
    )
    print(f"  Test: {len(test_ds)} examples")

    # ---- Generate All Answers ----
    print(f"\n--- Generating Answers ---")
    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        gen = generate_answer(model, ex['expression'], device,
                              max_length=200, has_halt_head=False)
        results.append({
            'idx': i,
            'expression': ex['expression'],
            'expected': ex['answer'],
            'parsed': gen['parsed_answer'],
            'is_correct': gen['parsed_answer'] == ex['answer'],
            'generated_text': gen['generated_text'],
            'generated_ids': gen['generated_ids'],
            'reasoning_tokens': gen['reasoning_tokens'],
            'states': gen['states'],
            'stop_reason': gen['stop_reason'],
            'tier': ex['tier'],
            'num_ops': ex['num_ops'],
            'effective_ops': ex['effective_ops'],
        })

    # ---- Overall Accuracy ----
    n_correct = sum(1 for r in results if r['is_correct'])
    n_total = len(results)
    print(f"\n  Overall accuracy: {n_correct}/{n_total} = {n_correct/n_total:.1%}")

    # Per-tier
    print(f"\n  Per-tier accuracy:")
    for tier in [0, 1, 2]:
        tier_results = [r for r in results if r['tier'] == tier]
        if tier_results:
            acc = np.mean([r['is_correct'] for r in tier_results])
            tok = np.mean([r['reasoning_tokens'] for r in tier_results])
            print(f"    T{tier}: {acc:.1%} ({sum(r['is_correct'] for r in tier_results)}"
                  f"/{len(tier_results)}) tokens={tok:.1f}")

    # Stop reason distribution
    stop_counts = defaultdict(int)
    for r in results:
        stop_counts[r['stop_reason']] += 1
    print(f"\n  Stop reasons: {dict(stop_counts)}")

    # ---- Focus on Incorrect Examples ----
    incorrect = [r for r in results if not r['is_correct']]
    print(f"\n{'=' * 60}")
    print(f"Failure Analysis ({len(incorrect)} incorrect examples)")
    print(f"{'=' * 60}")

    # ---- a) Output Completeness ----
    print(f"\n--- a) Output Completeness ---")
    completeness = defaultdict(list)
    for r in incorrect:
        has_result = 'Result:' in r['generated_text']
        if not has_result:
            completeness['INCOMPLETE_OUTPUT'].append(r)
        else:
            completeness['HAS_RESULT'].append(r)

    n_incomplete = len(completeness['INCOMPLETE_OUTPUT'])
    n_has_result = len(completeness['HAS_RESULT'])
    print(f"  INCOMPLETE_OUTPUT (no Result:): {n_incomplete} ({n_incomplete/max(len(incorrect),1):.1%})")
    print(f"  HAS_RESULT:                     {n_has_result} ({n_has_result/max(len(incorrect),1):.1%})")

    # ---- b) Intermediate Verification ----
    print(f"\n--- b) Intermediate Verification ---")
    verification_modes = defaultdict(list)
    for r in incorrect:
        mode, details = verify_intermediates(r['generated_text'])
        r['verification_mode'] = mode
        r['verification_details'] = details
        verification_modes[mode].append(r)

    for mode in ['NO_RESULT', 'WRONG_INTERMEDIATE', 'WRONG_COMPOSITION', 'ALL_CORRECT', 'UNPARSEABLE']:
        n = len(verification_modes[mode])
        pct = n / max(len(incorrect), 1)
        print(f"  {mode:25s}: {n:4d} ({pct:.1%})")

    # For ALL_CORRECT: these are cases where all intermediates are right
    # but the final parsed answer is wrong. This is WRONG_COMPOSITION.
    # Reclassify them.
    n_wrong_comp = 0
    for r in verification_modes['ALL_CORRECT']:
        r['verification_mode'] = 'WRONG_COMPOSITION'
        n_wrong_comp += 1
    if n_wrong_comp > 0:
        print(f"  (Reclassified {n_wrong_comp} ALL_CORRECT -> WRONG_COMPOSITION)")

    # ---- c) Regime at Final Step ----
    print(f"\n--- c) Regime at Final Step ---")
    regime_counts = defaultdict(list)
    for r in incorrect:
        if r['states']:
            regime, Sc, grad_H, H = classify_regime(r['states'])
        else:
            regime, Sc, grad_H, H = 'OTHER', 0.0, 0.0, 0.0
        r['regime'] = regime
        r['Sc'] = Sc
        r['grad_H'] = grad_H
        r['H'] = H
        regime_counts[regime].append(r)

    for regime in ['CONVERGING', 'ORBITING', 'PROGRESSING', 'DIFFUSING', 'OTHER']:
        n = len(regime_counts[regime])
        pct = n / max(len(incorrect), 1)
        if n > 0:
            mean_Sc = np.mean([r['Sc'] for r in regime_counts[regime]])
            mean_gH = np.mean([r['grad_H'] for r in regime_counts[regime]])
            print(f"  {regime:15s}: {n:4d} ({pct:.1%})  "
                  f"mean Sc={mean_Sc:.3f} mean grad_H={mean_gH:.4f}")
        else:
            print(f"  {regime:15s}: {n:4d} ({pct:.1%})")

    # ---- d) Expression Difficulty ----
    print(f"\n--- d) Expression Difficulty ---")

    # Error rate by num_ops
    print(f"\n  Error rate by num_ops:")
    for n_ops in sorted(set(r['num_ops'] for r in results)):
        subset = [r for r in results if r['num_ops'] == n_ops]
        err_rate = 1.0 - np.mean([r['is_correct'] for r in subset])
        n_err = sum(1 for r in subset if not r['is_correct'])
        print(f"    ops={n_ops}: {err_rate:.1%} error rate ({n_err}/{len(subset)})")

    # Error rate by effective_ops
    print(f"\n  Error rate by effective_ops:")
    for eff in sorted(set(r['effective_ops'] for r in results)):
        subset = [r for r in results if r['effective_ops'] == eff]
        err_rate = 1.0 - np.mean([r['is_correct'] for r in subset])
        n_err = sum(1 for r in subset if not r['is_correct'])
        print(f"    eff_ops={eff}: {err_rate:.1%} error rate ({n_err}/{len(subset)})")

    # Correlation
    ops_arr = np.array([r['num_ops'] for r in results])
    eff_arr = np.array([r['effective_ops'] for r in results])
    err_arr = np.array([0 if r['is_correct'] else 1 for r in results])

    if len(set(ops_arr)) > 1 and len(set(err_arr)) > 1:
        corr_ops = np.corrcoef(ops_arr, err_arr)[0, 1]
        corr_eff = np.corrcoef(eff_arr, err_arr)[0, 1]
        print(f"\n  Correlation with error:")
        print(f"    num_ops:       r = {corr_ops:.3f}")
        print(f"    effective_ops: r = {corr_eff:.3f}")

    # ---- Cross-tabulation: Completeness x Regime ----
    print(f"\n--- Cross-tabulation: Verification Mode x Regime ---")
    modes_list = ['NO_RESULT', 'WRONG_INTERMEDIATE', 'WRONG_COMPOSITION', 'UNPARSEABLE']
    regimes_list = ['CONVERGING', 'ORBITING', 'PROGRESSING', 'DIFFUSING', 'OTHER']

    # Header
    header = f"  {'Mode':25s}"
    for reg in regimes_list:
        header += f" | {reg:>11s}"
    header += f" | {'Total':>6s}"
    print(header)
    print("  " + "-" * (25 + 15 * len(regimes_list) + 10))

    for mode in modes_list:
        row = f"  {mode:25s}"
        total = 0
        for reg in regimes_list:
            n = sum(1 for r in incorrect
                    if r.get('verification_mode') == mode and r.get('regime') == reg)
            row += f" | {n:11d}"
            total += n
        row += f" | {total:6d}"
        print(row)

    # ---- Representative Error Examples ----
    print(f"\n{'=' * 60}")
    print("Representative Error Examples (up to 5 per failure mode)")
    print(f"{'=' * 60}")

    for mode in ['NO_RESULT', 'WRONG_INTERMEDIATE', 'WRONG_COMPOSITION', 'UNPARSEABLE']:
        examples = [r for r in incorrect if r.get('verification_mode') == mode]
        if not examples:
            continue

        print(f"\n  --- {mode} ({len(examples)} total) ---")
        for r in examples[:5]:
            print(f"    Expression:  {r['expression']}")
            print(f"    Expected:    {r['expected']}")
            print(f"    Parsed:      {r['parsed']}")
            print(f"    Regime:      {r['regime']} (Sc={r['Sc']:.3f} grad_H={r['grad_H']:.4f})")
            print(f"    Stop:        {r['stop_reason']} ({r['reasoning_tokens']} tokens)")
            # Truncate generated text for display
            gen_text = r['generated_text'][:120]
            print(f"    Generated:   {gen_text}")
            if r.get('verification_details'):
                details = r['verification_details']
                if 'wrong_steps' in details:
                    for ws in details['wrong_steps'][:2]:
                        print(f"      Wrong step: {ws['expression']}={ws['got']} "
                              f"(expected {ws['expected']})")
            print()

    # ---- Tier breakdown of failures ----
    print(f"\n--- Failure Breakdown by Tier ---")
    for tier in [0, 1, 2]:
        tier_incorrect = [r for r in incorrect if r['tier'] == tier]
        if not tier_incorrect:
            continue
        tier_total = sum(1 for r in results if r['tier'] == tier)
        print(f"\n  Tier {tier}: {len(tier_incorrect)}/{tier_total} incorrect")
        for mode in modes_list:
            n = sum(1 for r in tier_incorrect if r.get('verification_mode') == mode)
            if n > 0:
                print(f"    {mode}: {n}")
        for reg in regimes_list:
            n = sum(1 for r in tier_incorrect if r.get('regime') == reg)
            if n > 0:
                print(f"    [{reg}]: {n}")

    # ---- Save Results ----
    all_results = {
        'overall': {
            'n_total': n_total,
            'n_correct': n_correct,
            'accuracy': n_correct / n_total,
        },
        'per_tier': {},
        'stop_reasons': dict(stop_counts),
        'n_incorrect': len(incorrect),
        'completeness': {
            'INCOMPLETE_OUTPUT': n_incomplete,
            'HAS_RESULT': n_has_result,
        },
        'verification': {
            mode: len(verification_modes[mode]) for mode in verification_modes
        },
        'regime': {
            regime: len(regime_counts[regime]) for regime in regime_counts
        },
    }

    for tier in [0, 1, 2]:
        tier_results = [r for r in results if r['tier'] == tier]
        if tier_results:
            all_results['per_tier'][str(tier)] = {
                'n': len(tier_results),
                'accuracy': float(np.mean([r['is_correct'] for r in tier_results])),
                'mean_tokens': float(np.mean([r['reasoning_tokens'] for r in tier_results])),
            }

    results_path = os.path.join(args.results_dir, 'phase16_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {results_path}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("Phase 16: Summary")
    print(f"{'=' * 60}")
    print(f"  Total: {n_correct}/{n_total} correct ({n_correct/n_total:.1%})")
    print(f"  Errors: {len(incorrect)}")
    print(f"    INCOMPLETE_OUTPUT:  {n_incomplete}")
    print(f"    WRONG_INTERMEDIATE: {len(verification_modes['WRONG_INTERMEDIATE'])}")
    print(f"    WRONG_COMPOSITION:  {len(verification_modes['WRONG_COMPOSITION']) + len(verification_modes['ALL_CORRECT'])}")
    print(f"    UNPARSEABLE:        {len(verification_modes['UNPARSEABLE'])}")
    print(f"  Dominant regime:      ", end="")
    if regime_counts:
        dominant = max(regime_counts.keys(), key=lambda k: len(regime_counts[k]))
        print(f"{dominant} ({len(regime_counts[dominant])}/{len(incorrect)})")
    else:
        print("(no errors)")


if __name__ == '__main__':
    main()
