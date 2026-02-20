```markdown
# Phase 14F Evaluation Protocol
# Execute in order. All paths relative to project root.

## Task

Evaluate the basin-deepened model (`results/basin_model.pt`) and compare against
Phase 12 baseline (`results/rim_model.pt`). The basin model has no halt head —
it stops on `<EOS>` token. Write a single script `src/phase14f_evaluation.py`
that runs all evaluations below.

## File to Create: `src/phase14f_evaluation.py`

### Requirements

- Load both models: `results/basin_model.pt` (basin) and `results/rim_model.pt` (Phase 12)
- Test dataset: `CompressibleArithmeticDataset(num_samples=1000, min_ops=3, max_ops=8, max_seq_len=64, seed=456)`
- Import shared code from `src/compressible_task.py` (dataset, vocabulary, model classes)
- Device: CUDA if available, else CPU

### Section 1: Model Loading and Inspection

```python
"""
Load both models. Print architecture summary for each:
  - Total parameters
  - Whether halt_confidence head exists
  - d_model, d_state, n_layers (from checkpoint or model inspection)

Handle the case where basin_model may have a different class or missing halt head.
If the checkpoint contains the model class definition, use it.
If not, instantiate the same architecture as Phase 12 and try loading weights
with strict=False to handle missing halt head keys.

Print any missing/unexpected keys from state_dict loading.
"""
```

### Section 2: Generation Function

```python
"""
Write a generate function that works for BOTH models:

def generate_answer(model, expression, device, max_length=200, has_halt_head=True):
    - Tokenize: [BOS] + "Input:{expression} " 
    - Generate autoregressively (greedy, argmax)
    - Full reprocessing at each step (feed all generated_ids each time,
      same approach as ConfusionGatedGenerator from Phase 14)
    - Clamp input length to model.pos_encoding.num_embeddings
    - Stop conditions (checked in this order):
        1. next_token == VOCAB['<EOS>']  -> stop
        2. next_token == VOCAB.get('<HALT>', -1) -> stop  
        3. If has_halt_head is False: skip halt_confidence check
           If has_halt_head is True: DO NOT check halt_confidence 
           (we run both models without halt, matching Phase 14A-0 thresh=1.0)
        4. step reaches max_length -> stop
    - Return dict with:
        'parsed_answer': int or None (parsed from generated text)
        'generated_text': str
        'generated_ids': list[int]
        'reasoning_tokens': int (tokens between Input: and Result: or end)
        'states': list[Tensor] (hidden states at each step, for signal analysis)
        'stop_reason': str

Answer parsing (same as Phase 14):
    def parse_answer(text):
        # Priority 1: "Result:X" pattern
        match = re.search(r'Result:(-?\d+)', text)
        if match: return int(match.group(1))
        # Priority 2: last "=X" pattern  
        eq_matches = re.findall(r'=(-?\d+)', text)
        if eq_matches: return int(eq_matches[-1])
        # Priority 3: last number
        num_matches = re.findall(r'(-?\d+)', text)
        if num_matches: return int(num_matches[-1])
        return None

IMPORTANT: include negative number support (-?\d+) in all regex patterns.
"""
```

### Section 3: 14F-0 Terminal State Diagnostic

```python
"""
Run on BOTH models (Phase 12 and basin model). Teacher-forced forward pass.

For each test example:
  1. Feed the full token sequence (teacher-forced)
  2. Get states_sequence from model output
  3. Find the Result: token position in the sequence
  4. Measure at the Result: position:
     - state_delta: ||s[rp] - s[rp-1]||  (state change AT result)
     - state_stability: ||s[rp+1] - s[rp]|| / (||s[rp] - s[rp-1]|| + 1e-9)
       (ratio < 1 means state is stabilizing; > 1 means destabilizing)
     - H_at_result: state entropy at result position
       H = -(p * log2(p)).sum() where p = s² / sum(s²)
     - grad_H_approach: (H[rp] - H[rp-3]) / 3.0  (entropy gradient approaching result)
  5. Classify as correct/incorrect based on teacher-forced token prediction accuracy
     at the Result: position (does argmax(logits[rp]) == target_token[rp+1]?)

Print for each model:
  14F-0: Terminal State Diagnostic — {model_name}
    Correct examples (n=X):
      state_delta:     {mean:.4f}
      stability_ratio: {mean:.4f}
      H_at_result:     {mean:.3f}
      grad_H_approach: {mean:.4f}
    Incorrect examples (n=X):
      state_delta:     {mean:.4f}
      stability_ratio: {mean:.4f}  
      H_at_result:     {mean:.3f}
      grad_H_approach: {mean:.4f}
"""
```

### Section 4: Accuracy Evaluation (Both Models)

```python
"""
Generate answers for all 1000 test examples using generate_answer() with BOTH models.
Both models run WITHOUT halt (matching Phase 14A-0 thresh=1.0 for Phase 12).

For each model, compute:
  - Per-tier accuracy: T0 (incompressible), T1 (partially), T2 (fully compressible)
  - Overall accuracy
  - Mean reasoning tokens per tier
  - Mean reasoning tokens overall

Also track for each example:
  - Whether the generated text contains "Result:" (the model completed its output)
  - The stop_reason distribution

Print:
  14F Accuracy Comparison:
  Model          | T0 Acc | T1 Acc | T2 Acc | Overall | Tokens | Has Result:
  ---------------------------------------------------------------------------
  Phase 12       | XX.X%  | XX.X%  | XX.X%  | XX.X%   |  XX.X  |    XX.X%
  Basin Model    | XX.X%  | XX.X%  | XX.X%  | XX.X%   |  XX.X  |    XX.X%
  Delta          | +X.Xpp | +X.Xpp | +X.Xpp | +X.Xpp  |  XX.X  |    +X.Xpp

  Stop reason distribution:
  Model       | EOS   | HALT_token | max_length
  Phase 12    | XXX   | XXX        | XXX
  Basin Model | XXX   | XXX        | XXX
"""
```

### Section 5: Three-Signal Validation

```python
"""
Using the states captured during generation (Section 4), compute the three signals
for the basin model at the LAST generation step:

For each example:
  states = list of hidden state tensors from generation
  t = len(states) - 1  (last step)
  
  # Cycling score Sc (max over k=2,4,8)
  Sc = 0.0
  for k in [2, 4, 8]:
      if t - k >= 0:
          h_t = states[t]
          h_prev = states[t - k]
          dist = (h_t - h_prev).norm().item()
          norm_sum = h_t.norm().item() + h_prev.norm().item() + 1e-9
          Sc = max(Sc, 1.0 - dist / norm_sum)
  
  # Entropy H
  energy = states[t] ** 2
  probs = energy / (energy.sum() + 1e-9)
  H = -(probs * torch.log2(probs + 1e-9)).sum().item()
  
  # Entropy gradient (smoothed over last 3 steps)
  if len(states) >= 4:
      H_prev = compute_entropy(states[-4])
      grad_H = (H - H_prev) / 3.0
  elif len(states) >= 2:
      H_prev = compute_entropy(states[-2])
      grad_H = H - H_prev
  else:
      grad_H = 0.0

Classify each example:
  - correct_at_eos: parsed_answer matches expected answer
  - Define "true convergence" as: correct AND Sc < 0.10 AND grad_H < -0.03
  - Define "false convergence" as: NOT true convergence

Print:
  14F Three-Signal Validation — Basin Model
    True Convergence (n=XX):
      Sc:     {mean:.3f}
      grad_H: {mean:.4f}
      H:      {mean:.3f}
    False Convergence (n=XX):
      Sc:     {mean:.3f}
      grad_H: {mean:.4f}
      H:      {mean:.3f}
    True convergence rate: XX.X%

  Comparison:
                    | Phase 12  | Basin Model
  True conv rate    |    3.9%   |    XX.X%
  True conv Sc      |    0.025  |    X.XXX
  False conv Sc     |    0.333  |    X.XXX
  Separation (ΔSc)  |    0.308  |    X.XXX
"""
```

### Section 6: Op Detection Probe

```python
"""
Train a linear probe on the basin model's frozen hidden states to predict
operation type at each position. Same methodology as Phase 14A-2.

1. Freeze the basin model
2. Run teacher-forced on training set (8000 examples, max_ops=8, seed=42)
3. At each token position that corresponds to an operator (+, -, *):
   - Extract the hidden state s[t] (from states_sequence)
   - Label = operation type (0=add, 1=sub, 2=mul)
4. Train a linear probe: nn.Linear(d_state, 3) 
5. Train for 10 epochs, batch_size=256, lr=0.001, Adam
6. Evaluate on test set

Print:
  14F Op Detection Probe — Basin Model
    Epoch 1: loss=X.XXXX acc=XX.X%
    ...
    Epoch 10: loss=X.XXXX acc=XX.X%
    Final accuracy: XX.X% (Phase 12: 96.7%)
"""
```

### Section 7: Purity Evaluation

```python
"""
Evaluate corrected geodesic purity on basin model. 
Import and reuse the purity evaluation from Phase 14E if available,
or reimplement:

For each test example that gets correct answer:
  1. Generate with basin model (same as Section 4)
  2. Extract the reasoning token sequence
  3. For each intermediate "=X" in the reasoning:
     - Check if X follows from a valid arithmetic rule applied to 
       the preceding operands/operator
     - A step is "pure" if it represents a valid algebraic simplification
  4. Purity = (pure steps) / (total steps) for examples with constraints

If the Phase 14 corrected purity function exists in src/phase14_halt_control.py,
import and reuse it. Pass the basin model's generated texts.

Print:
  14F Purity Evaluation — Basin Model
    Mean corrected purity: X.XXX (n=XXX with constraints)
    Phase 12 purity:       0.749
    Delta:                 +X.XXX
"""
```

### Section 8: Failure Analysis

```python
"""
For all incorrect examples from the basin model (Section 4):

Classify each failure:
  1. Check if "Result:" appears in generated text
     - If NO: failure_mode = "INCOMPLETE_OUTPUT" (model didn't finish)
  2. If Result: exists, check the reasoning chain:
     - Extract all "X op Y = Z" patterns from the reasoning
     - Verify each: does X op Y actually equal Z?
     - If any step is wrong: failure_mode = "WRONG_INTERMEDIATE"
     - If all steps are correct but final answer is wrong: 
       failure_mode = "WRONG_COMPOSITION" (steps right, combination wrong)
     - If no intermediate steps found: failure_mode = "UNPARSEABLE"
  3. Additionally classify by regime at final step:
     - Compute Sc, grad_H at final step
     - If Sc > 0.15 and |grad_H| < 0.02: regime = "ORBITING"
     - If grad_H < -0.03 and Sc < 0.10: regime = "CONVERGING" (wrong basin)
     - Else: regime = "OTHER"

Print:
  14F Failure Analysis — Basin Model
    Total incorrect: XXX / 1000
    
    By output completeness:
      INCOMPLETE_OUTPUT:   XXX (XX.X%)
      WRONG_INTERMEDIATE:  XXX (XX.X%)
      WRONG_COMPOSITION:   XXX (XX.X%)
      UNPARSEABLE:         XXX (XX.X%)
    
    By regime at final step:
      ORBITING:            XXX (XX.X%)
      CONVERGING (wrong):  XXX (XX.X%)
      OTHER:               XXX (XX.X%)
    
    Phase 12 comparison:
      Phase 12 errors: 275 (100% COMPUTATION_ERROR)
      Basin errors:    XXX (breakdown above)
"""
```

### Section 9: Summary and Success Criteria

```python
"""
Print final comparison table and success criteria:

  ============================================================
  Phase 14F: Results Summary
  ============================================================
  
  Metric                  | Phase 12 | Basin Model | Target  | Status
  --------------------------------------------------------------------------
  T2 accuracy (no halt)   |   78.8%  |     XX.X%   |  >82%   | PASS/FAIL
  Overall accuracy        |   73.6%  |     XX.X%   |  >77%   | PASS/FAIL
  Geodesic purity         |   74.9%  |     XX.X%   |  >70%   | PASS/FAIL
  True convergence rate   |    3.9%  |     XX.X%   |  >10%   | PASS/FAIL
  Op detection probe      |   96.7%  |     XX.X%   |  >95%   | PASS/FAIL
  Computation errors      |    100%  |     XX.X%   |  <80%   | PASS/FAIL
  
  Score: X/6 criteria passed

Save results to results/phase14f_results.json
Save figures to figures/fig32_basin_comparison.png (bar chart comparing all metrics)
"""
```

## Run Command

```bash
python src/phase14f_evaluation.py 2>&1 | tee results/phase14f_output.txt
```

## Key Implementation Notes

1. **Model loading**: The basin model may have different state_dict keys than Phase 12.
   Use `strict=False` when loading and print any missing/unexpected keys.
   If the model class is different, check the checkpoint for a 'model_class' or 
   'config' key that specifies the architecture.

2. **Hidden states**: Access via `outputs['states_sequence']` from the model's forward pass.
   If the basin model doesn't return this key, try `outputs['hidden_states']` or 
   inspect the model's forward method return dict.

3. **No halt check**: Neither model should use halt_confidence for stopping during 
   generation. Both generate to EOS or max_length. This matches the proven-optimal
   strategy from Phase 14.

4. **Determinism**: Use `torch.manual_seed(42)` before each evaluation section.
   Use `model.eval()` and `torch.no_grad()` throughout.

5. **Progress bars**: Use tqdm for all loops over the test set (1000 examples).
   Format: `section_name: {n}/{total} [{elapsed}<{remaining}]`
```