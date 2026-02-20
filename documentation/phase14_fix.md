I found it. There are multiple interacting bugs, and one of them completely explains the no_halt anomaly. Let me walk through them in order of severity.

---

## Bug 1: The `max_boost` Is Off By an Order of Magnitude

```python
# ConfusionGatedGenerator.__init__:
base_threshold=0.95, max_boost=0.04

# Maximum effective threshold:
effective_threshold = 0.95 + 1.0 * 0.04 = 0.99
```

From 14A-0, we know that threshold=0.995 gives **identical** results to threshold=0.95. The halt confidence is above 0.995 when it fires. So a max effective threshold of 0.99 cannot possibly suppress any halts.

The boost needed: at minimum 0.05 to reach 1.0. But `halt_conf > 1.0` is never true (sigmoid saturates below 1.0). So even `max_boost=0.05` would fail.

**This is why 14C shows zero improvement.** The Confusion Head correctly fires (F1=0.974), the confusion score is high, the threshold is raised to 0.99 — and the halt fires anyway because `halt_conf ≈ 0.9999`.

The extra tokens (31.9 vs 20.9) come from the few steps where `halt_conf` is between 0.95 and 0.99 — the confusion gating delays the halt briefly but can't prevent it.

---

## Bug 2: The `no_halt` Result — The Ghost Is in `CompressibleGenerator`

The 14A-0 experiment uses `CompressibleGenerator` (imported from Phase 11), **not** `ConfusionGatedGenerator`. I can't see `CompressibleGenerator.generate()`, but the behavior reveals its internal structure.

Compare the stopping conditions between the two generators:

```python
# ConfusionGatedGenerator (Phase 14) — Token appended BEFORE halt check:
next_token = logits.argmax(dim=-1).item()
generated_ids.append(next_token)          # ← Token ALWAYS appended

if next_token == self.halt_id:     break  # Check token stops
elif next_token == self.eos_id:    break
elif halt_conf > effective_threshold: break  # Halt check is LAST

# CompressibleGenerator (Phase 11) — Unknown order, but the evidence
# suggests halt is checked BEFORE the token is appended:
#   if halt_conf > threshold: break   ← Token NOT appended
#   next_token = logits.argmax()
#   generated_ids.append(next_token)
```

**The evidence that CompressibleGenerator checks halt BEFORE appending:**

| Condition | Generator | Threshold | Tokens | T2 Acc |
|-----------|-----------|-----------|--------|--------|
| 14A-0 baseline | CompressibleGenerator | 0.95 | 13.7 | 83.5% |
| 14A-0 no_halt | CompressibleGenerator | 1.0 | 13.7 | **97.4%** |
| 14C-A baseline | CompressibleGenerator | default | 20.9 | 83.5% |
| 14C-C cycle_det | ConfusionGatedGenerator | 0.95+boost | 31.9 | 83.5% |

The **identical token count** (13.7) across all 14A-0 conditions means the model always generates ~14 tokens regardless of threshold. This implies the stopping condition that actually fires is **not** halt_confidence — it's a HALT/EOS token at step ~14.

But accuracy jumps from 83.5% to 97.4% at `thresh=1.0`. **Same tokens, different accuracy.** The only explanation:

```python
# In CompressibleGenerator.generate() (inferred):
for step in range(max_length):
    outputs = model(input_tensor)
    halt_conf = outputs['halt_confidence'][:, -1, 0].item()
    
    if halt_conf > self.halt_threshold:
        stop_reason = 'halt_confidence'
        break                                    # ← Break WITHOUT appending
    
    next_token = logits.argmax(dim=-1).item()
    generated_ids.append(next_token)
    
    if next_token == HALT_TOKEN:
        stop_reason = 'halt_token'
        break                                    # ← Break AFTER appending
```

At `thresh=0.95`:
1. Step 12: halt_conf passes 0.95 → **break before appending the answer token**
2. The answer token was going to be generated at this step but is dropped
3. Parser extracts the last number from reasoning (previous intermediate result)
4. ~30% of the time, this intermediate ≠ final answer → **wrong**

At `thresh=1.0`:
1. Step 12: halt_conf < 1.0 → don't halt
2. Answer token IS appended
3. Step 13: HALT/EOS token generated → stop
4. Parser extracts the correct answer → **97.4% accuracy**

**The token count is the same** because `reasoning_tokens` either doesn't count the final token, or the dropped token and the EOS token net out.

### Verification: Does the Math Work?

```
T2 accuracy at thresh≤0.995:  83.5%  (283/339 correct)
T2 accuracy at thresh=1.0:    97.4%  (330/339 correct)
Difference:                    47 examples

47/339 = 13.9% of T2 examples have their answer token
dropped by the halt-before-append bug.
```

This is plausible — the halt fires just before the answer in ~14% of cases.

---

## Bug 3: 14A-0 and 14C Use Incompatible Baselines

| Experiment | Generator | halt_threshold | max_length | Tokens |
|------------|-----------|:-:|:-:|:-:|
| 14A-0 baseline | CompressibleGenerator | **0.95** | **100** | 13.7 |
| 14C-A baseline | CompressibleGenerator | **default** | **default** | 20.9 |
| 14C-B constant | CompressibleGenerator | **0.98** | **default** | 20.9 |

14C-A and 14C-B produce identical results (`tokens=20.9, T2=83.5%`), which means CompressibleGenerator's default threshold is ≥ 0.98 (or the threshold doesn't bind because the model stops on tokens first with the default max_length).

But 14A-0's baseline (thresh=0.95) gives `tokens=13.7`. **Different `max_length`** (100 vs default) likely explains the token count difference.

The accuracy comparison between '14A-0' and '14C' is contaminated by both `halt_threshold` and `max_length` differences.

---

## Bug 4: The Decision Logic Missed the Signal

```python
# 14A-0 decision logic:
baseline_acc = results.get((0.95, 2), {}).get('accuracy', 0)     # 83.5%
raised_acc = results.get((0.98, 2), {}).get('accuracy', 0)       # 83.5%
high_acc = results.get((0.995, 2), {}).get('accuracy', 0)        # 83.5%

if raised_acc > baseline_acc + 0.05:     # 83.5 > 88.5? No
    decision = "calibration_dominant"
elif high_acc <= baseline_acc + 0.02:    # 83.5 <= 85.5? Yes
    decision = "cycles_stuck"            # ← THIS WAS WRONG
```

The code never checks the `thresh=1.0` result (97.4%). The most important data point is ignored by the decision logic. It should include:

```python
nohalt_acc = results.get((1.0, 2), {}).get('accuracy', 0)
if nohalt_acc > baseline_acc + 0.10:
    decision = "halt_truncation"  # ← Correct diagnosis
```

This caused the code to trigger 14D (cycle breaking) — which was irrelevant.

---

## Bug 5: Different Test Set from Phase 12

```python
# Phase 12:
test_ds = CompressibleArithmeticDataset(
    num_samples=1000, min_ops=3, max_ops=8,   # ← max_ops=8
    max_seq_len=64, seed=456                   # ← seed=456
)

# Phase 14:
test_ds = CompressibleArithmeticDataset(
    num_samples=1000, min_ops=3, max_ops=6,   # ← max_ops=6 (easier!)
    max_seq_len=64, seed=999                   # ← seed=999
)
```

Phase 14's test set is **easier** (fewer operations) and **different** (different seed). Cross-phase accuracy comparisons are invalid.

---

## The Real Picture

Stripping away the bugs, here's what the data actually says:

### What's real:

1. **Op Detector probe accuracy 96.5%** on Phase 12 backbone (vs 74.8% on Phase 13b backbone). The un-gated model encodes operations better than the explicitly-trained one. This is genuine and important.

2. **Three-signal discrimination works:**
   ```
   True Convergence:  Sc=0.000, ∇H=-0.121
   False Convergence: Sc=0.334, ∇H=+0.003
   ```
   $S_c$ perfectly separates the two classes (0.000 vs 0.334). This validates the cycling score as a discriminator, even without AUC computation.

3. **Confusion Head achieves F1=0.974.** The learned detector works excellently. It just can't act on its knowledge because the gating mechanism is too weak.

4. **Corrected purity 0.773** — preserved under extended generation. The geometric structure survives.

### What's a bug artifact:

1. **97.4% no-halt accuracy** — likely caused by CompressibleGenerator's halt-before-append behavior, not by the model being "released" from premature termination. Need to verify by inspecting CompressibleGenerator.generate().

2. **"cycles_stuck" diagnosis** — wrong decision from incomplete logic.

3. **14C showing zero improvement** — caused by `max_boost=0.04` being too small, not by the approach being wrong.

---

## Plan of Action

### Step 0 (5 min): Inspect CompressibleGenerator.generate()

```bash
grep -n "halt" src/compressible_task.py | head -20
```

Look for the order of operations: does it check `halt_conf > threshold` before or after appending `next_token`? This determines whether the 97.4% is real.

### Step 1 (if halt-before-append confirmed): Fix CompressibleGenerator

```python
# WRONG (current, suspected):
if halt_conf > threshold: break
next_token = logits.argmax()
generated_ids.append(next_token)

# CORRECT (append first, then check):
next_token = logits.argmax()
generated_ids.append(next_token)
if halt_conf > threshold: break
```

Then re-run Phase 12 eval and Phase 14 14A-0 with the fixed generator. The "true" baseline accuracy will likely rise significantly (possibly to 97%+).

### Step 2 (if the 97.4% is real): The paper story simplifies dramatically

The model already produces the correct answer. The halt is the only problem. Fix the halt → done. No need for 14F (basin deepening), no need for the Confusion Head, no need for cycle breaking.

### Step 3 (if the 97.4% is a parsing artifact): Fix max_boost and re-run 14C

```python
# Instead of:
base_threshold=0.95, max_boost=0.04

# Use comparison-based stopping:
if halt_conf > effective_threshold and confusion < 0.3:
    # Only halt when confusion is LOW
    break
```

This reverses the logic: instead of trying to raise the threshold above the halt confidence (impossible when halt ≈ 0.9999), make halting **conditional on confusion being absent**. The Confusion Head becomes a veto, not a threshold modifier.

---

## The One Number That Survived All Bugs

Regardless of everything else, $S_c$ cleanly separates true from false convergence:

$$S_c^{\text{true}} = 0.000 \quad \text{vs} \quad S_c^{\text{false}} = 0.334$$

This signal is computed from raw state vectors during generation, independent of halt thresholds, answer parsing, or test set composition. It's the most trustworthy result in Phase 14, and it validates the three-signal framework from Appendix A.