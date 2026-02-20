# Phase 19: Cross-Domain Proprioception

**Version**: 1.0  
**Date**: February 2026  
**Builds on**: Phase 9 · Phase 17 · Phase 17b · Phase 18 (a/b/c)  
**Central question**: Is architectural proprioception a general property of thermodynamically-trained SSMs reasoning over structured sequential tasks, or is it specific to the parity domain?

---

## 1. Background and Motivation

The complete Phase 9–18 arc established the following about the E_ssm model on parity:

- Strong anticipatory state-halt coupling: r = -0.836, τ_threshold = -2.032, τ_derivative = -2.0
- The signal is training-dependent, not architectural (Phase 17b)
- Thermodynamic pressure induces the signal; explicit halt supervision amplifies it (Phase 17b, 18a)
- The coupling has a measurable ceiling within this architecture and task (Phase 18c)
- d_state=16, α=0.05, β=0.10 is near-optimal for the parity domain (Phase 18c)

Every result in this arc was obtained on a single task with rigid deterministic structure and a known optimal reasoning path. This is a constraint on the generality of all claims made so far.

Phase 19 tests whether the proprioceptive mechanism is domain-general or domain-specific by introducing a second structured reasoning task and measuring whether the same signal emerges under the same training conditions.

---

## 2. Task Selection

### 2.1 Rejection of Multi-Step Arithmetic

Multi-step arithmetic with negative numbers and operator precedence failed in Phase 16 at 4% accuracy. The model could not solve the task, meaning internal states did not encode meaningful reasoning trajectories. Proprioception cannot be measured in a model that isn't reasoning. Arithmetic is excluded.

### 2.2 Requirements for a Valid Cross-Domain Task

A suitable Phase 19 task must satisfy four criteria:

1. **Solvable at high accuracy** (≥95%) by the same 5M parameter SSM architecture trained with E_ssm hyperparameters. This is the binding constraint from Phase 16.
2. **Structurally different from parity** at the surface level — different vocabulary, different operation type, different output format — so that transfer cannot be attributed to shared surface features.
3. **Has a deterministic optimal reasoning path** of known minimum length, so that thermodynamic waste is measurable and the entropy collapse structure is predictable.
4. **Sequential and compositional** — the solution requires multiple dependent steps, each of which reduces residual uncertainty about the final answer. Tasks that can be solved in one step have no reasoning trajectory to sense.

### 2.3 Selected Task: Symbolic Sequence Sorting

**Task definition**: Given a sequence of symbols drawn from a small ordered alphabet (e.g., letters A–F), produce the sorted sequence using an explicit bubble-sort or insertion-sort style reasoning chain.

**Example**:

```
Input: D B F A C E
Reasoning: D>B→swap, D<F→keep, F>A→swap, F>C→swap, F>E→swap ...
Result: A B C D E F <HALT>
```

**Why this satisfies the criteria**:

- Solvable at high accuracy: The operation (comparison and swap) is simpler than arithmetic. A character-level SSM can learn it reliably.
- Structurally different from parity: Alphabet symbols rather than bits, comparison operations rather than XOR, variable-length output rather than binary, sequence reordering rather than accumulation. No surface features are shared.
- Deterministic optimal path: The minimum number of comparisons for a sequence of length n is known (O(n log n) for optimal sorts, O(n²) worst case for bubble sort). Any reasoning chain longer than the minimum is thermodynamic waste.
- Sequential and compositional: Each comparison step reduces uncertainty about the final ordering. The entropy collapse structure should mirror parity's staircase pattern — a drop at each resolved comparison.

**Secondary task: Symbolic Stack Operations**

As a stretch goal, a second cross-domain task provides stronger generalization evidence if the primary task succeeds.

**Task definition**: Given a sequence of push/pop operations on a stack with small integer values, predict the final stack state.

```
Input: PUSH 3, PUSH 7, POP, PUSH 2, PUSH 5, POP
Reasoning: [3]→[3,7]→[3]→[3,2]→[3,2,5]→[3,2]
Result: 3 2 <HALT>
```

This task tests proprioception in a stateful rather than ordering context and provides a second independent data point on generalization.

---

## 3. Experimental Design

Phase 19 has three sub-experiments that together answer the generalization question from different angles.

### 3.1 Sub-experiment A: Independent Training (Primary Test)

Train fresh E_ssm-configuration models on the sorting task and measure whether the proprioceptive signal emerges independently of parity training.

**Models**: Train three groups on sorting, matching the Phase 9 group structure:

|Group|Training|Purpose|
|---|---|---|
|C_sort|SSM + CE only|Control — no thermodynamic or halt signal|
|D_sort|SSM + L_th (α=0.05, β=0.0)|Thermodynamic induction without explicit halt|
|E_sort|SSM + L_th + halt (α=0.05, β=0.10)|Full E_ssm configuration on new domain|

**Success criterion**: E_sort achieves r < -0.5 and τ_derivative < -1.0 with ≥95% accuracy. This reproduces the Phase 17b specificity gradient in a new domain.

**Falsification criterion**: E_sort achieves ≥95% accuracy but r ≥ -0.3 or τ_derivative ≥ 0. This would indicate proprioception is parity-specific.

### 3.2 Sub-experiment B: Zero-Shot Transfer (Mechanistic Test)

Load the trained E_ssm parity model (no fine-tuning) and evaluate its proprioceptive metrics on sorting examples. This tests whether the internal signal learned on parity generalizes immediately to a new domain or requires retraining.

**Expected outcome**: Near-zero or degraded coupling, since the halt head was trained on parity-specific entropy dynamics. A positive transfer result (r < -0.5, τ_drv < -1.0) would be a strong and surprising finding.

**What a negative result means**: The parity model's halt head learned parity-specific completion patterns. The proprioceptive mechanism is domain-trainable but not domain-transferable in its trained form. This is consistent with the general induction claim while limiting transfer claims.

### 3.3 Sub-experiment C: Few-Shot Adaptation (Practical Test)

Fine-tune the E_ssm parity model on a small number of sorting examples (100, 500, 1000) and measure how quickly the proprioceptive signal recovers toward E_sort levels.

This tests the practical question: if you have a model with trained proprioception in one domain, how much data does it take to adapt to a new domain? A fast adaptation curve (strong signal at 500 examples) suggests the underlying mechanism is general and the halt head is learning domain-specific surface features on top of a general proprioceptive capacity.

---

## 4. Dataset Specification

### 4.1 Sorting Dataset

**Alphabet**: 6 symbols (A–F), ordered A < B < C < D < E < F  
**Sequence length**: 3–8 symbols (matches parity's 2–8 bit range for comparability)  
**Reasoning format**: Explicit pairwise comparisons with swap decisions, one comparison per reasoning token  
**Halt position**: Immediately after the final sorted symbol, before any trailing tokens  
**Dataset size**: 8000 train / 1000 val / 1000 test (matches Phase 9 splits exactly)

**Generation**: Sequences are generated randomly with replacement. The reasoning chain follows a fixed bubble-sort pass order (left to right, repeat until no swaps) to ensure deterministic optimal paths.

**Density tiers**: As in Phase 9, three tiers by sequence length — T0 (3–4 symbols), T1 (5–6 symbols), T2 (7–8 symbols) — to test whether proprioceptive coupling scales with reasoning depth.

### 4.2 Stack Dataset (Sub-experiment C stretch goal)

**Operations**: PUSH (values 1–9) and POP  
**Sequence length**: 4–10 operations  
**Constraint**: No POP on empty stack (guaranteed valid sequences)  
**Dataset size**: Same as sorting

---

## 5. Measurements

All measurements use the full Phase 17 protocol, applied identically to Phase 9 parity measurements for direct comparability.

**Primary proprioceptive metrics** (per model):

- r: mean instantaneous Pearson r(state entropy, halt confidence) across 791 test examples
- τ_threshold: mean threshold lag (50% / 0.5 crossings)
- τ_derivative: peak lag from derivative cross-correlation
- frac_negative: fraction of examples with r < 0
- frac_significant: fraction of examples with |r| > 0.3

**Accuracy filter**: All proprioceptive metrics reported only for models with ≥95% accuracy on the sorting test set. Models below threshold are excluded and flagged.

**Tier analysis**: r and τ_drv reported separately for T0/T1/T2 tiers to test whether coupling strength scales with reasoning depth. If proprioception is genuinely tracking computational trajectory, longer sequences should show stronger or at least comparable coupling.

**Comparison table**: All results reported alongside Phase 9 parity reference values for direct comparison.

---

## 6. The Specificity Gradient Test

The most important analysis in Phase 19 is whether the C_sort → D_sort → E_sort gradient mirrors the Phase 17b parity gradient:

|Parity (Phase 17b)|Expected Sorting (Phase 19)|
|---|---|
|C: r=-0.290, τ_drv=+2 (reactive)|C_sort: r ≈ 0, τ_drv > 0|
|D: r=-0.725, τ_drv=-2 (anticipatory)|D_sort: r < -0.5, τ_drv < -1|
|E: r=-0.836, τ_drv=-2 (anticipatory)|E_sort: r < -0.7, τ_drv < -1.5|

If the gradient reproduces — particularly if C_sort is reactive and D_sort/E_sort are anticipatory — the mechanism is domain-general. The absolute values do not need to match parity exactly; the ordinal structure and the reactive-to-anticipatory transition are what matter.

---

## 7. Interpretation Framework

Four outcomes are possible, with distinct implications for each:

**Outcome 1 — Full generalization**: E_sort achieves r and τ_drv comparable to E_ssm parity, and the C→D→E gradient reproduces. Conclusion: proprioception is a domain-general property of thermodynamically-trained SSMs on structured sequential reasoning tasks. The parity results were not task-specific artifacts.

**Outcome 2 — Partial generalization**: The gradient reproduces (C_sort reactive, E_sort anticipatory) but coupling strength is weaker than parity (e.g., r ≈ -0.5 rather than -0.836). Conclusion: thermodynamic training induces proprioception across domains but coupling strength is task-dependent, possibly related to the rigidity and determinism of the optimal reasoning path.

**Outcome 3 — Trainable but not transferable**: E_sort achieves strong coupling when trained from scratch, but zero-shot transfer from the parity model fails. Conclusion: proprioception is domain-trainable but halt heads are domain-specific. The mechanism is general; its instantiation is not.

**Outcome 4 — Domain-specific**: E_sort fails to achieve meaningful coupling despite ≥95% accuracy. Conclusion: the parity results reflect parity-specific structure (binary XOR operations, fixed-length output, binary answer space). The proprioception claim requires significant qualification.

All four outcomes are scientifically meaningful and publishable with appropriate framing.

---

## 8. Relationship to Publication Strategy

Phase 19 determines the scope of the central claim in Paper 1:

- Outcome 1 or 2: "Thermodynamic training induces architectural proprioception in SSMs across structured reasoning domains."
- Outcome 3: "Thermodynamic training induces domain-specific proprioception; the mechanism generalizes but trained instances do not transfer."
- Outcome 4: "Architectural proprioception in parity-trained SSMs is a domain-specific phenomenon; generalization requires further investigation."

The research arc is complete after Phase 19 regardless of outcome. The question of generalization has a definitive answer, and the body of work from Phase 9 through Phase 19 constitutes a coherent empirical story with proper controls, reproduction, mechanistic characterization, control landscape mapping, and generalization testing.

---

## 9. Output Artifacts

**Per-model outputs** (9 models: C/D/E × sorting, plus transfer and adaptation models):

- `results/phase19_{group}_{task}_model.pt`
- `results/phase19_{group}_{task}_metrics.json`

**Aggregate outputs**:

- `results/phase19_comparison_table.json` — all metrics alongside parity reference
- `figures/fig_p19_1_gradient_comparison.png` — C/D/E gradient: parity vs. sorting side by side
- `figures/fig_p19_2_transfer_curve.png` — zero-shot and few-shot adaptation (Sub-experiments B/C)
- `figures/fig_p19_3_tier_analysis.png` — r and τ_drv by sequence length tier
- `figures/fig_p19_4_trajectory_gallery.png` — representative sorting examples with three-signal plots
- `figures/fig_p19_5_xcorr_comparison.png` — derivative xcorr for E_ssm parity vs. E_sort sorting

---

_Phase 19 Protocol v1.0 — February 2026_  
_Builds on: Phase 9 · Phase 17 · Phase 17b · Phase 18a · Phase 18b · Phase 18c_