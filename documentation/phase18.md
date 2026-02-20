# Phase 18: Thermodynamic Control of Anticipatory State-Halt Coupling

**Version**: 1.0  
**Date**: February 2026  
**Builds on**: Phase 17 (exact reproduction of r = -0.836, τ = -2.032)  
**Central question**: Can the proprioceptive coupling between state entropy and halt confidence be systematically controlled and amplified through training hyperparameters?

---

## 1. Background and Motivation

Phase 17b established a clean specificity gradient across three training conditions:

| Group | Training | mean r | τ threshold | drv τ* |
|---|---|---|---|---|
| C | SSM + CE (no halt) | -0.290 | 0.000 | +2 |
| D | SSM + L_th (thermo) | -0.725 | -1.449 | -2 |
| E_ssm | SSM + halt (explicit) | -0.836 | -2.032 | -2 |

Two findings motivate Phase 18:

**Finding 1 — Thermodynamic induction.** Group D achieves strong anticipatory coupling (r = -0.725, τ = -1.449) without any explicit halt supervision. The energy penalty in L_th induces proprioception as a side effect of efficiency optimization. This means α (energy penalty weight) is a potential control lever.

**Finding 2 — Explicit halt as amplifier.** E_ssm improves on D by approximately 0.11 in r and 0.58 in τ by adding explicit halt supervision (β). The explicit halt term sharpens an already-present thermodynamic signal rather than creating a new mechanism. This means β is a second independent control lever.

**Hypothesis**: r and τ are monotonically controllable functions of α and β, with accuracy as the binding constraint. There exists a 2D control landscape (α × β) over which proprioceptive coupling can be tuned continuously.

---

## 2. Experimental Design

### 2.1 Primary Experiment: 2D Hyperparameter Sweep

Train a grid of models varying α and β independently, holding all other parameters fixed at E_ssm values.

**Grid specification:**

| Hyperparameter | Values | Rationale |
|---|---|---|
| α (energy penalty) | 0.0, 0.01, 0.05, 0.10, 0.20, 0.50 | Spans null through high-pressure regime |
| β (halt loss weight) | 0.0, 0.05, 0.10, 0.20, 0.40 | Spans null through strong explicit supervision |

This produces a 6 × 5 = 30-model grid. The corners define the experimental space:

- (α=0, β=0): Group C baseline — no thermodynamic or halt signal
- (α=0.05, β=0): Group D baseline — thermodynamic only
- (α=0, β=0.10): Halt-only control — explicit supervision without thermodynamic pressure
- (α=0.05, β=0.10): Group E_ssm baseline — the known reference point

The halt-only control (α=0, β>0) is new and critical. It isolates whether explicit halt supervision alone produces anticipatory coupling, or whether thermodynamic pressure is necessary as a foundation.

**Fixed parameters** (held constant across all 30 models):
- Architecture: SSM, d_model=512, d_state=16, same as Groups C/D/E_ssm
- Dataset: Parity, 2–8 bits, same train/val/test splits
- Optimizer, learning rate, batch size, epochs: identical to Phase 9 training
- γ (state entropy penalty): 0.0 — excluded to avoid the interference documented in Phase 17b

### 2.2 Secondary Experiment: d_state Dimensionality Sweep

The proprioceptive signal is geometrically encoded in h_t. Larger d_state gives the recurrent state more degrees of freedom to represent computational trajectory. Test whether d_state amplifies the ceiling of achievable coupling independent of loss function tuning.

**Models**: Train 5 variants of E_ssm (α=0.05, β=0.10) with d_state ∈ {8, 16, 32, 64, 128}.

d_model is adjusted to maintain approximately constant total parameter count (~5M) across variants.

**Expected pattern**: If proprioception is geometrically bottlenecked by d_state, r and τ should improve as d_state increases up to some saturation point. If d_state is not the bottleneck, the curve flattens early and loss function tuning remains the primary lever.

### 2.3 Accuracy Filter (Critical Control)

All analysis is conditioned on task accuracy ≥ 95% on the held-out parity test set. Models below this threshold are excluded from the response surface analysis and flagged separately.

This is not optional. Without the accuracy filter, the response surface conflates "stronger proprioceptive coupling" with "the model stopped solving the task." A high-α model that achieves r = -0.95 with 60% accuracy is not a positive result.

---

## 3. Measurements

Every model in the grid is evaluated with the full Phase 17 protocol, producing four metrics per model:

**Primary metrics:**
- r: mean instantaneous Pearson correlation between state entropy and halt confidence (791 examples)
- τ_threshold: mean threshold lag (50% / 0.5 crossings)
- τ_derivative: peak lag from derivative cross-correlation (the anticipatory measure)
- frac_significant: fraction of examples with |r| > 0.3

**Secondary metrics:**
- Task accuracy (parity test set)
- Halt precision and recall at result_pos
- Training stability (gradient norm variance across final 20% of training)

**Probe metric:**
- Probe accuracy at result_pos (confirms d_model states encode the answer — expected ≈99% across all groups given Phase 17b consistency)

---

## 4. Analysis Plan

### 4.1 Response Surface

For the 30 models passing the accuracy filter, fit a response surface r(α, β) and τ(α, β). The primary question is whether these surfaces are monotone, exhibit interaction effects, or show saturation.

Specific contrasts of interest:

**Contrast 1 — α main effect** (β=0 column): Does thermodynamic pressure alone produce monotonically increasing coupling? This isolates the induction mechanism.

**Contrast 2 — β main effect** (α=0 column): Does explicit halt supervision alone produce anticipatory coupling? This tests whether thermodynamic induction is a prerequisite or merely one pathway.

**Contrast 3 — Interaction** (α × β): Is the improvement from combining both losses additive, subadditive (redundant), or superadditive (synergistic)? The Phase 17b data (D → E_ssm improvement of ~0.11 in r) suggests additive, but the grid will characterize the full interaction.

**Contrast 4 — Ceiling**: At what (α, β) does r saturate? Is E_ssm near the ceiling or far from it?

### 4.2 d_state Analysis

Plot r and τ_derivative as a function of log(d_state) at fixed α=0.05, β=0.10. Fit a saturation curve. The inflection point indicates the geometric bottleneck: below it, more state dimensions improve proprioception; above it, loss function tuning is the binding constraint.

### 4.3 Derivative Lag Stability

For each model, report τ_derivative alongside τ_threshold. The Phase 17 finding was that derivative xcorr (τ = -2) was a cleaner and more interpretable measure than threshold lag. Track whether both metrics move in concert across the grid, or whether they dissociate under certain (α, β) combinations.

Dissociation — where τ_threshold increases but τ_derivative does not — would indicate that α/β is shifting the threshold crossing position without affecting the genuine anticipatory dynamics. That would be an important negative finding.

---

## 5. Falsification Criteria

Phase 18 produces a negative result under the following conditions:

**Negative result 1**: The halt-only column (α=0, β>0) shows comparable r and τ to the full (α>0, β>0) models. This would mean thermodynamic pressure is not necessary for proprioceptive induction — explicit supervision alone is sufficient. The thermodynamic control framing would require revision.

**Negative result 2**: r and τ do not vary monotonically with α in the β=0 column. Non-monotone behavior would indicate that the energy penalty does not directly control coupling strength, and some other mechanism is responsible for the C → D gradient.

**Negative result 3**: Accuracy degrades before proprioceptive saturation. If the accuracy filter eliminates all high-α or high-β models, the "control" interpretation collapses — the coupling can only be amplified by breaking the task.

**Negative result 4**: The d_state sweep shows no improvement across the tested range. This would mean proprioception is not geometrically bottlenecked by state dimensionality, and the recurrent state at d_state=16 already has sufficient capacity.

All four negative results are scientifically informative and should be reported as findings, not failures.

---

## 6. Success Criteria

Phase 18 confirms the thermodynamic control hypothesis if:

1. The β=0 column (thermodynamic only) shows monotonically increasing r from α=0.01 to at least α=0.20, with all models maintaining ≥95% accuracy.
2. The α=0 column (halt only) shows meaningfully weaker coupling than the diagonal (α>0, β>0), confirming thermodynamic pressure is not interchangeable with explicit supervision.
3. τ_derivative tracks τ_threshold across the grid (no dissociation), confirming the two measures are reading the same underlying phenomenon.
4. At least one (α, β) combination produces r < -0.836 (the E_ssm reference) with ≥95% accuracy, demonstrating that E_ssm is not the ceiling.

---

## 7. Connection to Existing Results

Phase 18 extends the established chain:

- **Phase 9**: Proprioceptive coupling exists in E_ssm (r = -0.836, τ = -2.03)
- **Phase 17**: Exact reproduction confirmed, derivative xcorr validates anticipatory mechanism
- **Phase 17b**: Coupling is training-specific, not architectural; gradient across C → D → E_ssm is monotone
- **Phase 18**: Characterizes the full control landscape; determines whether coupling can be amplified beyond E_ssm and whether thermodynamic pressure is the primary lever

The 2D grid provides what the three-point C/D/E_ssm comparison could not: a continuous characterization of the response surface rather than three isolated samples.

---

## 8. Output Artifacts

Each of the 30 primary models produces:

- `results/phase18_grid_α{a}_β{b}_model.pt` — checkpoint
- `results/phase18_grid_α{a}_β{b}_metrics.json` — r, τ, accuracy, all Phase 17 metrics

Aggregate outputs:

- `results/phase18_response_surface.json` — full grid metrics
- `figures/fig_p18_1_r_surface.png` — r(α, β) heatmap with accuracy filter overlay
- `figures/fig_p18_2_tau_surface.png` — τ(α, β) heatmap
- `figures/fig_p18_3_alpha_main.png` — r and τ vs α at β=0 (induction curve)
- `figures/fig_p18_4_beta_main.png` — r and τ vs β at α=0 (amplification curve)
- `figures/fig_p18_5_dstate.png` — r and τ vs d_state at fixed (α=0.05, β=0.10)
- `figures/fig_p18_6_interaction.png` — interaction contrast: additive vs. superadditive

---

*Phase 18 Protocol v1.0 — February 2026*  
*Builds on: Phase 9 · Phase 17 · Phase 17b*