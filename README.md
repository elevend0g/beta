# PNA-SSM: Thermodynamic Loss on State Space Models

An experimental investigation into whether State Space Models (SSMs) show synergistic efficiency gains over Transformers when trained with a thermodynamic loss function. The core hypothesis is that SSMs' built-in Markovian compression naturally aligns with thermodynamic training objectives, producing measurable phenomena in the model's internal state dynamics.

## Key Findings

**SSM halt heads genuinely track answer uncertainty** — unlike Transformer halt heads that pattern-match on syntax, SSM halt heads exhibit strong negative correlation (r = -0.73) between answer entropy and halt confidence, demonstrating real uncertainty tracking.

**SSM state entropy collapse is real and synchronized** — the SSM recurrent state compresses as the model resolves uncertainty (r = -0.84 correlation with halt confidence), with the halt head firing *before* state entropy fully collapses, suggesting predictive rather than reactive behavior.

**Halt detection transfers across tasks** — SSM halt heads achieve 94.5% F1 when transferred from parity to arithmetic, vs 86.4% for Transformers, consistent with task-general uncertainty tracking.

**Variable-depth reasoning confirmed** — on compressible arithmetic chains, the model produces 48% fewer reasoning tokens on heavily compressible inputs while achieving higher accuracy (65% vs 40%).

**Algebraic rule constraints hit a transfer ceiling** — soft rule constraints minimize to near-zero during teacher-forced training but yield only 16.6% geodesic purity during autoregressive generation, exposing a fundamental teacher-forced/autoregressive gap.

**Halt-gating with a confusion head recovers lost accuracy** — a learned confusion head (F1=0.97) detects when the SSM is stuck in a limit cycle and suppresses premature halting. Halt gating raises overall autoregressive accuracy from 51.7% to 73.6%. With basin-depth training the best model reaches 84.6% overall and 89.7% on the hardest reasoning tier. Remaining errors (100%) are wrong-basin convergence, not limit cycles — a model capacity limitation, not a halting problem.

**Proprioceptive coupling is continuously controllable** — a 30-model 2D sweep over the thermodynamic hyperparameters (α × β) maps the coupling response surface on parity. Near-optimal is α ≈ 0.01–0.05, β ≈ 0.05–0.10. Increasing d_state from 16 to 64 does not improve coupling, indicating d_state=16 is not the geometric bottleneck.

**Proprioception partially generalizes across domains** — on symbolic sorting (structurally different from parity: alphabet symbols, comparison operations, variable-length output), the specificity gradient reproduces: C_sort is reactive (r=+0.317) and E_sort is anticipatory (r=-0.450, τ_drv=-2, 100% frac_negative). Coupling strength is weaker than parity (r=-0.836), confirming partial generalization — the mechanism is domain-trainable but coupling strength varies with task structure.

## Experimental Design

4-way comparison crossing architecture (Transformer vs SSM) with loss function (Cross-Entropy vs Thermodynamic):

| Group | Architecture | Loss | Accuracy | Halt F1 | OOD Generalization |
|-------|-------------|------|:--------:|:-------:|:------------------:|
| A | Transformer | Cross-Entropy | 100.0% | 48.1% | ~51% |
| B | Transformer | Thermodynamic | 99.8% | 98.8% | ~49% |
| C | SSM (Mamba) | Cross-Entropy | 99.9% | 0.0% | 100% |
| D | SSM (Mamba) | Thermodynamic | 99.7% | 99.2% | 100% |

All groups use ~5M parameters on symbolic parity and arithmetic tasks.

## Phases

| Phase | Description | Key Result |
|-------|-------------|------------|
| 1-2 | Architecture + loss implementation | 4-way matched-param setup |
| 3-4 | Training + evaluation | All groups >99.7% teacher-forced accuracy |
| 5 | Generalization | SSMs generalize perfectly to OOD lengths; Transformers collapse |
| 6 | Autoregressive generation | Accuracy-calibration tradeoff: thermodynamic loss costs ~20% free gen accuracy |
| 7 | Halt-only ablation | L_halt alone drives halt F1; energy/state penalties are harmful |
| 8 | Entropy-halt correlation | SSM halt heads genuinely track uncertainty (r=-0.73); Transformers don't |
| 9 | State entropy collapse | Three-signal synchrony confirmed; halt leads state collapse by 1-2 steps |
| 10 | Cross-task transfer | SSM halt transfers at 94.5% F1 vs Transformer 86.4% |
| 11 | Compressible reasoning | 48% fewer tokens on compressible chains; USS basin analysis |
| 12 | Rule-initialized models | Soft constraints → 83% geodesic purity; constraints minimize to near-zero in training |
| 13 | SSM state gating (negative) | Invasive SSM modification collapses geodesic purity from 83% → 18%; soft constraints are necessary |
| 14 | Halt control + confusion detection | Confusion head F1=0.97; halt gating raises autoregressive accuracy 51.7% → 73.6% |
| 14f | Basin-depth training | 84.6% overall accuracy; 89.7% on hard tier (best autoregressive result) |
| 15 | Epistemic controller | Perturbation strategies (re-injection, temp spike, best-of-N) fail to improve over halt veto; 100% residual errors are wrong-basin convergence |
| 16 | Arithmetic failure analysis | Multi-step arithmetic at 4% accuracy; task too complex for 5M parameter model |
| 17 | Proprioception reproduction | Exact replication: r=-0.836, τ_drv=-2.032; derivative xcorr confirmed as the cleaner anticipatory measure |
| 17b | Specificity gradient | C→D→E gradient (r: -0.290→-0.725→-0.836) confirmed as training-dependent, not architectural |
| 18 | Hyperparameter landscape | 2D (α×β) grid maps coupling; α=0.01–0.05 + β=0.05–0.10 near-optimal; d_state=16 not the geometric bottleneck |
| 19 | Cross-domain generalization | Sorting: specificity gradient reproduces, E_sort r=-0.450 (τ_drv=-2); partial generalization confirmed (Outcome 2/3) |

## Project Structure

```
src/
  models.py              # PNA_SSM (S6Block/MambaBlock) and TransformerModel
  losses.py              # ThermodynamicLoss: L_ce + α·L_energy + β·L_halt + γ·L_state
  dataset.py             # Symbolic parity dataset with reasoning chains
  train.py               # Training loop and evaluation
  generate.py            # Autoregressive generation (FreeGenerator)
  eval_generation.py     # Generation evaluation pipeline
  ablation_halt.py       # Halt-only ablation (Phase 7)
  entropy_halt_correlation.py      # Linear probe entropy analysis (Phase 8)
  ssm_state_entropy_collapse.py    # State entropy collapse detection (Phase 9)
  cross_task_transfer.py           # Cross-task halt transfer (Phase 10)
  compressible_task.py             # Variable-depth compressible arithmetic (Phase 11)
  rule_initialization.py           # Rule-initialized models with algebraic constraints (Phase 12)
  phase14_halt_control.py          # Halt control + confusion head (Phase 14)
  phase14f_basin_depth.py          # Basin-depth training (Phase 14f)
  phase15_epistemic_controller.py  # Perturbation ablation (Phase 15)
  phase16_failure_analysis.py      # Arithmetic failure analysis (Phase 16)
  phase17_proprioception_repro.py  # Phase 9 exact reproduction (Phase 17)
  phase17b_proprioception_comparison.py  # Specificity gradient (Phase 17b)
  phase18_thermodynamic_control.py       # 2D hyperparameter sweep (Phase 18)
  phase18b_focused_grid.py               # Focused grid near optimal (Phase 18b)
  phase18c_dstate_sweep.py               # d_state dimensionality sweep (Phase 18c)
  phase19_cross_domain.py                # Cross-domain generalization on sorting (Phase 19)
  visualize.py           # Figure generation utilities

results/                 # Model checkpoints (.pt) and experiment results (.json)
figures/                 # Publication-ready figures (fig0–fig_p19_5)
documentation/           # Design docs and phase protocols
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA-capable GPU.

## Usage

Train the 4-way comparison (Phases 1-5):
```bash
python src/train.py
```

Run autoregressive generation evaluation:
```bash
python src/eval_generation.py
```

Run individual experiment phases:
```bash
python src/ablation_halt.py                      # Phase 7
python src/entropy_halt_correlation.py           # Phase 8
python src/ssm_state_entropy_collapse.py         # Phase 9
python src/cross_task_transfer.py                # Phase 10
python src/compressible_task.py                  # Phase 11
python src/rule_initialization.py                # Phase 12
python src/phase14_halt_control.py               # Phase 14
python src/phase14f_basin_depth.py               # Phase 14f
python src/phase15_epistemic_controller.py       # Phase 15
python src/phase16_failure_analysis.py           # Phase 16
python src/phase17_proprioception_repro.py       # Phase 17
python src/phase17b_proprioception_comparison.py # Phase 17b
python src/phase18_thermodynamic_control.py      # Phase 18
python src/phase18b_focused_grid.py              # Phase 18b
python src/phase18c_dstate_sweep.py              # Phase 18c
python src/phase19_cross_domain.py               # Phase 19
```

## Thermodynamic Loss

The multi-component loss function:

```
L_th = L_ce + α·L_energy + β·L_halt + γ·L_state
```

- **L_ce**: Standard cross-entropy for next-token prediction
- **L_energy** (α=0.05): Penalizes token entropy during reasoning, rewarding certainty
- **L_halt** (β=0.1): Supervised halt confidence — ramps to 1.0 at the Result token
- **L_state** (γ=0.05): Penalizes high SSM state entropy, encouraging state compression

Key finding: **L_halt alone is sufficient** for halt calibration. The energy and state penalties degrade autoregressive accuracy without improving halt F1 (Phase 7 ablation). Near-optimal hyperparameters from the Phase 18 control landscape: α=0.01–0.05, β=0.05–0.10.

## The USS Model (Uncertainty-State Synchrony)

The central emergent phenomenon, refined across Phases 8-19:

1. **Exploration**: High state entropy, halt silent
2. **Convergence**: State entropy falling, halt confidence rising (predictive)
3. **Basin Entry**: Entropy low, halt fires — answer determined
4. **Fixed Point** (3.9% of cases): State settles, 89.7% accuracy
5. **Limit Cycle** (96.1% of cases): State oscillates within basin, 72.9% accuracy

The halt head detects basin entry rather than fixed-point convergence. True convergence is rare but highly predictive of correctness. The dominant failure mode in autoregressive generation is not premature halting (which a confusion head can correct) but convergence to the wrong attractor basin — a training capacity limit, not a halting problem.

**Cross-domain validity (Phase 19)**: On symbolic sorting, the USS structure reproduces. The specificity gradient (C_sort reactive → E_sort anticipatory) confirms that thermodynamic training induces proprioception across structured reasoning domains. Coupling strength is task-dependent (r=-0.450 sorting vs r=-0.836 parity), likely reflecting the rigidity and determinism of the optimal reasoning path.
