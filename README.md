# PNA-SSM: Thermodynamic Loss on State Space Models

An experimental investigation into whether State Space Models (SSMs) show synergistic efficiency gains over Transformers when trained with a thermodynamic loss function. The core hypothesis is that SSMs' built-in Markovian compression naturally aligns with thermodynamic training objectives, producing measurable phenomena in the model's internal state dynamics.

## Key Findings

**SSM halt heads genuinely track answer uncertainty** — unlike Transformer halt heads that pattern-match on syntax, SSM halt heads exhibit strong negative correlation (r = -0.73) between answer entropy and halt confidence, demonstrating real uncertainty tracking.

**SSM state entropy collapse is real and synchronized** — the SSM recurrent state compresses as the model resolves uncertainty (r = -0.84 correlation with halt confidence), with the halt head firing *before* state entropy fully collapses, suggesting predictive rather than reactive behavior.

**Halt detection transfers across tasks** — SSM halt heads achieve 94.5% F1 when transferred from parity to arithmetic, vs 86.4% for Transformers, consistent with task-general uncertainty tracking.

**Variable-depth reasoning confirmed** — on compressible arithmetic chains, the model produces 48% fewer reasoning tokens on heavily compressible inputs while achieving higher accuracy (65% vs 40%).

**Algebraic rule constraints hit a transfer ceiling** — soft rule constraints minimize to near-zero during teacher-forced training but yield only 16.6% geodesic purity during autoregressive generation, exposing a fundamental teacher-forced/autoregressive gap.

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
| 12 | Rule-initialized models | Constraints minimize in training but limited autoregressive transfer |

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
  entropy_halt_correlation.py   # Linear probe entropy analysis (Phase 8)
  ssm_state_entropy_collapse.py # State entropy collapse detection (Phase 9)
  cross_task_transfer.py # Cross-task halt transfer (Phase 10)
  compressible_task.py   # Variable-depth compressible arithmetic (Phase 11)
  rule_initialization.py # Rule-initialized models with algebraic constraints (Phase 12)
  visualize.py           # Figure generation utilities

results/                 # Model checkpoints (.pt) and experiment results (.json)
figures/                 # Publication-ready figures (fig0-fig21)
documentation/           # Design docs and architecture references
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
python src/ablation_halt.py              # Phase 7
python src/entropy_halt_correlation.py   # Phase 8
python src/ssm_state_entropy_collapse.py # Phase 9
python src/cross_task_transfer.py        # Phase 10
python src/compressible_task.py          # Phase 11
python src/rule_initialization.py        # Phase 12
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

Key finding: **L_halt alone is sufficient** for halt calibration. The energy and state penalties degrade autoregressive accuracy without improving halt F1 (Phase 7 ablation).

## The USS Model (Uncertainty-State Synchrony)

The central emergent phenomenon, refined across Phases 8-11:

1. **Exploration**: High state entropy, halt silent
2. **Convergence**: State entropy falling, halt confidence rising (predictive)
3. **Basin Entry**: Entropy low, halt fires — answer determined
4. **Fixed Point** (3.6% of cases): State settles, 94% accuracy
5. **Limit Cycle** (96.4% of cases): State oscillates within basin, 47% accuracy

The halt head detects basin entry rather than fixed-point convergence. True convergence is rare but highly predictive of correctness.
