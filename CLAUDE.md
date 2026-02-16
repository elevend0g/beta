# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PNA-SSM: A machine learning research experiment testing whether State Space Models (SSMs) show synergistic efficiency gains over Transformers when trained with a thermodynamic loss function. The core hypothesis is that SSMs' built-in Markovian compression naturally aligns with thermodynamic training objectives.

## Running the Code

No build system or package manager is configured yet. Individual modules can be validated via their `__main__` blocks:

```bash
python documentation/pna_ssm_architecture.py    # Test forward pass, verify ~7M params
python documentation/ssm_thermodynamic_loss.py   # Test loss computation on mock tensors
```

### Dependencies (not yet in a requirements file)
PyTorch 2.0+, einops, scipy, pandas, matplotlib, seaborn, numpy

## Architecture

### 4-Way Experimental Design
| Group | Architecture | Loss | Purpose |
|-------|-------------|------|---------|
| A | Transformer | Cross-Entropy | Baseline |
| B | Transformer | Thermodynamic | Existing PNA result |
| C | SSM (Mamba) | Cross-Entropy | Architecture control |
| D | SSM (Mamba) | Thermodynamic | Hypothesis test |

All groups use ~7M parameters on the same dataset (symbolic parity + arithmetic).

### Key Modules (`documentation/`)
- **pna_ssm_architecture.py** — SSM model: S6Block → MambaBlock → PNA_SSM (6 layers, d_model=512, d_state=16). Two output heads: token prediction and halt prediction (halt head uses SSM state directly).
- **ssm_thermodynamic_loss.py** — Multi-component loss: `L_th = L_ce + α·L_energy + β·L_halt + γ·L_state`. Includes SSMAdaptiveGovernor for dynamic weight adjustment during training.
- **pna_ssm_experiment_protocol.py** — Full experimental protocol with training config, 8+ evaluation metrics, success criteria (6 conditions), and statistical tests.
- **pna_ssm_visualization.py** — Publication-ready figure generation; requires completed training results to run.

### Key Concept: State Entropy
Unlike Transformers, SSMs expose an internal state `h_t` whose entropy can be measured and optimized independently from token entropy. The `L_state` loss component (γ weight) penalizes high state uncertainty during reasoning — this is the novel SSM-specific contribution. The hypothesis predicts state entropy collapse will *lead* token entropy collapse in time.

### Public Interfaces
```python
model = PNA_SSM(vocab_size, d_model=512, n_layers=6, d_state=16)
outputs = model(input_ids)  # Returns: {"logits", "halt_confidence", "final_state"}

loss_fn = SSMThermodynamicLoss(alpha=0.05, beta=0.1, gamma=0.05)
losses = loss_fn(logits, targets, halt_confidence, states_sequence,
                 reasoning_mask, result_token_positions)
# Returns: {"total", "ce_loss", "energy_loss", "halt_loss", "state_loss",
#           "token_delta_h", "state_delta_h", "state_leads", ...}
```
