# PNA-SSM Experiment Protocol
## Thermodynamic Loss in State Space Models vs. Transformers

**Hypothesis**: SSMs trained with thermodynamic loss (L_th) will exhibit superior halt calibration, 
entropy collapse dynamics, and training stability compared to Transformers with L_th, because the 
SSM's built-in Markovian compression aligns with the thermodynamic training objective.

---

## 1. Experimental Design (4-Way Comparison)

### Group A: Transformer + Cross-Entropy (Baseline)
- Architecture: 4-layer Transformer, 8 heads, d=512
- Loss: Standard cross-entropy
- Purpose: Control for standard training

### Group B: Transformer + Thermodynamic Loss (PNA v1)
- Architecture: Same as Group A
- Loss: L_th with adaptive α
- Purpose: Your existing result - Transformer with thermodynamic training

### Group C: SSM + Cross-Entropy (Architecture Control)
- Architecture: 6-layer Mamba-style SSM, d=512, d_state=16
- Loss: Standard cross-entropy
- Purpose: Isolate effect of SSM architecture alone

### Group D: SSM + Thermodynamic Loss (PNA-SSM Hypothesis)
- Architecture: Same as Group C
- Loss: L_th with adaptive α
- Purpose: **The test group** - does thermodynamic loss work better in SSMs?

---

## 2. Training Configuration

All groups share:
- **Dataset**: Symbolic parity (2-8 bits) + multi-step arithmetic
- **Train/Val/Test split**: 8000/1000/1000 examples
- **Batch size**: 32
- **Learning rate**: 3e-4 with cosine decay
- **Epochs**: 50 (with early stopping on validation)
- **Optimizer**: AdamW (β1=0.9, β2=0.999, weight_decay=0.01)

Thermodynamic loss parameters (Groups B & D):
- **Initial α**: 0.01 (very gentle start)
- **α range**: [0.01, 0.5] (adaptive governor adjusts within this)
- **β (halt weight)**: 0.1
- **Stagnation threshold**: ΔH < 0.02 bits/token

---

## 3. Primary Evaluation Metrics

### 3.1 Core Performance (All Groups)

| Metric | Definition | Success Criterion |
|--------|-----------|-------------------|
| **Accuracy** | Exact match on Result token | ≥95% for all groups |
| **Reasoning Length** | Tokens from Input: to Result: | Groups B,D < Groups A,C |
| **Halt F1** | Precision/recall of <HALT> placement | Groups B,D ≥ 90% |

### 3.2 Entropy Dynamics (All Groups)

| Metric | What It Measures | Expected Pattern |
|--------|-----------------|------------------|
| **Token Entropy Curve** | H(next_token) at each position | Group D: sharpest step-function |
| **Collapse Steepness** | Max slope of entropy drop | Group D > B > A,C |
| **Entropy Floor** | Final H before <HALT> | All thermodynamic groups < 0.1 bits |

### 3.3 SSM-Specific Metrics (Groups C & D Only)

These metrics are **unique to SSMs** and test whether thermodynamic loss better exploits the state structure:

| Metric | Definition | Hypothesis |
|--------|-----------|------------|
| **State Entropy Collapse** | H(h_t) over sequence | Group D shows step-function collapse in STATE space, not just token space |
| **State Utilization** | Effective dimensionality of h_t | Group D uses fewer dimensions more efficiently |
| **State-to-Answer Correlation** | How much h_t "knows" about final answer | Group D's state converges to answer faster |

**Computing State Entropy:**

```python
def compute_state_entropy_trajectory(model, example):
    """
    Track how the SSM state h_t evolves during reasoning.
    
    For SSMs only: we can measure the entropy of the state itself,
    not just the token predictions.
    
    Hypothesis: In PNA-SSM (Group D), the state entropy should collapse
    in a step-function pattern that mirrors the token entropy collapse.
    """
    model.eval()
    states = []
    
    with torch.no_grad():
        for t in range(1, len(example.tokens)):
            input_ids = torch.tensor([example.tokens[:t]])
            outputs = model(input_ids)
            
            # Get final SSM state
            state = outputs['final_state']  # [1, d_state]
            
            # Compute state entropy
            # Method 1: Treat state as distribution
            state_probs = F.softmax(state, dim=-1)
            entropy = -(state_probs * torch.log2(state_probs + 1e-9)).sum().item()
            
            states.append({
                'position': t,
                'state_entropy': entropy,
                'state_vector': state.cpu().numpy()
            })
    
    return states
```

### 3.4 Training Stability Under Pressure

Test hypothesis: SSMs handle high thermodynamic pressure (high α) better than Transformers.

**Procedure:**
1. For Groups B (Transformer+L_th) and D (SSM+L_th):
2. Run additional training sweep with fixed α ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
3. Measure training stability (gradient norm variance, loss volatility)

**Expected Result:** Group D maintains stable training at higher α values where Group B becomes unstable.

---

## 4. Extended Generalization Test

Test length generalization: train on short sequences, test on longer ones.

**Setup:**
- **Training**: Parity on 2-5 bits only
- **Test**: Parity on 6-10 bits (out of distribution)

**Hypothesis:** SSMs should generalize better to longer sequences because:
1. Linear scaling in sequence length (vs quadratic for Transformers)
2. State-based reasoning less dependent on absolute position

| Group | Test Accuracy (6-10 bits) | Reasoning Length Growth |
|-------|---------------------------|------------------------|
| A (Trans+CE) | Baseline | Linear |
| B (Trans+L_th) | Better than A | Sub-linear (thermodynamic pressure) |
| C (SSM+CE) | Better than A | Sub-linear (SSM architecture) |
| D (SSM+L_th) | **Best** | **Most efficient** (compound effect) |

---

## 5. Detailed Analysis Plan

### 5.1 Entropy Collapse Visualization

For each group, generate the following plots:

**Plot 1: Token Entropy Trajectories (All Groups)**
- X-axis: Token position in sequence
- Y-axis: H(next_token)
- Overlay: Example-level trajectories (thin lines) + group mean (thick line)
- Expected: Group D shows sharpest step-function

**Plot 2: State Entropy Trajectories (SSM Groups Only)**
- Same as Plot 1, but measuring H(h_t) instead of H(next_token)
- Expected: Group D shows cleaner collapse than Group C

**Plot 3: Dual Entropy (SSM+L_th Only)**
- Two Y-axes: H(token) and H(state) on same plot
- Expected: State entropy leads token entropy (state "decides" before generating token)

### 5.2 Statistical Comparisons

All comparisons use Mann-Whitney U test (p < 0.05 significance threshold):

**Comparison 1: Thermodynamic Loss Effect**
- Groups A vs B (Transformer)
- Groups C vs D (SSM)
- Question: Does L_th improve efficiency in both architectures?

**Comparison 2: Architecture Effect**
- Groups A vs C (both CE)
- Groups B vs D (both L_th)
- Question: Does SSM architecture improve efficiency?

**Comparison 3: Interaction Effect (The Key Test)**
- Groups (A,B,C) vs D
- Question: Is Group D (SSM+L_th) better than the sum of architecture and loss effects alone?
- **This is the PNA-SSM hypothesis**: expect synergy, not just additive improvement

### 5.3 Ablation: Halt Head Design

Test whether the SSM halt head (using state h_t) is better than the Transformer halt head:

**Swap Test:**
1. Take trained Group D (SSM+L_th)
2. Replace halt head with Transformer-style halt head (uses sequence embedding, not state)
3. Measure halt F1

**Expected:** Halt F1 drops, confirming that the state-based halt prediction is superior.

---

## 6. Qualitative Analysis: Reasoning Path Comparison

Manually inspect reasoning traces for the same problem across all groups:

**Example: Parity of 1101**

```
Group A (Trans+CE):
"Input:1101 Let's calculate step by step. First 1 and 1 gives us 0. 
Then 0 and 0 gives us 0. Finally 0 and 1 gives us 1. Result:1<HALT>"
(Wordy, but complete)

Group B (Trans+L_th):
"Input:1101 1^1=0 0^0=0 0^1=1 Result:1<HALT>"
(Concise due to thermodynamic pressure)

Group C (SSM+CE):
"Input:1101 Compute parity: 1^1=0, 0^0=0, 0^1=1 Result:1<HALT>"
(Slightly concise due to SSM's compression)

Group D (SSM+L_th):
"Input:1101 1^1=0 0^1=1 Result:1<HALT>"
(Most concise - skips redundant middle step entirely)
```

**Key observation to look for in Group D:** Does it learn to skip "obvious" intermediate steps 
(like 0^0=0) that don't reduce uncertainty? This would be the "structural sarcasm" mentioned in 
your training notes - the model compressing away redundancy not for token economy, but because 
those steps genuinely don't collapse the probability manifold.

---

## 7. Computational Requirements

**Hardware:** Single GPU (RTX 3060 or better)

**Training time estimate:**
- Group A (Trans+CE): ~2 hours
- Group B (Trans+L_th): ~2.5 hours (extra entropy computation)
- Group C (SSM+CE): ~1.5 hours (SSM faster than attention)
- Group D (SSM+L_th): ~2 hours

**Total for all groups + ablations:** ~15-20 GPU hours

---

## 8. Expected Results & Publication Path

### If Hypothesis Confirmed (Group D >> Groups A,B,C)

**Result 1: Synergistic Efficiency**
- Group D achieves same accuracy with 40-50% fewer reasoning tokens (vs 30% for Group B alone)
- Compound effect: architecture + loss work together, not just additively

**Result 2: Superior Halt Calibration**
- Group D halt F1 > 95% (vs 90% target)
- State-based halt prediction more reliable than token-based

**Result 3: Cleaner Entropy Dynamics**
- State entropy collapse in Group D is sharper than token entropy collapse in any other group
- Evidence that SSM state naturally encodes "progress toward goal"

**Result 4: Better Generalization**
- Group D maintains >90% accuracy on 6-10 bit parity (trained on 2-5)
- Other groups degrade more severely

**Publication Angle:**
"Thermodynamic Loss Functions Achieve Natural Alignment with State Space Model Dynamics"
- NeurIPS: Architecture-Loss co-design
- ICML: Efficiency gains in SSM reasoning
- ICLR: Theoretical connection between SSM compression and thermodynamic principles

### If Hypothesis Partially Confirmed

**Scenario:** Group D better than B, but C also shows some improvement.

**Conclusion:** SSM architecture itself has thermodynamic properties. Thermodynamic loss 
further enhances them, but the architecture is already aligned with efficiency.

**Publication Angle:** 
"State Space Models as Naturally Thermodynamic Architectures"

### If Hypothesis Rejected

**Scenario:** No significant difference between Groups B and D.

**Conclusion:** Thermodynamic loss is architecture-agnostic. The efficiency gains come 
entirely from the loss function, not from architectural alignment.

**Publishable Finding:** The Transformer results (Group B) are still novel and strong.

---

## 9. Implementation Checklist

Week 1:
- [ ] Implement PNA-SSM architecture (based on spec in pna_ssm_architecture.py)
- [ ] Verify parameter count matches Transformer baseline (~7M)
- [ ] Implement state entropy tracker
- [ ] Test forward/backward pass

Week 2:
- [ ] Adapt thermodynamic loss for SSM (same formula, different inputs)
- [ ] Implement state-based halt head
- [ ] Run single-example tests (does Group D generate reasonable output?)

Week 3:
- [ ] Train all 4 groups in parallel
- [ ] Monitor training curves (loss, accuracy, reasoning length)
- [ ] Checkpoint best models from each group

Week 4:
- [ ] Run full evaluation suite
- [ ] Generate entropy collapse visualizations
- [ ] Statistical comparisons
- [ ] Ablation studies

Week 5:
- [ ] Length generalization test (6-10 bit parity)
- [ ] Qualitative analysis (reasoning path inspection)
- [ ] Write-up results

---

## 10. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SSM training unstable | Medium | High | Start with very low α, use gradient clipping |
| State entropy metric uninformative | Medium | Medium | Try alternative formulations (variance, effective rank) |
| No synergy between SSM + L_th | Low | Medium | Architecture effect alone is still publishable |
| Groups C & D both fail | Low | High | Indicates SSM implementation issue - debug before proceeding |

---

## 11. Success Criteria (Hypothesis Confirmation)

The PNA-SSM hypothesis is **confirmed** if ALL of the following hold:

1. **Accuracy**: Group D ≥ 95% (maintains performance)
2. **Efficiency**: Group D reasoning tokens < 0.6 × Group A (better than B's 0.7×)
3. **Halt precision**: Group D F1 > 93% (better than B's 90%)
4. **State collapse**: Group D state entropy shows step-function (visual inspection)
5. **Synergy**: Group D improvement > (Group B improvement + Group C improvement - Group A)
   - i.e., the interaction effect is positive and significant
6. **Generalization**: Group D accuracy on 6-10 bits > Groups A,B,C

If 4+ of these hold, the hypothesis is **partially confirmed** (publishable).
If 6/6 hold, the hypothesis is **strongly confirmed** (top-tier publication).

---

*Experiment Protocol v1.0 - PNA-SSM Hypothesis Test*
