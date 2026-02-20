
---

## Phase 15: Epistemic Controller

### Objective

Build and test a closed-loop inference controller that uses the three-signal framework ($S_c$, $\nabla\mathcal{H}$, $\mathcal{H}$) to monitor reasoning trajectory health and intervene when the model is stuck. Target: break through the 78.8% T2 ceiling at inference time, without retraining.

### Position in the Project

```
Phase 14:  Proved halt veto works (+12.4pp via confusion detection)
Phase 15:  Proves whether PERTURBATION adds value beyond halt veto
           Target: the 21.2% residual error that halt control can't touch
```

### Architecture Constraint

The Phase 12 backbone is **frozen** during inference. We cannot modify hidden states directly (the model reprocesses the full sequence at each step). All interventions are **token-level**:

| Intervention | Mechanism | Invasiveness |
|---|---|---|
| Halt veto | Suppress premature stopping | None (delay only) |
| Expression re-injection | Append original expression to generated text | Low (adds context) |
| Temperature spike | Sample at $T > 1$ for 1-2 steps | Medium (introduces randomness) |
| Logit boosting | Add bias to answer-related token logits | Medium (changes token distribution) |
| Best-of-N selection | Generate N completions, pick best trajectory | None per trajectory (selection only) |

True state surgery (directly modifying $h_t$) requires model architecture changes and is deferred to Phase 16 if needed.

### Prerequisites

- Phase 12 checkpoint (`results/rim_model.pt`)
- Phase 14 confusion head (`results/confusion_head.pt`) — optional, for comparison
- Phase 14 confusion veto as the baseline to beat (T2 = 78.8%)

---

### Experiment 15A: Regime Classification Diagnostic (15 min)

**Question**: What regime is the model in when it produces wrong answers?

```python
class RegimeClassifier:
    """
    Classifies each generation step into one of four regimes
    using validated Phase 14 signals.
    
    Thresholds calibrated from Phase 14A-1:
      True convergence:  Sc=0.025, ∇H=-0.082
      False convergence: Sc=0.333, ∇H=+0.008
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
```

**Evaluate on the full test set:**

```
For each example:
  1. Generate with no halt (thresh=1.0) to get full trajectory
  2. At each step, classify regime
  3. Record regime at the step where halt WOULD have fired (step ~24)
  4. Record final regime (at EOS)
  5. Record correctness

Report:
  Regime at halt point | n | Accuracy | Action implication
  CONVERGING           | ? | ?%       | Allow halt (correct answers)
  ORBITING             | ? | ?%       | Veto halt + perturb (stuck in loop)
  DIFFUSING            | ? | ?%       | Veto halt + retrieve/concentrate
  PROGRESSING          | ? | ?%       | Veto halt + wait (still computing)
```

**Decision logic from 15A:**

- If >50% of incorrect examples are ORBITING at halt → perturbation has high potential
- If most incorrect examples are CONVERGING → model is converging to wrong answers → need 14F
- If most are PROGRESSING → model just needs more time → halt veto is sufficient
- If most are DIFFUSING → model is lost → may need external information (future work)

---

### Experiment 15B: Halt Veto Baseline (15 min)

Reproduce Phase 14's confusion veto result using the EpistemicController's regime classifier instead of the Confusion Head. This validates the controller integration and provides the baseline for perturbation experiments.

```python
class EpistemicController:
    def __init__(self, window=8, max_interventions=3):
        self.classifier = RegimeClassifier()
        self.window = window
        self.max_interventions = max_interventions
        # ... state tracking from earlier design
    
    def step(self, h_t, halt_conf):
        signals = self._compute_signals(h_t)
        regime = self.classifier.classify(
            signals['Sc'], signals['grad_H'], signals['H']
        )
        
        action = 'CONTINUE'
        
        # Halt only when CONVERGING
        if halt_conf > 0.95:
            if regime == 'CONVERGING':
                action = 'HALT'
            else:
                action = 'CONTINUE'  # Veto
        
        return {'action': action, 'regime': regime, 'signals': signals}
```

**Expected result**: T2 ≈ 78.8%, matching Phase 14D. If significantly different, the regime classifier's thresholds need recalibration.

---

### Experiment 15C: Perturbation Ablation (1 hour)

The core new contribution. Five perturbation strategies, each tested with halt veto active.

#### 15C-1: Expression Re-injection

When ORBITING for `patience` steps, re-append the expression.

```python
if regime == 'ORBITING' and orbit_count >= patience:
    reminder_tokens = tokenize(f" Input:{expression} Result:")
    generated_ids.extend(reminder_tokens)
    orbit_count = 0
    n_interventions += 1
```

**Rationale**: The model may have lost track of the original problem during cycling. Re-injecting the expression acts as an attention anchor, pulling the model back to the relevant context.

**Risk level**: Low. Tested in Phase 14D (neutral to slightly negative). But Phase 14D didn't have the halt veto active.

#### 15C-2: Temperature Spike

When ORBITING, sample with high temperature for 2 steps to escape the local mode.

```python
if regime == 'ORBITING' and orbit_count >= patience:
    # Next 2 tokens: sample at T=2.0 instead of greedy
    probs = F.softmax(logits / 2.0, dim=-1)
    next_token = torch.multinomial(probs, 1).item()
    temp_steps_remaining = 2
```

**Rationale**: Greedy decoding follows the highest-likelihood path, which may be a cycling path. High temperature introduces randomness that can escape the orbit.

**Risk level**: Medium. Random tokens may corrupt context. Use `temp_steps_remaining` to limit exposure.

#### 15C-3: Separator Injection

When ORBITING, insert a separator token sequence to signal a fresh computation phase.

```python
if regime == 'ORBITING' and orbit_count >= patience:
    separator = tokenize(" Check: ")  # Or " = " or " Result:"
    generated_ids.extend(separator)
```

**Rationale**: The model may have learned during training that `Result:` follows reasoning. Injecting this marker early may cause the model to transition from reasoning mode to answer-emission mode.

**Risk level**: Low. Small token injection.

#### 15C-4: Logit Boosting

When ORBITING, bias the output logits toward answer-related tokens.

```python
if regime == 'ORBITING' and orbit_count >= patience:
    # Boost logits for digits and 'Result:' token
    digit_ids = [VOCAB[str(d)] for d in range(10) if str(d) in VOCAB]
    result_id = VOCAB.get('Result:', -1)
    for did in digit_ids:
        if did >= 0: logits[0, did] += 3.0
    if result_id >= 0:
        logits[0, result_id] += 5.0
    boost_steps_remaining = 3
```

**Rationale**: The model's logits during cycling may have the answer tokens as second- or third-highest probability. Boosting pushes them over the threshold.

**Risk level**: Medium. May produce grammatically incorrect output. Limit to 3 steps.

#### 15C-5: Best-of-N Selection

Generate N completions. Use the controller's regime signals to select the best one.

```python
def best_of_n_generate(model, controller, expression, N=5):
    candidates = []
    for _ in range(N):
        # Generate with temperature=0.8 for diversity
        result = controlled_generate(model, controller, expression, 
                                     temperature=0.8)
        # Score: prefer low Sc, negative grad_H, low H at final step
        signals = result['controller_log']['signals'][-1]
        score = -signals['Sc'] + signals['grad_H'] - 0.1 * signals['H']
        candidates.append((score, result))
    
    # Return highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
```

**Rationale**: Different random seeds produce different trajectories. Some may escape the orbit that greedy decoding gets stuck in. The controller selects the trajectory with the best convergence signature.

**Risk level**: Low (no model modification). Cost: N× inference time.

#### Ablation Matrix

For each strategy, sweep:

| Parameter | Values |
|-----------|--------|
| `patience` (orbiting steps before intervening) | 3, 5, 8 |
| `max_interventions` per generation | 1, 2, 3 |
| `kick_strength` / `temperature` / `boost` | strategy-specific |

**Report format:**

```
Strategy        | patience | max_int | T2 Acc | Overall | Tokens | Δ vs 15B
----------------|----------|---------|--------|---------|--------|----------
15B (veto only) |    —     |    —    | 78.8%  | 73.6%   | 36.6   |   —
15C-1 re-inject |    3     |    2    | ??.?%  | ??.?%   | ??.?   | +?.?pp
15C-2 temp spike|    5     |    3    | ??.?%  | ??.?%   | ??.?   | +?.?pp
15C-3 separator |    3     |    2    | ??.?%  | ??.?%   | ??.?   | +?.?pp
15C-4 logit     |    5     |    3    | ??.?%  | ??.?%   | ??.?   | +?.?pp
15C-5 best-of-5 |    —     |    —    | ??.?%  | ??.?%   | 5×base | +?.?pp
```

---

### Experiment 15D: Full Controller (30 min)

Combine the best perturbation strategy from 15C with the halt veto. Use regime-adaptive behavior:

```python
def step(self, h_t, halt_conf):
    signals = self._compute_signals(h_t)
    regime = self.classifier.classify(...)
    
    if halt_conf > 0.95:
        if regime == 'CONVERGING':
            return {'action': 'HALT'}
        else:
            return {'action': 'CONTINUE'}  # Veto
    
    if regime == 'ORBITING' and self.orbit_count >= self.patience:
        return {'action': 'PERTURB', 
                'strategy': best_strategy_from_15C}
    
    if regime == 'DIFFUSING' and self.diffuse_count >= self.diffuse_patience:
        return {'action': 'PERTURB', 
                'strategy': 'concentrate'}  # Or best-of-N restart
    
    return {'action': 'CONTINUE'}
```

**Comparison matrix:**

```
Condition              |  T2 Acc | Overall | Mechanism
-----------------------|---------|---------|--------------------
Phase 12 baseline      |  66.4%  |  51.7%  | No control
Phase 14D halt veto    |  78.8%  |  73.6%  | Veto only
Phase 15D full control |  ??.?%  |  ??.?%  | Veto + perturbation
```

---

### Experiment 15E: Failure Analysis (15 min)

For examples still incorrect after full controller, classify the failure mode:

```python
for ex in incorrect_examples:
    log = ex['controller_log']
    final_regime = log['regimes'][-1]
    n_interventions = log['n_interventions']
    final_Sc = log['signals'][-1]['Sc']
    
    if final_regime == 'CONVERGING' and final_Sc < 0.1:
        failure_mode = 'WRONG_BASIN'        # Converged to wrong answer
    elif final_regime == 'ORBITING':
        failure_mode = 'PERSISTENT_ORBIT'   # Perturbation didn't break cycle
    elif n_interventions == max_interventions:
        failure_mode = 'EXHAUSTED_BUDGET'   # Ran out of interventions
    else:
        failure_mode = 'COMPUTATION_ERROR'  # Wrong intermediate steps
```

**Report:**

```
Failure Mode        |  n  | % of errors | Addressable by
--------------------|-----|-------------|------------------
WRONG_BASIN         |  ?  |    ?%       | 14F (basin deepening)
PERSISTENT_ORBIT    |  ?  |    ?%       | Stronger perturbation / more N
EXHAUSTED_BUDGET    |  ?  |    ?%       | Higher max_interventions
COMPUTATION_ERROR   |  ?  |    ?%       | More training / larger model
```

This tells you exactly where the remaining errors live and which intervention (14F, more perturbation, more training) would address each one.

---

### Success Criteria

| Criterion | Target | Phase 14 Baseline |
|-----------|--------|-------------------|
| T2 accuracy (full controller) | >80% | 78.8% (veto only) |
| Overall accuracy | >75% | 73.6% |
| At least one perturbation > veto alone | YES | — |
| Token overhead | <2× (or <5× for best-of-N) | 1.55× |
| Regime classification correlates with accuracy | $r > 0.3$ | — |
| Failure analysis identifies addressable errors | >30% of remaining errors | — |

---

### Ordering and Dependencies

```
         ┌─────────────────────┐
         │   Phase 12 Backbone │
         └──────────┬──────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐     ┌─────────────────┐
│  Phase 14F    │     │    Phase 15     │
│  Basin Deep   │     │  Epistemic Ctrl │
│  (training)   │     │  (inference)    │
│  ~2 hours     │     │  ~1.5 hours     │
└───────┬───────┘     └────────┬────────┘
        │                      │
        └──────────┬───────────┘
                   ▼
        ┌──────────────────────┐
        │  Phase 15 + 14F      │
        │  Controller on       │
        │  deepened backbone   │
        │  (30 min)            │
        └──────────────────────┘
```

