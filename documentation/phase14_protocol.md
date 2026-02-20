

# Phase 14: Halt Control on a Geometrically Sound Manifold

## Revised Protocol

---

## Context: What We Now Know

| Finding | Value | Implication |
|---------|-------|-------------|
| Tier 2 Geodesic Purity | **83.0%** | The geometric structure is real — states respect algebraic rules |
| False Convergence Rate | **96.1%** | The Halt Head fires during limit cycles, not at fixed points |
| True Conv. Accuracy | **89.7%** | When the model reaches a real fixed point, it's almost always right |
| True Conv. Rate | **3.9%** | But it almost never gets there |
| Phase 13b Purity | 18.0% | Invasive SSM modification destroyed the geometry |

**The manifold is sound. The halting is broken.**

Phase 14 is no longer about fixing the geometry (Phase 12 already did that). It's about **halting control**: detecting when the Halt Head is about to fire prematurely, suppressing it, and giving the trajectory time to reach a true fixed point.

---

## Foundation

All experiments build on the **Phase 12 checkpoint** (`results/rim_model.pt`). Phase 13b is archived. No SSM recurrence modification.

```bash
# All Phase 14 experiments use:
MODEL_CLASS = PNA_SSM_RIM
CHECKPOINT = results/rim_model.pt
BACKBONE_FROZEN = True  # For Confusion Head training only
```

---

## Experimental Sequence

```
14A: Diagnostics (no training)                    ~45 min
 ├── 14A-0: Extended generation budget
 ├── 14A-1: Three-signal validation
 └── 14A-2: Op Detector probe on Phase 12 backbone

14B: Confusion Head (frozen backbone)             ~2 hours
 ├── 14B-1: Non-parametric cycle detector
 └── 14B-2: Learned Confusion Head

14C: Halt Gating (inference-time only)            ~1 hour
 └── Confusion-modulated halt threshold

14D: Cycle Breaking (if 14C insufficient)         ~2 hours
 └── State perturbation when patience exhausted

14E: Integration + Figures                        ~1 hour
 └── Full pipeline, paper-ready evaluation
```

---

## 14A: Diagnostics

### 14A-0: Extended Generation Budget

The simplest possible test. No architecture changes. No training. Just raise the halt threshold and see what happens.

```python
def experiment_14a0_extended_budget(model, test_ds, device):
    """
    Does giving the model more time improve accuracy?

    Three conditions:
      1. Default halt threshold (0.5) — baseline
      2. Raised threshold (0.8) — model must be more confident to halt
      3. Raised threshold (0.95) — model almost never halts voluntarily
      4. No halt at all — generate max_length tokens, take final answer

    If accuracy improves with raised thresholds:
      → False convergence is partly a confidence calibration problem
      → Confusion Head just needs to raise the bar

    If accuracy doesn't improve (or gets worse):
      → Model is stuck in limit cycles regardless of halt timing
      → Need active cycle-breaking intervention (14D)
    """
    thresholds = [0.5, 0.8, 0.95, None]  # None = ignore halt
    results = {}

    for thresh in thresholds:
        generator = CompressibleGenerator(
            model, device=device,
            halt_threshold=thresh,
            max_length=100,  # Extended from default
        )

        gen_results = []
        for i in range(len(test_ds.examples)):
            ex = test_ds.examples[i]
            gen = generator.generate(ex['expression'])

            is_correct = (gen['parsed_answer'] == ex['answer'])
            convergence = classify_convergence(
                gen, detect_oscillation(gen['state_vectors'])
            )

            gen_results.append({
                'tier': ex['tier'],
                'is_correct': is_correct,
                'convergence': convergence,
                'reasoning_tokens': gen['reasoning_tokens'],
            })

        # Per-tier accuracy
        for tier in [0, 1, 2]:
            tier_results = [r for r in gen_results if r['tier'] == tier]
            acc = np.mean([r['is_correct'] for r in tier_results])
            tokens = np.mean([r['reasoning_tokens'] for r in tier_results])
            true_conv = sum(
                1 for r in tier_results
                if r['convergence'] == 'true_convergence'
            )
            results[(thresh, tier)] = {
                'accuracy': acc,
                'mean_tokens': tokens,
                'true_convergence': true_conv,
                'n': len(tier_results),
            }

        label = f"thresh={thresh}" if thresh else "no_halt"
        t2 = results.get((thresh, 2), {})
        print(f"  {label:15s} | T2 acc={t2.get('accuracy',0):.1%} "
              f"tokens={t2.get('mean_tokens',0):.1f} "
              f"true_conv={t2.get('true_convergence',0)}")

    return results
```

**Expected output format:**

```
  thresh=0.5      | T2 acc=66.4%  tokens=16.7  true_conv=35    ← baseline
  thresh=0.8      | T2 acc=??     tokens=??    true_conv=??
  thresh=0.95     | T2 acc=??     tokens=??    true_conv=??
  thresh=None     | T2 acc=??     tokens=??    true_conv=??
```

**Decision logic:**

```
If T2 acc at thresh=0.8 > T2 acc at thresh=0.5 + 5pp:
    → Halt calibration is the dominant problem
    → 14C (threshold gating) is likely sufficient
    → Skip 14D (perturbation)

If T2 acc is flat or decreasing across thresholds:
    → More time doesn't help → model is stuck in cycles
    → 14D (cycle breaking) is essential
    → 14C alone won't be enough

If true_conv increases with higher thresholds:
    → Some trajectories DO converge if given time
    → Quantifies the ceiling for halt-only intervention
```

**Cost**: ~25 minutes (4× generation, no training)

---

### 14A-1: Three-Signal Validation

Validate the three-signal certainty coordinate from Appendix A against the true/false convergence boundary.

```python
def experiment_14a1_three_signal(model, test_ds, device):
    """
    Compute H, ∇H, Sc at each generation step.
    Test whether the three-signal framework discriminates
    true convergence from false convergence.
    """
    generator = CompressibleGenerator(model, device=device)

    trajectories = []
    for i in tqdm(range(len(test_ds.examples)), desc="14A-1"):
        ex = test_ds.examples[i]
        gen = generator.generate(ex['expression'])

        states = gen['state_vectors']
        entropies = gen['state_entropies']
        osc = detect_oscillation(states)
        convergence = classify_convergence(gen, osc)
        is_correct = (gen['parsed_answer'] == ex['answer'])

        # Compute per-step signals
        per_step = []
        for t in range(len(states)):
            H = entropies[t] if t < len(entropies) else 0.0
            grad_H = compute_entropy_gradient(entropies, t, window=3)
            Sc = compute_cycling_score(states, t, k_values=[2, 4, 8])

            per_step.append({
                'H': H,
                'grad_H': grad_H,
                'Sc': Sc,
                'halt_conf': (gen['halt_confidences'][t]
                              if t < len(gen['halt_confidences'])
                              else 0.0),
            })

        # Summary features at halt point
        T = len(per_step) - 1
        if T >= 0:
            trajectories.append({
                'tier': ex['tier'],
                'convergence': convergence,
                'is_correct': is_correct,
                'n_steps': len(states),
                'H_at_halt': per_step[T]['H'],
                'grad_H_at_halt': per_step[T]['grad_H'],
                'Sc_at_halt': per_step[T]['Sc'],
                'halt_conf_at_halt': per_step[T]['halt_conf'],
                # Window features (last 5 steps)
                'mean_Sc_last5': np.mean(
                    [s['Sc'] for s in per_step[max(0,T-4):T+1]]
                ),
                'mean_gradH_last5': np.mean(
                    [s['grad_H'] for s in per_step[max(0,T-4):T+1]]
                ),
                'max_Sc': max(s['Sc'] for s in per_step),
                'per_step': per_step,
            })

    # Discrimination analysis
    true_conv = [t for t in trajectories
                 if t['convergence'] == 'true_convergence']
    false_conv = [t for t in trajectories
                  if t['convergence'] == 'false_convergence']

    print(f"\n  True Convergence (n={len(true_conv)}):")
    print(f"    Sc at halt:    {np.mean([t['Sc_at_halt'] for t in true_conv]):.3f}")
    print(f"    ∇H at halt:    {np.mean([t['grad_H_at_halt'] for t in true_conv]):.3f}")
    print(f"    H at halt:     {np.mean([t['H_at_halt'] for t in true_conv]):.3f}")

    print(f"\n  False Convergence (n={len(false_conv)}):")
    print(f"    Sc at halt:    {np.mean([t['Sc_at_halt'] for t in false_conv]):.3f}")
    print(f"    ∇H at halt:    {np.mean([t['grad_H_at_halt'] for t in false_conv]):.3f}")
    print(f"    H at halt:     {np.mean([t['H_at_halt'] for t in false_conv]):.3f}")

    # ROC analysis: can these features predict convergence type?
    from sklearn.metrics import roc_auc_score
    labels = [1 if t['convergence'] == 'true_convergence' else 0
              for t in trajectories]

    for feature in ['Sc_at_halt', 'grad_H_at_halt', 'mean_Sc_last5']:
        values = [t[feature] for t in trajectories]
        try:
            auc = roc_auc_score(labels, [-v for v in values])
            # Negate because true convergence should have LOWER Sc
            print(f"    AUC ({feature}): {auc:.3f}")
        except ValueError:
            print(f"    AUC ({feature}): undefined (single class)")

    return trajectories


def compute_entropy_gradient(entropies, t, window=3):
    """Smoothed entropy gradient. Negative = converging."""
    if t < 1:
        return 0.0
    start = max(0, t - window)
    if start == t:
        return 0.0
    return (entropies[t] - entropies[start]) / (t - start)


def compute_cycling_score(state_vectors, t, k_values=[2, 4, 8]):
    """
    Detect periodic orbits in state space.
    Sc(t,k) = 1 - ||h_t - h_{t-k}|| / (||h_t|| + ||h_{t-k}|| + ε)
    High Sc → state has returned to a previous position.
    """
    if t < min(k_values):
        return 0.0

    scores = []
    h_t = state_vectors[t]
    for k in k_values:
        if t - k >= 0:
            h_prev = state_vectors[t - k]
            dist = (h_t - h_prev).norm().item()
            norm_sum = h_t.norm().item() + h_prev.norm().item() + 1e-9
            scores.append(1.0 - dist / norm_sum)

    return max(scores) if scores else 0.0
```

**Success criteria:**

| Signal | True Conv. vs False Conv. | Target AUC |
|--------|--------------------------|------------|
| $S_c$ at halt | Lower for true conv. (fixed point, not cycling) | > 0.70 |
| $\nabla\mathcal{H}$ at halt | More negative for true conv. (still decreasing) | > 0.65 |
| Combined | Three signals together | > 0.80 |

**Cost**: ~15 minutes (one generation pass + analysis)

---

### 14A-2: Op Detector Probe on Phase 12 Backbone

Confirm the Phase 12 states carry operation-type information without retraining the backbone.

```python
def experiment_14a2_op_probe(model, train_ds, test_ds, device, epochs=10):
    """
    Train a lightweight Op Detector on Phase 12's FROZEN backbone.
    Tests: "Do Phase 12 states encode operation types?"
    """
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    # Build Op Detector (same as Phase 13b but trained read-only)
    op_detector = nn.Sequential(
        nn.Linear(512, 128),  # d_model = 512
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 6),   # 6 op types
    ).to(device)

    optimizer = torch.optim.Adam(op_detector.parameters(), lr=1e-3)
    rim_train = RIMDatasetWrapper(train_ds)
    loader = DataLoader(rim_train, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        total_correct = 0
        total_samples = 0
        total_loss = 0

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            op_types = batch['op_types'].to(device)
            op_before_pos = batch['op_before_pos'].to(device)
            n_ops = batch['n_ops'].to(device)

            with torch.no_grad():
                outputs = model(input_ids)
                # Extract hidden representations (pre-head)
                # Use the embedding + layer output, not states
                h = model.embedding(input_ids) + model.pos_encoding(
                    torch.arange(input_ids.size(1), device=device).unsqueeze(0)
                )
                for layer in model.layers:
                    h, _ = layer(h)
                h = model.norm(h)

            # Op detection on hidden states
            op_logits = op_detector(h)

            # Build per-token op labels (same as Phase 13b)
            B, L = input_ids.shape
            per_token_labels = torch.zeros(B, L, dtype=torch.long, device=device)
            # Default: OP_REAL for non-pad tokens
            pad_mask = (input_ids != VOCAB['<PAD>'])
            per_token_labels[pad_mask] = OP_REAL

            for b_idx in range(B):
                n = n_ops[b_idx].item()
                for op_i in range(n):
                    ot = op_types[b_idx, op_i].item()
                    bp = op_before_pos[b_idx, op_i].item()
                    if bp < L:
                        per_token_labels[b_idx, bp] = ot

            loss = F.cross_entropy(
                op_logits.reshape(-1, 6),
                per_token_labels.reshape(-1),
                ignore_index=OP_PAD,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = op_logits.argmax(dim=-1)
            mask = per_token_labels != OP_PAD
            total_correct += (preds[mask] == per_token_labels[mask]).sum().item()
            total_samples += mask.sum().item()
            total_loss += loss.item()

        acc = total_correct / max(total_samples, 1)
        print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f} acc={acc:.1%}")

    # Unfreeze backbone after probe
    for p in model.parameters():
        p.requires_grad = True

    return op_detector
```

**Expected result**: ~74% accuracy (matching Phase 13b, confirming states are equally informative without gating).

**Cost**: ~10 minutes

---

## 14B: Confusion Head

### 14B-1: Non-Parametric Cycle Detector (Baseline)

Before training anything, test whether a simple statistical rule detects false convergence:

```python
class CycleDetector:
    """
    Non-parametric false convergence detector.
    Uses state delta statistics to distinguish
    fixed points from limit cycles.

    No training required — purely statistical.
    """

    def __init__(self, window=5, convergence_threshold=0.05,
                 cycling_cv_threshold=0.3):
        self.window = window
        self.conv_thresh = convergence_threshold
        self.cycling_cv_thresh = cycling_cv_threshold

    def detect(self, state_vectors):
        """
        Returns:
          'converged':  deltas are all small (fixed point)
          'cycling':    deltas are consistent but non-zero (limit cycle)
          'diverging':  deltas are growing (unstable)
          'uncertain':  not enough data
        """
        if len(state_vectors) < self.window + 1:
            return 'uncertain', 0.0

        recent = torch.stack(state_vectors[-(self.window + 1):])
        deltas = (recent[1:] - recent[:-1]).norm(dim=-1)  # [window]

        mean_delta = deltas.mean().item()
        max_delta = deltas.max().item()

        # Fixed point: all deltas are tiny
        if max_delta < self.conv_thresh:
            return 'converged', 0.0  # Confidence: 0 = no confusion

        # Limit cycle: consistent non-zero deltas
        cv = deltas.std().item() / (mean_delta + 1e-9)
        if cv < self.cycling_cv_thresh:
            # Low coefficient of variation → regular oscillation
            confusion_score = 1.0 - cv / self.cycling_cv_thresh
            return 'cycling', confusion_score

        # Diverging: growing deltas
        if deltas[-1] > deltas[0] * 1.5:
            return 'diverging', 0.8

        return 'uncertain', 0.3

    def should_suppress_halt(self, state_vectors, halt_confidence):
        """
        Should we suppress the halt at this step?
        Returns True if the state is cycling AND halt wants to fire.
        """
        status, confusion = self.detect(state_vectors)
        if status == 'cycling' and halt_confidence > 0.5:
            return True, confusion
        return False, confusion
```

**Evaluation**: Run generation with CycleDetector alongside, measure:

```python
def evaluate_cycle_detector(model, test_ds, device):
    """
    How well does the CycleDetector predict true/false convergence?
    No intervention — pure detection accuracy.
    """
    generator = CompressibleGenerator(model, device=device)
    detector = CycleDetector()

    tp, fp, tn, fn = 0, 0, 0, 0

    for i in range(len(test_ds.examples)):
        ex = test_ds.examples[i]
        gen = generator.generate(ex['expression'])
        osc = detect_oscillation(gen['state_vectors'])
        truth = classify_convergence(gen, osc)

        status, _ = detector.detect(gen['state_vectors'])

        # Positive = cycling (should suppress halt)
        # True condition = false convergence
        predicted_cycling = (status == 'cycling')
        actual_false_conv = (truth == 'false_convergence')

        if predicted_cycling and actual_false_conv:
            tp += 1
        elif predicted_cycling and not actual_false_conv:
            fp += 1
        elif not predicted_cycling and actual_false_conv:
            fn += 1
        else:
            tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    print(f"  CycleDetector: P={precision:.3f} R={recall:.3f} F1={f1:.3f}")
    print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"    False alarm rate: {fp}/{fp+tn} = {fp/max(fp+tn,1):.1%}")

    return {'precision': precision, 'recall': recall, 'f1': f1}
```

**Cost**: ~10 minutes

---

### 14B-2: Learned Confusion Head

If the non-parametric detector achieves F1 > 0.6, skip this — the simple version is sufficient. Otherwise, train a learned model:

```python
class ConfusionHead(nn.Module):
    """
    Detects false convergence (limit cycles) from trajectory features.

    Trained on Phase 12 backbone (FROZEN).
    Input: window of state deltas + entropy features + halt confidences.
    Output: confusion probability (high = likely false convergence).

    Key design: operates on DIFFERENCES between consecutive states,
    not raw states. This makes it invariant to the state's absolute
    position on the manifold and sensitive only to dynamics.
    """

    def __init__(self, d_state, window=5, d_hidden=32):
        super().__init__()
        self.window = window
        # Input per step: d_state (delta) + 3 (H, ∇H, Sc) + 1 (halt_conf)
        input_per_step = d_state + 4
        self.net = nn.Sequential(
            nn.Linear(input_per_step * window, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, features):
        """
        features: [B, window, d_state + 4]
        Returns: confusion logits [B, 1]
        """
        B = features.size(0)
        return self.net(features.reshape(B, -1))


def collect_confusion_labels(model, train_ds, device, n_samples=2000):
    """
    Generate trajectories and label each step with confusion signal.

    Label = 1 at step t if:
      (a) Trajectory is false convergence, AND
      (b) Step t is within 5 steps of the halt point
          (the region where the Confusion Head would need to fire)

    Label = 0 if:
      (a) True convergence, OR
      (b) Step t is far from the halt point
    """
    generator = CompressibleGenerator(model, device=device)
    windows = []

    for i in tqdm(range(min(n_samples, len(train_ds.examples))),
                  desc="Collecting confusion labels"):
        ex = train_ds.examples[i]
        gen = generator.generate(ex['expression'])

        states = gen['state_vectors']
        entropies = gen['state_entropies']
        halt_confs = gen['halt_confidences']

        osc = detect_oscillation(states)
        convergence = classify_convergence(gen, osc)
        is_false_conv = (convergence == 'false_convergence')

        if len(states) < 6:
            continue

        # Extract windows
        halt_step = len(states) - 1
        window = 5

        for t in range(window, len(states)):
            # Build feature vector for this window
            step_features = []
            for w in range(t - window, t):
                if w == 0:
                    delta = torch.zeros_like(states[0])
                else:
                    delta = states[w] - states[w - 1]

                H = entropies[w] if w < len(entropies) else 0.0
                grad_H = compute_entropy_gradient(entropies, w)
                Sc = compute_cycling_score(states, w)
                hc = halt_confs[w] if w < len(halt_confs) else 0.0

                feat = torch.cat([
                    delta.cpu(),
                    torch.tensor([H, grad_H, Sc, hc]),
                ])
                step_features.append(feat)

            features = torch.stack(step_features)  # [window, d_state+4]

            # Label: confusion if false convergence AND near halt
            near_halt = (halt_step - t) < 5
            label = 1.0 if (is_false_conv and near_halt) else 0.0

            windows.append({
                'features': features,
                'label': label,
                'tier': ex['tier'],
            })

    return windows


def train_confusion_head(d_state, training_windows, device,
                         epochs=20, lr=1e-3):
    """Train Confusion Head on retrospective trajectory labels."""
    confusion_head = ConfusionHead(d_state=d_state).to(device)

    n_pos = sum(1 for w in training_windows if w['label'] > 0.5)
    n_neg = len(training_windows) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    optimizer = torch.optim.Adam(confusion_head.parameters(), lr=lr)

    # Simple dataset
    features = torch.stack([w['features'] for w in training_windows])
    labels = torch.tensor([w['label'] for w in training_windows])

    dataset = torch.utils.data.TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for feat_batch, label_batch in loader:
            feat_batch = feat_batch.to(device)
            label_batch = label_batch.to(device).unsqueeze(-1)

            logits = confusion_head(feat_batch)
            loss = F.binary_cross_entropy_with_logits(
                logits, label_batch, pos_weight=pos_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = (logits > 0).float()
            correct += (preds == label_batch).sum().item()
            total += label_batch.numel()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/len(loader):.4f} "
                  f"acc={correct/total:.1%}")

    return confusion_head
```

**Success criteria:**

| Metric | Target | Rationale |
|--------|--------|-----------|
| Confusion detection F1 | > 0.70 | Must reliably identify limit cycles |
| False alarm rate (confusion on true convergence) | < 10% | Must not interrupt genuine fixed points |
| Detection lead time (fires ≥ N steps before halt) | ≥ 2 steps | Time to intervene |

---

## 14C: Halt Gating

The core intervention. When confusion is detected, **raise the halt threshold** to give the trajectory more time.

```python
class ConfusionGatedGenerator(CompressibleGenerator):
    """
    Generation with confusion-modulated halt threshold.

    Normal:    halt when halt_conf > 0.5
    Confused:  halt when halt_conf > 0.5 + confusion * boost

    No SSM modification. No retry. Just patience.
    """

    def __init__(self, model, confusion_detector, device,
                 base_threshold=0.5, max_boost=0.4,
                 extended_max_length=100):
        super().__init__(model, device=device)
        self.confusion_detector = confusion_detector
        self.base_threshold = base_threshold
        self.max_boost = max_boost
        self.extended_max_length = extended_max_length

    def generate(self, expression):
        """
        Modified generation loop with confusion-gated halting.
        """
        prompt_text = f"Input: {expression} Target:"
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        state_vectors = []
        halt_confidences = []
        state_entropies = []
        confusion_scores = []

        for step in range(self.extended_max_length):
            input_tensor = torch.tensor(
                [generated_ids], dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                outputs = self.model(input_tensor)

            # Capture state
            last_state = outputs['states_sequence'][:, -1, :]
            state_vectors.append(last_state.squeeze(0).cpu())

            halt_conf = outputs['halt_confidence'][:, -1, 0].item()
            halt_confidences.append(halt_conf)

            entropy = compute_state_entropy(last_state.squeeze(0))
            state_entropies.append(entropy)

            # Compute confusion
            if isinstance(self.confusion_detector, CycleDetector):
                _, confusion = self.confusion_detector.detect(state_vectors)
            else:
                # Learned ConfusionHead — build feature window
                confusion = self._compute_learned_confusion(
                    state_vectors, state_entropies,
                    halt_confidences
                )
            confusion_scores.append(confusion)

            # Dynamic halt threshold
            effective_threshold = (
                self.base_threshold + confusion * self.max_boost
            )

            # Check halt
            if halt_conf > effective_threshold:
                stop_reason = 'halt_confidence'
                break

            # Get next token
            logits = outputs['logits'][:, -1, :]
            next_token = logits.argmax(dim=-1).item()

            if next_token == VOCAB.get('<HALT>', -1):
                stop_reason = 'halt_token'
                break

            generated_ids.append(next_token)
        else:
            stop_reason = 'max_length'

        # Parse answer from generated tokens
        generated_text = detokenize(generated_ids[len(prompt_ids):])
        parsed_answer = self._parse_answer(generated_text)

        return {
            'parsed_answer': parsed_answer,
            'generated_text': generated_text,
            'generated_ids': generated_ids,
            'reasoning_tokens': len(generated_ids) - len(prompt_ids),
            'total_tokens': len(generated_ids),
            'stop_reason': stop_reason,
            'state_vectors': state_vectors,
            'halt_confidences': halt_confidences,
            'state_entropies': state_entropies,
            'confusion_scores': confusion_scores,
        }

    def _compute_learned_confusion(self, state_vectors,
                                    entropies, halt_confs):
        """Build feature window and run ConfusionHead."""
        window = 5
        t = len(state_vectors) - 1
        if t < window:
            return 0.0

        step_features = []
        for w in range(t - window, t):
            delta = (state_vectors[w] - state_vectors[w - 1]
                     if w > 0 else torch.zeros_like(state_vectors[0]))
            H = entropies[w] if w < len(entropies) else 0.0
            grad_H = compute_entropy_gradient(entropies, w)
            Sc = compute_cycling_score(state_vectors, w)
            hc = halt_confs[w] if w < len(halt_confs) else 0.0

            feat = torch.cat([
                delta, torch.tensor([H, grad_H, Sc, hc])
            ])
            step_features.append(feat)

        features = torch.stack(step_features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.confusion_detector(features)
            return torch.sigmoid(logit).item()
```

### Evaluation: Ablation Matrix

| Condition | Confusion Detector | Halt Boost | Tests |
|-----------|:--:|:--:|---|
| A. Baseline | None | 0 | Phase 12 as-is |
| B. Fixed raised threshold | None | +0.3 (constant) | Does patience alone help? |
| C. CycleDetector | Non-parametric | Dynamic 0–0.4 | Does detection help vs constant boost? |
| D. ConfusionHead | Learned | Dynamic 0–0.4 | Does learning help vs hand-crafted? |

**The critical comparison is B vs C.** If the CycleDetector (no training) matches the learned ConfusionHead, the simple version wins. If constant boost (no detection) matches detection, then the problem is purely patience, not detection.

### Success Criteria

| Metric | Target | Meaning |
|--------|--------|---------|
| T2 Accuracy (best condition) | > 72% (+6pp over baseline) | Halt gating improves outcomes |
| True Convergence Rate | > 10% (from 3.9%) | More trajectories reach fixed points |
| Token overhead | < 2× baseline | Efficiency is acceptable |
| Best condition ≠ B (constant boost) | Significant difference | Detection adds value beyond patience |

---

## 14D: Cycle Breaking

**Only proceed here if 14C shows: extended generation budget alone doesn't help (from 14A-0) AND the CycleDetector fires but accuracy doesn't improve (from 14C).**

This means the model is genuinely stuck in limit cycles that it can't escape even with more time. Active intervention is needed.

```python
class CycleBreaker:
    """
    When the model has been confused for K consecutive steps,
    apply a state perturbation to break the limit cycle.

    Three strategies (ablate all):
      1. Random noise: push state in random direction
      2. Anti-cycling: push state AWAY from the previous state
         (oppose the limit cycle direction)
      3. Entropy reduction: push toward lower-entropy state
         (encourage convergence)
    """

    def __init__(self, strategy='anti_cycling', strength=0.1,
                 patience=5):
        self.strategy = strategy
        self.strength = strength
        self.patience = patience

    def should_perturb(self, confusion_history):
        """
        Fire after `patience` consecutive confused steps.
        """
        if len(confusion_history) < self.patience:
            return False
        return all(c > 0.5 for c in confusion_history[-self.patience:])

    def compute_perturbation(self, state_vectors):
        """
        Compute perturbation direction based on strategy.
        Returns: perturbation vector (same shape as state).
        """
        h = state_vectors[-1]

        if self.strategy == 'random':
            noise = torch.randn_like(h)
            return self.strength * h.norm() * noise / (noise.norm() + 1e-9)

        elif self.strategy == 'anti_cycling':
            if len(state_vectors) < 2:
                return torch.zeros_like(h)
            # Push AWAY from the direction of the cycle
            cycle_dir = h - state_vectors[-2]
            # Perpendicular perturbation (break the cycle orbit)
            random_component = torch.randn_like(h)
            # Project out the cycle direction
            cycle_norm = cycle_dir / (cycle_dir.norm() + 1e-9)
            perp = (random_component
                    - (random_component @ cycle_norm) * cycle_norm)
            return self.strength * h.norm() * perp / (perp.norm() + 1e-9)

        elif self.strategy == 'entropy_reduction':
            # Push toward lower entropy (more peaked distribution)
            energy = h ** 2
            probs = energy / (energy.sum() + 1e-9)
            # Gradient of entropy w.r.t. h (toward lower entropy)
            # ∂H/∂h = -2h(1 + log(h²/Z))/Z where Z = sum(h²)
            log_probs = torch.log(probs + 1e-9)
            grad = -2 * h * (1 + log_probs) / (energy.sum() + 1e-9)
            return self.strength * grad / (grad.norm() + 1e-9)
```

**Note on implementation**: Applying the perturbation requires modifying the hidden state before the next forward pass. Since Phase 12's model outputs `corrected_states` (not the raw hidden states that feed the next layer), the perturbation would need to be injected into the model's internal state. This is architecturally invasive — the very thing Phase 13b showed is harmful.

**Alternative**: Instead of perturbing the state, perturb the **token input**. When the cycle breaker fires, inject a special "think again" token or repeat the expression prefix. This is non-invasive and keeps the SSM dynamics clean.

```python
# Non-invasive cycle breaking: modify the INPUT, not the STATE
def break_cycle_via_input(generated_ids, expression):
    """
    When stuck in a cycle, re-inject the expression
    as a "reminder" to re-process from scratch.
    """
    reminder_tokens = tokenize(f" {expression}")
    generated_ids.extend(reminder_tokens)
    return generated_ids
```

### Ablation Matrix for 14D

| Strategy | Invasive? | Tests |
|----------|:-:|---|
| Random noise on state | Yes | Baseline perturbation |
| Anti-cycling on state | Yes | Directed perturbation |
| Input re-injection | No | Non-invasive cycle breaking |
| Temperature spike (sample with T=2 for 1 step) | No | Diversity injection |

---

## 14E: Integration and Figures

### End-to-End Pipeline

```python
def run_phase14_full(model, confusion_detector, cycle_breaker,
                     test_ds, device):
    """
    Full Phase 14 pipeline:
      1. Generate with confusion-gated halting (14C)
      2. If patience exhausted, apply cycle breaking (14D)
      3. Evaluate with corrected purity measurement

    Compare against Phase 12 baseline (no intervention).
    """
    # Baseline
    print("--- Phase 12 Baseline ---")
    baseline_gen = CompressibleGenerator(model, device=device)
    baseline_results = run_generation(baseline_gen, test_ds)

    # Confusion-gated
    print("--- Phase 14: Confusion-Gated ---")
    gated_gen = ConfusionGatedGenerator(
        model, confusion_detector, device,
        base_threshold=0.5, max_boost=0.4,
        extended_max_length=100,
    )
    gated_results = run_generation(gated_gen, test_ds)

    # Corrected purity on both
    print("--- Corrected Purity ---")
    for name, results in [('Baseline', baseline_results),
                          ('Gated', gated_results)]:
        for r in results:
            purity = evaluate_purity_corrected(
                model, r['state_vectors'], r['generated_ids'],
                r['expression'], device, GeodesicPurity()
            )
            r['corrected_purity'] = purity['purity']

    return baseline_results, gated_results
```

### Figures

**Fig 25: The Halt Control Problem** (motivating figure)

```
Layout: 1×3

Left:    Phase 12 trajectory that halted correctly
         (true convergence, correct answer)
         Plot: H, Sc, halt_conf over steps
         Show: Sc drops to 0, halt fires → good

Center:  Phase 12 trajectory that halted prematurely
         (false convergence, wrong answer)
         Plot: H, Sc, halt_conf over steps
         Show: Sc stays high, halt fires anyway → bad

Right:   Distribution of Sc at halt for true vs false convergence
         Two overlapping histograms
         Show: these are separable → Confusion Head is feasible
```

**Fig 26: Confusion Detection Performance**

```
Layout: 1×2

Left:    ROC curve for false convergence detection
         Three lines: Sc alone, ∇H alone, combined (three-signal)
         Compare: non-parametric vs learned Confusion Head

Right:   Detection lead time distribution
         Histogram: how many steps before halt does confusion fire?
         Must show: detection leads halt by ≥ 2 steps
```

**Fig 27: Halt Gating Results** (the money figure)

```
Layout: 1×3

Left:    Accuracy by tier: baseline vs each condition (grouped bar)
         Conditions: Phase 12, constant boost, CycleDetector, ConfusionHead

Center:  Accuracy vs mean tokens (Pareto frontier)
         Each condition is a point; connect for efficiency curve
         X = mean tokens, Y = accuracy
         Target: upper-left quadrant (fewer tokens, higher accuracy)

Right:   True convergence rate by condition
         Bar chart showing the 3.9% → ??% improvement
         Annotate: "89.7% accuracy when truly converged"
```

**Fig 28: The Corrected Story** (summary figure for paper)

```
Layout: 2×2

Top-left:     Geodesic purity by phase (corrected)
              Phase 11 (no constraints) vs Phase 12 (soft) vs Phase 13b (hard)
              Show: 83% purity for Phase 12 — constraints work

Top-right:    Accuracy by phase
              Phase 11 vs Phase 12 vs Phase 14 (confusion-gated)
              Show: purity + halt control → best accuracy

Bottom-left:  True convergence rate by phase
              Show: 3.9% → ??% with halt gating
              
Bottom-right: The precision localization:
              "Geometry: SOLVED (83% purity)"
              "Halting: Phase 12 BROKEN, Phase 14 IMPROVED"
              Table format with checkmarks
```

---

## Global Success Criteria

| # | Criterion | Target | Paper Section |
|---|-----------|--------|---------------|
| 1 | $S_c$ discriminates true/false convergence | AUC > 0.70 | Appendix A |
| 2 | Three-signal combined AUC | > 0.80 | Appendix A |
| 3 | Confusion detection F1 (either method) | > 0.70 | §5.2 |
| 4 | T2 Accuracy with halt gating | > 72% (+6pp) | §5 |
| 5 | True convergence rate with halt gating | > 10% (from 3.9%) | §5 |
| 6 | Detection adds value over constant boost | Significant ($p < 0.05$) | §5.2 |
| 7 | Token overhead | < 2× baseline | §6 |
| 8 | Corrected purity preserved during extended generation | > 75% | §4 |

---

## Contingency Plans

<details>
<summary><strong>If 14A-0 shows accuracy INCREASES with higher threshold</strong></summary>

This is the best-case scenario. It means the model can self-correct given more time — the limit cycles eventually break on their own. Phase 14 becomes primarily about halt calibration, not cycle breaking. Skip 14D entirely. The Confusion Head's value is in **selective patience** — extending the budget only for trajectories that need it, keeping the token cost low for easy examples.

</details>

<details>
<summary><strong>If 14A-0 shows accuracy FLAT across thresholds</strong></summary>

The model is genuinely stuck. More time doesn't help. The limit cycles are stable. This means 14D (cycle breaking) is essential. But check first: does the model still achieve 83% purity with extended generation? If purity degrades over longer trajectories, the model is drifting off the manifold, and the problem is deeper than halting.

</details>

<details>
<summary><strong>If 14A-1 shows Sc doesn't discriminate convergence types (AUC < 0.60)</strong></summary>

The three-signal framework from Appendix A doesn't hold empirically. This is a significant negative result for the theory. Options:
1. Try different features (state norm, spectral gap, recurrence statistics)
2. The true/false convergence distinction may not be detectable from trajectory features alone
3. Appendix A needs revision: the certainty coordinate exists but doesn't cleanly separate the convergence types

This doesn't kill Phase 14 — the non-parametric CycleDetector uses delta statistics, not entropy signals. But it weakens the theoretical narrative.

</details>

<details>
<summary><strong>If the non-parametric CycleDetector achieves F1 > 0.80</strong></summary>

Skip the learned Confusion Head (14B-2). The simple version is sufficient. This is actually the best outcome for the paper — it means false convergence is detectable from first principles (state delta statistics), not requiring learned features. Report the CycleDetector as the primary result and the learned head as an ablation.

</details>

<details>
<summary><strong>If halt gating improves accuracy but token overhead > 3×</strong></summary>

The model needs too many extra steps. Implement an adaptive patience budget:
- Start with normal threshold (0.5)
- Raise threshold when confusion detected
- Apply cycle-breaking after K confused steps
- Hard ceiling at 2× baseline tokens
- Accept reduced accuracy improvement for acceptable efficiency

The paper should report both the unconstrained-budget accuracy (what's achievable) and the budget-constrained accuracy (what's practical).

</details>

---

## Connection to the Paper

After Phase 14, the experimental narrative becomes:

```
Phase 9:   Measurement operator exists (r = -0.836, τ = -2.0)
Phase 11:  False convergence discovered (90%+ rate)
Phase 12:  Soft constraints → 83% geodesic purity ← GEOMETRY WORKS
Phase 13b: Hard gating destroys structure (18% purity) ← NEGATIVE RESULT
Phase 14A: Three-signal framework validated empirically
Phase 14C: Confusion-gated halting → accuracy + convergence improvement
Phase 14E: Halt control on geometrically sound manifold → best results
```

The paper's central claim updates from:

> ~~"Rules must be encoded as metric singularities"~~

to:

> "Soft constraint losses create persistent geometric structure (83% purity). The remaining error is a halting problem, not a geometry problem. A confusion-detecting halt gate improves accuracy by selectively extending generation for trajectories stuck in limit cycles, while preserving the manifold's algebraic properties."