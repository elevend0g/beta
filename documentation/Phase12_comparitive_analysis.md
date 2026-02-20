No. The training is clean — the bug is only in the **evaluation metric**, not the trained model.

## What's Actually Wrong

The `RuleConstraintLoss` during training operates on teacher-forced states where indexing is correct:

```python
# TRAINING (correct) — states is [B, L, d_state] from full forward pass
s_before = states[b, bp]   # bp = token position → correct state
s_after  = states[b, ap]   # ap = token position → correct state
identity_loss += F.mse_loss(s_after, s_before)  # ✓ Right states compared
```

The bug is only in `GeodesicPurity.evaluate_trajectory()` during **generation**, where generation-step indices are conflated with token positions. The Phase 12 model learned the right constraints — we just measured the outcome wrong.

## What To Do Instead

Re-evaluate the **existing checkpoint** with a corrected purity measurement. One additional forward pass on the complete generated sequence gives properly-indexed states:

```python
def evaluate_purity_corrected(model, state_vectors_gen, generated_ids,
                               expression, device, purity_eval):
    """
    Correct purity evaluation: run one final forward pass on the
    complete generated sequence to get states aligned with token positions.
    
    state_vectors_gen: original generation states (for convergence/oscillation)
    generated_ids: full token sequence including prompt + generated
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        
        # Truncate to max_seq_len if needed
        max_pos = getattr(model, 'max_seq_len', 256)
        if hasattr(model, 'pos_encoding'):
            max_pos = model.pos_encoding.num_embeddings
        if input_tensor.size(1) > max_pos:
            input_tensor = input_tensor[:, :max_pos]
        
        outputs = model(input_tensor)
        # states_sequence[0, pos, :] = state at token position pos
        full_states = outputs['states_sequence'][0].cpu()  # [L, d_state]
    
    # Now indexing is correct: full_states[pos] = state at token pos
    position_states = [full_states[i] for i in range(full_states.size(0))]
    
    purity_result = purity_eval.evaluate_trajectory(
        position_states, expression, generated_ids[:full_states.size(0)]
    )
    
    return purity_result
```

Slot this into `run_rim_generation` alongside the existing (wrong) purity call:

```python
def run_rim_generation_corrected(model, test_ds, device):
    """Phase 12 generation with corrected purity evaluation."""
    generator = CompressibleGenerator(model, device=device)
    purity_eval = GeodesicPurity()
    
    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        gen_result = generator.generate(ex['expression'])
        
        # Convergence analysis (uses generation-step states — correct for this purpose)
        osc_result = detect_oscillation(gen_result['state_vectors'])
        convergence = classify_convergence(gen_result, osc_result)
        
        # CORRECTED purity: single forward pass on complete sequence
        purity_result = evaluate_purity_corrected(
            model, gen_result['state_vectors'], gen_result['generated_ids'],
            ex['expression'], device, purity_eval
        )
        
        is_correct = (gen_result['parsed_answer'] == ex['answer'])
        
        constraint_at_halt = 0.0
        if gen_result['state_vectors']:
            last_state = gen_result['state_vectors'][-1]
            if len(gen_result['state_vectors']) > 1:
                prev_state = gen_result['state_vectors'][-2]
                constraint_at_halt = (last_state - prev_state).norm().item()
        
        results.append({
            'expression': ex['expression'],
            'tier': ex['tier'],
            'ground_truth': ex['answer'],
            'parsed_answer': gen_result['parsed_answer'],
            'is_correct': is_correct,
            'reasoning_tokens': gen_result['reasoning_tokens'],
            'total_tokens': gen_result['total_tokens'],
            'stop_reason': gen_result['stop_reason'],
            'halt_confidences': gen_result['halt_confidences'],
            'state_entropies': gen_result['state_entropies'],
            'convergence': convergence,
            'has_oscillation': osc_result['has_oscillation'],
            'max_rho': {str(k): v for k, v in osc_result['max_rho'].items()},
            'effective_ops': ex['effective_ops'],
            'num_ops': ex['num_ops'],
            'generated_text': gen_result['generated_text'][:200],
            'geodesic_purity': purity_result['purity'],
            'n_constraints': purity_result['n_constraints'],
            'n_satisfied': purity_result.get('n_satisfied', 0),
            'constraint_at_halt': constraint_at_halt,
        })
    
    return results
```

## What This Gets You

```bash
# ~20 minutes, not ~60 minutes for full retrain
python src/rule_initialization.py --skip-training --eval-only
```

| What changes | What stays the same |
|---|---|
| Geodesic purity numbers (could go up or down) | Model weights |
| Constraint-at-halt measurement | Accuracy per tier |
| Purity vs correctness correlation | Convergence rates |
| | Halt F1, token counts |

## Why the Corrected Number Matters

There are two possible outcomes, and they tell very different stories:

**If true purity is HIGHER than 34.6%:**

> The soft constraints in Phase 12 work *better than we measured*. The training-time geometric loss actually does persist into generation. The paper's claim that "rules constrain the manifold" gets stronger evidence.

**If true purity is LOWER than 34.6%:**

> The 34.6% was an artifact of accidental state stability at matching generation-step intervals. The constraint loss minimizes during teacher forcing but leaves even less trace during generation than we thought. This sharpens the motivation for Phase 14: training-time constraints alone are insufficient.

Either way, you need the correct number before publishing anything about geodesic purity. And you get it in 20 minutes without retraining.

## Also Worth Doing: Probe Experiment A

While the eval runs, attach an Op Detector probe to the Phase 12 backbone:

```python
# Freeze Phase 12 backbone, train only the Op Detector
model = PNA_SSM_RIM(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                     max_seq_len=64).to(device)
model.load_state_dict(torch.load('results/rim_model.pt', map_location=device))

for p in model.parameters():
    p.requires_grad = False

# The Op Detector reads from the same intermediate hidden states
# that Phase 13b used — output of layer 3's norm
op_detector = OperationDetector(d_model=512, n_op_types=6).to(device)
optimizer = torch.optim.Adam(op_detector.parameters(), lr=1e-3)

for epoch in range(10):
    for batch in rim_train_loader:
        with torch.no_grad():
            outputs = model(batch['input_ids'].to(device))
            # Extract hidden states from intermediate layer
            # (would need a forward hook or model modification to get layer 3 output)
            hidden = outputs['states_sequence']  # Using states as proxy
        
        op_logits = op_detector(hidden)
        # ... standard op detection training ...
```

**This takes 15 minutes** and answers: "Does the Phase 12 backbone carry the same operation-type information that Phase 13b found?" If yes (likely), Phase 14's Confusion Head has a solid foundation.

## Bottom Line

The model is good. The evaluation was wrong. Fix the ruler, not the table.