1. Load basin model (results/basin_model.pt)
2. Generate answers for 1000 test examples (same dataset, seed=456)
3. For each incorrect example (expected ~154):
   
   a. OUTPUT COMPLETENESS
      - Does generated text contain "Result:"?
      - If no → INCOMPLETE_OUTPUT
      
   b. INTERMEDIATE VERIFICATION (if Result: exists)
      - Extract all "X op Y = Z" patterns
      - Verify each: does X op Y == Z?
      - Any wrong → WRONG_INTERMEDIATE (which step? which operation?)
      - All right but final wrong → WRONG_COMPOSITION
      
   c. REGIME AT FINAL STEP
      - Compute Sc, ∇H, H from generation states
      - Classify: ORBITING / CONVERGING / PROGRESSING / DIFFUSING
      
   d. EXPRESSION DIFFICULTY
      - num_ops, effective_ops (after compression)
      - Is error rate correlated with difficulty?

4. Print:
   Failure breakdown (completeness × regime × difficulty)
   Specific error examples (5 representative cases per failure mode)
   Comparison with Phase 12/15 failure analysis