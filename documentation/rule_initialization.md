 Here is the complete implementation roadmap for **Rule-Initialized Models (RIMs)** in your experimental environment. This builds directly on your existing SSM architecture (`PNA_SSM`) and integrates with your Halt/Confusion heads.

---

## Phase 0: Define the Rule System

Create `src/rules/` directory:

```python
# src/rules/base.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class Rule(ABC):
    """Abstract base for manifold constraints."""
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        
    @abstractmethod
    def constraint_function(self, state: torch.Tensor, context: dict) -> torch.Tensor:
        """
        Returns constraint violation scalar (0 = satisfied).
        Shape: [batch]
        """
        pass
    
    @abstractmethod
    def projection_matrix(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns projection operator onto constraint nullspace.
        Shape: [batch, state_dim, state_dim]
        """
        pass

# src/rules/algebraic.py
class AdditiveInverseRule(Rule):
    """
    Enforces: a + b - a = b
    Detects states claiming "result = c" where c ≠ expected algebraic simplification
    """
    def __init__(self, weight=10.0):
        super().__init__("additive_inverse", weight)
        
    def constraint_function(self, state, context):
        """
        context must contain: 
        - 'expression': tokenized expression
        - 'expected_result': computed ground truth
        - 'current_claim': what model currently claims (from decoder probe)
        """
        current_claim = context.get('current_claim', state[:, :10])  # Assume first 10 dims encode value
        expected = context['expected_result']
        
        # violation = |claim - expected| for compressible patterns
        # This is a simplified geometric proxy
        violation = torch.norm(current_claim - expected, dim=-1)
        return violation
    
    def projection_matrix(self, state):
        # Project onto subspace where additive cancellation holds
        # Implementation: orthogonal projection to constraint manifold
        batch, dim = state.shape
        # Identity minus constraint gradient outer product
        I = torch.eye(dim, device=state.device).unsqueeze(0).repeat(batch, 1, 1)
        
        # Simplified: assume constraint is linear (Ax = b)
        # In practice, you'd compute the Jacobian of constraint_function
        return I  # Placeholder for actual constraint Jacobian computation

class AssociativityRule(Rule):
    """Enforces (a+b)+c = a+(b+c)"""
    pass  # Implement similar structure

class MultiplicativeIdentityRule(Rule):
    """Enforces a*1 = a, a/1 = a"""
    pass
```

---

## Phase 1: Implement RuleProjector

```python
# src/models/rule_projector.py
import torch.nn as nn

class RuleProjector(nn.Module):
    """
    Projects SSM states onto rule-satisfying submanifold.
    Can operate in "hard" (always project) or "soft" (learned projection) mode.
    """
    def __init__(self, state_dim, rules, mode='hybrid'):
        super().__init__()
        self.state_dim = state_dim
        self.rules = nn.ModuleList(rules)
        self.mode = mode  # 'hard', 'soft', or 'hybrid'
        
        # Soft projection: learned correction
        self.correction_mlp = nn.Sequential(
            nn.Linear(state_dim + len(rules), state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
            nn.Tanh()  # Bounded correction
        )
        
        # Rule encoder: which rules are active in current context
        self.rule_embedding = nn.Embedding(len(rules), 64)
        
    def forward(self, state, rule_indices=None, context=None):
        """
        Args:
            state: [batch, state_dim]
            rule_indices: list of active rule indices for this batch
            context: dict with expression metadata for constraint calculation
        """
        batch_size = state.shape[0]
        original_state = state.clone()
        
        if self.mode == 'hard' or context is None:
            # Simple hard projection (geometric constraint)
            return self._hard_project(state, rule_indices, context)
        elif self.mode == 'soft':
            # Learned projection with penalty
            return self._soft_project(state, rule_indices, context)
        else:  # hybrid
            hard_state = self._hard_project(original_state, rule_indices, context)
            # Blend between hard and soft based on training progress
            alpha = self._get_constraint_strength()  # annealed during training
            return alpha * hard_state + (1 - alpha) * self._soft_project(state, rule_indices, context)
    
    def _hard_project(self, state, rule_indices, context):
        """Successive projection onto each active constraint manifold"""
        for idx in rule_indices:
            rule = self.rules[idx]
            # Geometric projection: state -> state - J^T(JJ^T)^{-1}C(state)
            violation = rule.constraint_function(state, context)
            proj = rule.projection_matrix(state)
            
            # Project state to nullspace of constraint
            state = torch.bmm(proj, state.unsqueeze(-1)).squeeze(-1)
            
            # If violation is high, trigger "fall off manifold" signal
            if violation.mean() > 0.5:
                return state, True  # True = violated constraint
        
        return state, False
    
    def _soft_project(self, state, rule_indices, context):
        """Learned correction toward constraint satisfaction"""
        if rule_indices is None:
            return state
            
        # Encode which rules should apply
        rule_conds = torch.zeros(state.size(0), len(self.rules), device=state.device)
        rule_conds[:, rule_indices] = 1.0
        
        combined = torch.cat([state, rule_conds], dim=-1)
        correction = self.correction_mlp(combined)
        
        # Apply correction ( pushes state toward valid region )
        return state + 0.1 * correction  # Small step
    
    def compute_loss(self, state, rule_indices, context):
        """Constraint violation loss for training"""
        total_violation = 0.0
        for idx in rule_indices:
            rule = self.rules[idx]
            violation = rule.constraint_function(state, context)
            total_violation += rule.weight * violation.mean()
        return total_violation
    
    def _get_constraint_strength(self):
        # Annealing schedule: start soft, become hard
        # Access global step via trainer or module hook
        return min(1.0, 0.01 * self.current_epoch)  # Example annealing
```

---

## Phase 2: Modify the SSM

Extend your existing `PNA_SSM` class:

```python
# src/models/pna_ssm.py (modifications)

class PNA_SSM_RIM(PNA_SSM):
    def __init__(self, input_dim, state_dim, output_dim, num_rules=3):
        super().__init__(input_dim, state_dim, output_dim)
        
        # Add rule projector
        rules = [
            AdditiveInverseRule(weight=10.0),
            AssociativityRule(weight=5.0),
            MultiplicativeIdentityRule(weight=2.0)
        ]
        self.rule_projector = RuleProjector(state_dim, rules, mode='hybrid')
        
        # Probes for extracting semantic content from state
        self.value_decoder = nn.Linear(state_dim, 20)  # Decodes current numerical claim
        
        # Track manifold deviation for Confusion Head
        self.manifold_violation = 0.0
        
    def forward(self, x, prev_state, rule_context=None):
        """
        Args:
            x: input token [batch, input_dim]
            prev_state: [batch, state_dim]
            rule_context: dict with 'expression', 'tier', etc.
        """
        # Standard SSM step
        proposed_state, output = super().step(x, prev_state)
        
        # Extract current claim for constraint checking
        if rule_context is not None:
            current_claim = self.value_decoder(proposed_state)
            rule_context['current_claim'] = current_claim
            
            # Project onto rule manifold
            constrained_state, violated = self.rule_projector(
                proposed_state, 
                rule_indices=self._get_active_rules(rule_context),
                context=rule_context
            )
            
            self.manifold_violation = violated.float().mean()
            
            # If violated and we're in "strict mode", trigger confusion
            if violated and hasattr(self, 'confusion_head'):
                self.confusion_signal = True
        else:
            constrained_state = proposed_state
            
        return constrained_state, output
    
    def _get_active_rules(self, context):
        """Determine which rules apply based on expression structure"""
        tier = context.get('tier', 0)
        rules = []
        if tier > 0:  # Compressible tiers
            rules.append(0)  # Additive inverse
        if '*' in context.get('expression', ''):
            rules.append(2)  # Multiplicative identity
        return rules
    
    def get_aux_losses(self):
        """Additional losses for constraint satisfaction"""
        return {
            'constraint_violation': self.rule_projector.compute_loss(
                self.current_state, 
                self.active_rules,
                self.current_context
            ) if hasattr(self, 'current_context') else 0.0
        }
```

---

## Phase 3: Modify Training Loop

Update `src/compressible_task.py`:

```python
# In training loop

for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
    # metadata contains: expression string, tier, expected operations
    
    optimizer.zero_grad()
    
    # Forward pass with rule context
    rule_context = {
        'expression': metadata['expr'],
        'tier': metadata['tier'],
        'expected_result': targets['final_answer'],
        'full_steps': metadata['ground_truth_steps']
    }
    
    states, outputs, halt_probs = model(
        inputs, 
        return_states=True,
        rule_context=rule_context
    )
    
    # Standard losses
    ce_loss = criterion(outputs, targets['tokens'])
    halt_loss = halt_criterion(halt_probs, targets['halt_pos'])
    
    # NEW: Rule constraint loss
    constraint_loss = model.get_aux_losses()['constraint_violation']
    
    # Combined loss with annealing
    # Early epochs: focus on learning the task
    # Later epochs: enforce hard constraints
    epoch_factor = min(1.0, epoch / 20)
    total_loss = ce_loss + 0.1 * halt_loss + (1.0 * epoch_factor) * constraint_loss
    
    total_loss.backward()
    optimizer.step()
    
    # Logging
    if batch_idx % 10 == 0:
        print(f"Loss: {total_loss:.4f} | "
              f"CE: {ce_loss:.4f} | "
              f"Halt: {halt_loss:.4f} | "
              f"Constraint: {constraint_loss:.4f} | "
              f"Manifold Violations: {model.manifold_violation:.2%}")
```

---

## Phase 4: Evaluation Metrics

Add `GeodesicPurity` metric:

```python
# src/evaluation/rim_metrics.py

class GeodesicPurity:
    """Measures % of trajectory spent in valid (rule-satisfying) regions"""
    
    def __init__(self, rules):
        self.rules = rules
        
    def evaluate_trajectory(self, trajectory_states, context):
        """
        Args:
            trajectory_states: list of [state] over time steps
            context: expression metadata
        Returns:
            purity_score: float [0,1]
            violation_points: list of time steps where rules violated
        """
        valid_steps = 0
        violations = []
        
        for t, state in enumerate(trajectory_states):
            # Check all applicable rules
            violated = False
            for rule in self.rules:
                if self._rule_applies(rule, context, t):
                    v = rule.constraint_function(state.unsqueeze(0), context)
                    if v.item() > 0.5:
                        violated = True
                        violations.append(t)
                        break
            
            if not violated:
                valid_steps += 1
                
        purity = valid_steps / len(trajectory_states)
        
        return {
            'purity': purity,
            'violation_points': violations,
            'avg_violation_time': np.mean(violations) if violations else 0
        }
    
    def _rule_applies(self, rule, context, timestep):
        # Logic to determine if rule should be checked at this timestep
        return True  # Simplified
```

---

## Phase 5: Instantiate and Run

```python
# src/run_rim_experiment.py

def run_rim_compressible_trial():
    # Initialize model
    model = PNA_SSM_RIM(
        input_dim=128,
        state_dim=256,
        output_dim=64,
        num_rules=3
    ).cuda()
    
    # Dataset with rich metadata
    train_dataset = CompressibleArithmeticDataset(
        n=8000,
        rules=['additive_inverse', 'associativity'],
        provide_metadata=True  # Includes expression structure
    )
    
    # Optimizer with rule-specific LR (lower for projection layers)
    base_params = [p for n, p in model.named_parameters() if 'rule' not in n]
    rule_params = [p for n, p in model.named_parameters() if 'rule' in n]
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 1e-3},
        {'params': rule_params, 'lr': 1e-4}  # Slower learning for constraints
    ])
    
    # Train
    for epoch in range(40):
        train_epoch(model, train_dataset, optimizer, epoch)
        
        # Evaluate Geodesic Purity
        purity_metric = GeodesicPurity(model.rule_projector.rules)
        results = evaluate_purity(model, val_dataset, purity_metric)
        
        print(f"Epoch {epoch}: Geodesic Purity = {results['mean_purity']:.2%}")
        print(f"  False Convergence reduced to: {results['false_conv_rate']:.2%}")
        
        # Target: Purity > 95%, False Convergence < 5%, maintain token economy
        if results['mean_purity'] > 0.95 and results['false_conv_rate'] < 0.05:
            print("RIM convergence achieved!")
            break
```

---

## Phase 6: Debugging Checklist

**Problem: RuleProjector kills gradients**
- **Fix**: Use `torch.nn.utils.clip_grad_norm_` with large threshold (10.0)
- Ensure projection isn't too aggressive early in training (annealing)

**Problem: Model can't learn (high constraint loss)**
- **Fix**: Start with `mode='soft'` for 10 epochs, then switch to `mode='hybrid'`
- Verify your constraint_function actually returns 0 for valid states

**Problem: Geodesic Purity high but accuracy still low**
- **Fix**: The rules are being satisfied but wrong final answer?
- **Cause**: Constraint function checks intermediate steps but not final result
- **Fix**: Add terminal constraint: final state must decode to correct answer

**Problem: No token economy improvement**
- **Fix**: Ensure constraint projection preserves shortcut paths
- Check that additive_inverse rule doesn't force step-by-step computation

---

## Expected Results

After successful implementation:

| Metric | Baseline (Phase 11) | RIM (Target) |
|--------|-------------------|--------------|
| Tier 2 Accuracy | 65.2% | **>95%** |
| False Convergence | 90% | **<5%** |
| Geodesic Purity | ~40% | **>95%** |
| Tokens (Tier 2) | 16.5 | **<20** (maintain efficiency) |
| Constraint Violation | N/A | **<0.01** |

**Success Criterion**: Model finds shortcuts automatically (low tokens) but **only valid shortcuts** (high purity), proving that rules shape the manifold topology rather than filter post-hoc.

Run this and report back on Geodesic Purity scores.