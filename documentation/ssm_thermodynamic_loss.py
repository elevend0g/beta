"""
SSM-Adapted Thermodynamic Loss (L_th)

Key innovation: In SSMs, we can measure BOTH token entropy and state entropy.
The state h_t should "know" about the final answer before the model generates it.

This creates a dual-entropy training signal:
1. Token entropy (same as Transformer version)
2. State entropy (unique to SSMs)

Hypothesis: Training on state entropy collapse should produce sharper halt
calibration because the state is a direct encoding of "progress toward goal."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMThermodynamicLoss(nn.Module):
    """
    Thermodynamic loss adapted for State Space Models.
    
    Extends the base PNA thermodynamic loss with SSM-specific components:
    - L_state: Penalty for high state entropy when answer is determined
    - State-based halt prediction (more reliable than token-based)
    
    L_th = L_ce + α·L_energy + β·L_halt + γ·L_state
    
    The γ·L_state term is NEW and unique to SSMs.
    """
    
    def __init__(self, 
                 alpha: float = 0.05,      # Energy penalty weight
                 beta: float = 0.1,        # Halt reward weight
                 gamma: float = 0.05,      # State entropy penalty weight (NEW)
                 halt_token_id: int = None,
                 d_state: int = 16):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.halt_token_id = halt_token_id
        self.d_state = d_state
    
    def compute_token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Standard token entropy: H(next_token)
        Same as Transformer version.
        """
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)
        return entropy
    
    def compute_state_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        NEW: Entropy of the SSM state itself.
        
        The state h_t is a d_state-dimensional vector that encodes
        the model's "understanding" of the problem so far.
        
        We want this state to collapse (low entropy) as the model
        becomes more certain about the final answer.
        
        Args:
            state: [batch, d_state] or [batch, seq_len, d_state]
        
        Returns:
            entropy: [batch] or [batch, seq_len]
        """
        # Normalize state to probability distribution
        # (This is one interpretation - alternatives exist)
        state_probs = F.softmax(state, dim=-1)
        
        # Shannon entropy
        entropy = -(state_probs * torch.log2(state_probs + 1e-9)).sum(dim=-1)
        
        return entropy
    
    def compute_state_delta_h(self, states_sequence: torch.Tensor) -> torch.Tensor:
        """
        Measure how much the state entropy reduces at each step.
        
        This is the SSM equivalent of "measurement-as-collapse":
        each reasoning step should collapse the state's uncertainty.
        
        Args:
            states_sequence: [batch, seq_len, d_state]
        
        Returns:
            delta_h: [batch, seq_len] - entropy reduction per step
        """
        state_entropy = self.compute_state_entropy(states_sequence)  # [B, T]
        
        delta_h = torch.zeros_like(state_entropy)
        delta_h[:, 1:] = F.relu(state_entropy[:, :-1] - state_entropy[:, 1:])
        
        return delta_h
    
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor,
                halt_confidence: torch.Tensor,
                states_sequence: torch.Tensor,  # NEW: full state trajectory
                reasoning_mask: torch.Tensor,
                result_token_positions: torch.Tensor) -> dict:
        """
        Args:
            logits: [B, T, V] - next token logits
            targets: [B, T] - ground truth tokens
            halt_confidence: [B, T, 1] - model's halt signal
            states_sequence: [B, T, d_state] - SSM states at each position (NEW)
            reasoning_mask: [B, T] - 1 where tokens are reasoning
            result_token_positions: [B] - index of Result: token
        
        Returns:
            dict with total loss and component breakdowns
        """
        B, T, V = logits.shape
        
        # === Component 1: Cross-Entropy (accuracy) ===
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction='mean'
        )
        
        # === Component 2: Token Energy Penalty (same as Transformer) ===
        with torch.no_grad():
            token_entropy = self.compute_token_entropy(logits)  # [B, T]
            token_delta_h = torch.zeros_like(token_entropy)
            token_delta_h[:, 1:] = F.relu(token_entropy[:, :-1] - token_entropy[:, 1:])
            
            # Stagnation: tokens that don't reduce token entropy
            token_stagnation = (token_delta_h < 0.01).float() * reasoning_mask
        
        energy_cost = reasoning_mask.float() * (1.0 + 2.0 * token_stagnation)
        energy_loss = energy_cost.mean()
        
        # === Component 3: Halt Reward/Penalty (same as Transformer) ===
        halt_conf = halt_confidence.squeeze(-1)
        halt_target = torch.zeros_like(halt_conf)
        for b in range(B):
            pos = result_token_positions[b].item()
            if pos < T:
                halt_target[b, max(0, pos-2):pos] = 0.5
                halt_target[b, pos:] = 1.0
        
        halt_loss = F.binary_cross_entropy(halt_conf, halt_target, reduction='mean')
        
        # === Component 4: State Entropy Penalty (NEW - SSM specific) ===
        with torch.no_grad():
            state_entropy = self.compute_state_entropy(states_sequence)  # [B, T]
            state_delta_h = self.compute_state_delta_h(states_sequence)  # [B, T]
            
            # State stagnation: when state entropy isn't collapsing
            state_stagnation = (state_delta_h < 0.01).float() * reasoning_mask
        
        # Penalty for high state entropy in reasoning region
        # The state should be collapsing toward certainty
        state_entropy_penalty = (state_entropy * reasoning_mask).mean()
        
        # Bonus penalty for state stagnation
        # (state isn't changing even though we're still generating tokens)
        state_stagnation_penalty = state_stagnation.mean()
        
        state_loss = state_entropy_penalty + state_stagnation_penalty
        
        # === Total Loss ===
        total = (ce_loss + 
                 self.alpha * energy_loss + 
                 self.beta * halt_loss + 
                 self.gamma * state_loss)
        
        # === Diagnostics ===
        with torch.no_grad():
            # Compare token vs state entropy reduction
            token_collapse_rate = token_delta_h[reasoning_mask.bool()].mean().item() \
                                  if reasoning_mask.any() else 0.0
            state_collapse_rate = state_delta_h[reasoning_mask.bool()].mean().item() \
                                  if reasoning_mask.any() else 0.0
            
            # Which collapses faster?
            state_leads_token = state_collapse_rate > token_collapse_rate
        
        return {
            "total": total,
            "ce_loss": ce_loss.item(),
            "energy_loss": energy_loss.item(),
            "halt_loss": halt_loss.item(),
            "state_loss": state_loss.item(),  # NEW
            "token_delta_h": token_collapse_rate,
            "state_delta_h": state_collapse_rate,  # NEW
            "state_leads": state_leads_token,  # NEW diagnostic
            "token_stagnation_rate": token_stagnation.mean().item(),
            "state_stagnation_rate": state_stagnation.mean().item()  # NEW
        }


class StateAwareHaltPredictor(nn.Module):
    """
    Alternative halt head design that uses BOTH state and token embeddings.
    
    Hypothesis: The SSM state h_t contains more reliable halt information
    than the token embeddings alone.
    
    This can be tested via ablation:
    1. State-only halt head (uses h_t)
    2. Token-only halt head (uses transformer-style sequence embedding)
    3. Hybrid halt head (uses both) <- this version
    """
    
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        
        # Process state information
        self.state_processor = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Process token embedding information
        self.token_processor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        # Combine both signals
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),  # 32 + 32 = 64
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, token_embedding: torch.Tensor, 
                state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embedding: [batch, d_model] - token representation
            state: [batch, d_state] - SSM state
        
        Returns:
            halt_confidence: [batch, 1]
        """
        state_features = self.state_processor(state)
        token_features = self.token_processor(token_embedding)
        
        combined = torch.cat([state_features, token_features], dim=-1)
        
        return self.fusion(combined)


class SSMAdaptiveGovernor:
    """
    Training-time governor adapted for SSMs.
    
    Adjusts α (token energy penalty) and γ (state entropy penalty)
    based on training dynamics.
    
    Key insight: If state entropy is collapsing nicely but token
    entropy isn't, increase α to force concise generation.
    
    If token entropy is collapsing but state isn't, increase γ to
    force the state to encode more information.
    """
    
    def __init__(self,
                 initial_alpha: float = 0.05,
                 initial_gamma: float = 0.05,
                 min_alpha: float = 0.01,
                 max_alpha: float = 0.5,
                 min_gamma: float = 0.01,
                 max_gamma: float = 0.3,
                 sensitivity: float = 1.15):
        self.alpha = initial_alpha
        self.gamma = initial_gamma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.sensitivity = sensitivity
        
        self.token_stagnation_history = []
        self.state_stagnation_history = []
        self.window_size = 100
    
    def step(self, metrics: dict) -> tuple[float, float]:
        """
        Adapt both α and γ based on training progress.
        
        Args:
            metrics: dict from SSMThermodynamicLoss
        
        Returns:
            (new_alpha, new_gamma)
        """
        self.token_stagnation_history.append(metrics["token_stagnation_rate"])
        self.state_stagnation_history.append(metrics["state_stagnation_rate"])
        
        if len(self.token_stagnation_history) < self.window_size:
            return self.alpha, self.gamma
        
        recent_token_stagnation = sum(self.token_stagnation_history[-self.window_size:]) / self.window_size
        recent_state_stagnation = sum(self.state_stagnation_history[-self.window_size:]) / self.window_size
        
        # Adjust α (token energy penalty)
        if recent_token_stagnation > 0.1:
            self.alpha = min(self.alpha * self.sensitivity, self.max_alpha)
        else:
            self.alpha = max(self.alpha * 0.98, self.min_alpha)
        
        # Adjust γ (state entropy penalty)
        if recent_state_stagnation > 0.1:
            self.gamma = min(self.gamma * self.sensitivity, self.max_gamma)
        else:
            self.gamma = max(self.gamma * 0.98, self.min_gamma)
        
        return self.alpha, self.gamma


# ===== Evaluation Functions =====

def compare_entropy_dynamics(transformer_model, ssm_model, 
                              example, tokenizer):
    """
    Side-by-side comparison of entropy collapse in Transformer vs SSM.
    
    For Transformer: Only token entropy available
    For SSM: Both token entropy AND state entropy
    
    Returns data for visualization.
    """
    results = {
        'transformer': {'token_entropy': []},
        'ssm': {'token_entropy': [], 'state_entropy': []}
    }
    
    tokens = tokenizer.encode(example.dense_str)
    
    # === Transformer ===
    transformer_model.eval()
    with torch.no_grad():
        for t in range(1, len(tokens)):
            input_ids = torch.tensor([tokens[:t]])
            outputs = transformer_model(input_ids)
            
            probs = F.softmax(outputs["logits"][:, -1, :], dim=-1)
            entropy = -(probs * torch.log2(probs + 1e-9)).sum().item()
            results['transformer']['token_entropy'].append(entropy)
    
    # === SSM ===
    ssm_model.eval()
    with torch.no_grad():
        for t in range(1, len(tokens)):
            input_ids = torch.tensor([tokens[:t]])
            outputs = ssm_model(input_ids)
            
            # Token entropy
            probs = F.softmax(outputs["logits"][:, -1, :], dim=-1)
            token_entropy = -(probs * torch.log2(probs + 1e-9)).sum().item()
            results['ssm']['token_entropy'].append(token_entropy)
            
            # State entropy (NEW)
            state = outputs['final_state']
            state_probs = F.softmax(state, dim=-1)
            state_entropy = -(state_probs * torch.log2(state_probs + 1e-9)).sum().item()
            results['ssm']['state_entropy'].append(state_entropy)
    
    return results


def measure_state_to_answer_correlation(ssm_model, examples, tokenizer):
    """
    Measure how much the SSM state h_t "knows" about the final answer
    at each position in the sequence.
    
    Method: Project state onto answer vocabulary and measure confidence.
    
    Hypothesis: In PNA-SSM (Group D), the state should converge to
    high confidence on the correct answer BEFORE the model generates
    the Result: token.
    """
    # Create a simple linear probe: state -> answer logits
    # Train it on a held-out set, then measure on test set
    
    # This measures: "If we only had access to h_t at position t,
    # how accurately could we predict the final answer?"
    
    # Expected result: Group D's states become highly predictive
    # of the answer earlier in the sequence than Group C.
    
    pass  # Implementation details omitted for brevity


if __name__ == "__main__":
    # Test the SSM-adapted loss
    batch_size = 4
    seq_len = 32
    vocab_size = 50
    d_model = 512
    d_state = 16
    
    # Mock inputs
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    halt_confidence = torch.rand(batch_size, seq_len, 1)
    states_sequence = torch.randn(batch_size, seq_len, d_state)
    reasoning_mask = torch.ones(batch_size, seq_len)
    result_positions = torch.tensor([20, 22, 21, 23])
    
    # Instantiate loss
    loss_fn = SSMThermodynamicLoss(d_state=d_state)
    
    # Forward pass
    losses = loss_fn(
        logits, targets, halt_confidence, states_sequence,
        reasoning_mask, result_positions
    )
    
    print("Loss components:")
    for key, value in losses.items():
        print(f"  {key}: {value}")
    
    print(f"\nState leads token collapse: {losses['state_leads']}")
