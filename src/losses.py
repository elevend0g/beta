"""
Loss functions for all 4 experimental groups.
Groups A/C: Standard Cross-Entropy
Groups B/D: Thermodynamic Loss (L_th = L_ce + α·L_energy + β·L_halt + γ·L_state)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for Groups A & C."""

    def __init__(self, pad_token_id: int = 0):
        super().__init__()
        self.pad_token_id = pad_token_id

    def forward(self, logits, targets, halt_confidence=None,
                states_sequence=None, reasoning_mask=None,
                result_token_positions=None):
        B, T, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1),
            ignore_index=self.pad_token_id, reduction='mean'
        )
        return {
            "total": ce_loss,
            "ce_loss": ce_loss.item(),
            "energy_loss": 0.0,
            "halt_loss": 0.0,
            "state_loss": 0.0,
            "token_delta_h": 0.0,
            "state_delta_h": 0.0,
            "state_leads": False,
            "token_stagnation_rate": 0.0,
            "state_stagnation_rate": 0.0,
        }


class ThermodynamicLoss(nn.Module):
    """
    Thermodynamic loss for Groups B & D.
    L_th = L_ce + α·L_energy + β·L_halt + γ·L_state
    γ·L_state is only active for SSM groups (when states_sequence is provided).
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.1,
                 gamma: float = 0.05, pad_token_id: int = 0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.pad_token_id = pad_token_id

    def compute_token_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)
        return entropy

    def compute_state_entropy(self, state):
        state_probs = F.softmax(state, dim=-1)
        entropy = -(state_probs * torch.log2(state_probs + 1e-9)).sum(dim=-1)
        return entropy

    def forward(self, logits, targets, halt_confidence,
                states_sequence, reasoning_mask, result_token_positions):
        B, T, V = logits.shape

        # Component 1: Cross-Entropy
        ce_loss = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1),
            ignore_index=self.pad_token_id, reduction='mean'
        )

        # Component 2: Token Energy Penalty
        with torch.no_grad():
            token_entropy = self.compute_token_entropy(logits)
            token_delta_h = torch.zeros_like(token_entropy)
            token_delta_h[:, 1:] = F.relu(token_entropy[:, :-1] - token_entropy[:, 1:])
            token_stagnation = (token_delta_h < 0.01).float() * reasoning_mask

        energy_cost = reasoning_mask.float() * (1.0 + 2.0 * token_stagnation)
        energy_loss = energy_cost.mean()

        # Component 3: Halt Reward/Penalty
        halt_conf = halt_confidence.squeeze(-1)
        halt_target = torch.zeros_like(halt_conf)
        for b in range(B):
            pos = result_token_positions[b].item()
            if pos < T:
                halt_target[b, max(0, pos - 2):pos] = 0.5
                halt_target[b, pos:] = 1.0

        halt_loss = F.binary_cross_entropy(halt_conf, halt_target, reduction='mean')

        # Component 4: State Entropy Penalty (SSM-specific)
        state_loss_val = 0.0
        state_collapse_rate = 0.0
        state_stag_rate = 0.0
        state_leads = False

        if states_sequence is not None:
            with torch.no_grad():
                state_entropy = self.compute_state_entropy(states_sequence)
                state_delta_h = torch.zeros_like(state_entropy)
                state_delta_h[:, 1:] = F.relu(state_entropy[:, :-1] - state_entropy[:, 1:])
                state_stagnation = (state_delta_h < 0.01).float() * reasoning_mask

            state_entropy_penalty = (state_entropy * reasoning_mask).mean()
            state_stagnation_penalty = state_stagnation.mean()
            state_loss = state_entropy_penalty + state_stagnation_penalty
            state_loss_val = state_loss.item()

            with torch.no_grad():
                if reasoning_mask.any():
                    state_collapse_rate = state_delta_h[reasoning_mask.bool()].mean().item()
                state_stag_rate = state_stagnation.mean().item()
        else:
            state_loss = torch.tensor(0.0, device=logits.device)

        # Total
        total = (ce_loss +
                 self.alpha * energy_loss +
                 self.beta * halt_loss +
                 self.gamma * state_loss)

        with torch.no_grad():
            token_collapse_rate = 0.0
            if reasoning_mask.any():
                token_collapse_rate = token_delta_h[reasoning_mask.bool()].mean().item()
            state_leads = state_collapse_rate > token_collapse_rate

        return {
            "total": total,
            "ce_loss": ce_loss.item(),
            "energy_loss": energy_loss.item(),
            "halt_loss": halt_loss.item(),
            "state_loss": state_loss_val,
            "token_delta_h": token_collapse_rate,
            "state_delta_h": state_collapse_rate,
            "state_leads": state_leads,
            "token_stagnation_rate": token_stagnation.mean().item(),
            "state_stagnation_rate": state_stag_rate,
        }


class AdaptiveGovernor:
    """Adjusts α and γ during training based on stagnation rates."""

    def __init__(self, initial_alpha=0.05, initial_gamma=0.05,
                 min_alpha=0.01, max_alpha=0.5,
                 min_gamma=0.01, max_gamma=0.3,
                 sensitivity=1.15, window_size=100):
        self.alpha = initial_alpha
        self.gamma = initial_gamma
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.sensitivity = sensitivity
        self.window_size = window_size
        self.token_stagnation_history = []
        self.state_stagnation_history = []

    def step(self, metrics):
        self.token_stagnation_history.append(metrics["token_stagnation_rate"])
        self.state_stagnation_history.append(metrics["state_stagnation_rate"])

        if len(self.token_stagnation_history) < self.window_size:
            return self.alpha, self.gamma

        recent_token = sum(self.token_stagnation_history[-self.window_size:]) / self.window_size
        recent_state = sum(self.state_stagnation_history[-self.window_size:]) / self.window_size

        if recent_token > 0.1:
            self.alpha = min(self.alpha * self.sensitivity, self.max_alpha)
        else:
            self.alpha = max(self.alpha * 0.98, self.min_alpha)

        if recent_state > 0.1:
            self.gamma = min(self.gamma * self.sensitivity, self.max_gamma)
        else:
            self.gamma = max(self.gamma * 0.98, self.min_gamma)

        return self.alpha, self.gamma


def create_loss_fn(group: str, pad_token_id: int = 0):
    """Create loss function for a given experimental group."""
    if group in ('A', 'C'):
        return CrossEntropyLoss(pad_token_id=pad_token_id), None
    elif group in ('B', 'D'):
        loss_fn = ThermodynamicLoss(
            alpha=0.05, beta=0.1, gamma=0.05, pad_token_id=pad_token_id
        )
        governor = AdaptiveGovernor()
        return loss_fn, governor
    else:
        raise ValueError(f"Unknown group: {group}")
