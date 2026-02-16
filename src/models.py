"""
Model implementations for all 4 experimental groups.
Group A/B: Transformer (4 layers, 8 heads, d=512)
Group C/D: SSM Mamba-style (6 layers, d=512, d_state=16)
Both matched at ~7M parameters.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ============================================================
# SSM Model (Groups C & D)
# ============================================================

class S6Block(nn.Module):
    """Simplified Selective State Space block inspired by Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(
            in_channels=d_model, out_channels=d_model,
            kernel_size=d_conv, padding=d_conv - 1, groups=d_model
        )
        self.x_proj = nn.Linear(d_model, d_state * 2)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        Args: x: [batch, seq_len, d_model]
        Returns: (output [batch, seq_len, d_model], states [batch, seq_len, d_state])
        """
        batch, L, D = x.shape

        x_and_gate = self.in_proj(x)
        x_ssm, gate = x_and_gate.chunk(2, dim=-1)

        x_conv = rearrange(x_ssm, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        BC = self.x_proj(x_conv)
        B_mat, C_mat = BC.chunk(2, dim=-1)

        A = -torch.exp(self.A_log.float())
        A_discrete = torch.exp(A)

        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        states = []

        for t in range(L):
            h = A_discrete.unsqueeze(0) * h + x_conv[:, t, :].unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)
            y = torch.sum(C_mat[:, t, :].unsqueeze(1) * h, dim=-1)
            y = y + self.D * x_conv[:, t, :]
            outputs.append(y)
            states.append(h.mean(dim=1))  # [batch, d_state]

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        y = self.out_proj(y)

        all_states = torch.stack(states, dim=1)  # [batch, L, d_state]
        return y, all_states


class MambaBlock(nn.Module):
    """Full Mamba-style block with normalization and residual."""

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S6Block(d_model, d_state)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        ssm_out, states = self.ssm(x)
        return residual + ssm_out, states


class PNA_SSM(nn.Module):
    """SSM model for Groups C & D."""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 n_layers: int = 6, d_state: int = 16,
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, vocab_size)
        self.halt_head = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)

        all_states = None
        for layer in self.layers:
            h, all_states = layer(h)

        h = self.norm(h)
        logits = self.token_head(h)

        # Halt confidence from SSM states (last layer)
        halt_confidence = self.halt_head(all_states)  # [B, L, 1]

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
            "states_sequence": all_states,  # [B, L, d_state]
            "final_state": all_states[:, -1, :],  # [B, d_state]
        }


# ============================================================
# Transformer Model (Groups A & B)
# ============================================================

class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        L = x.size(1)
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, is_causal=True)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    """Transformer model for Groups A & B."""

    def __init__(self, vocab_size: int, d_model: int = 512,
                 n_layers: int = 4, n_heads: int = 8,
                 max_seq_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, vocab_size)
        # Transformer halt head uses token embeddings (no SSM state)
        self.halt_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.dropout(self.embedding(x) + self.pos_encoding(positions))

        for layer in self.layers:
            h = layer(h)

        h = self.norm(h)
        logits = self.token_head(h)
        halt_confidence = self.halt_head(h)

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
            "states_sequence": None,  # Transformers don't have SSM state
            "final_state": None,
        }


# ============================================================
# Helpers
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(group: str, vocab_size: int, max_seq_len: int = 256, device='cpu'):
    """
    Create model for a given experimental group.
    Groups A/B: Transformer, Groups C/D: SSM.
    """
    if group in ('A', 'B'):
        # d_model=320, n_layers=4, n_heads=8 gives ~5M params to match SSM
        model = TransformerModel(
            vocab_size=vocab_size, d_model=320, n_layers=4,
            n_heads=8, max_seq_len=max_seq_len
        )
    elif group in ('C', 'D'):
        model = PNA_SSM(
            vocab_size=vocab_size, d_model=512, n_layers=6,
            d_state=16, max_seq_len=max_seq_len
        )
    else:
        raise ValueError(f"Unknown group: {group}")

    return model.to(device)


if __name__ == "__main__":
    vocab_size = 25
    for group in ['A', 'B', 'C', 'D']:
        model = create_model(group, vocab_size)
        n_params = count_parameters(model)
        arch = "Transformer" if group in ('A', 'B') else "SSM"
        print(f"Group {group} ({arch}): {n_params:,} params")

        x = torch.randint(0, vocab_size, (2, 32))
        out = model(x)
        print(f"  logits: {out['logits'].shape}, halt: {out['halt_confidence'].shape}")
