"""
PNA-SSM: Mamba-style State Space Model with Thermodynamic Loss
Experiment Design - Architecture Specification

This implements a simplified Mamba-style SSM matched to the 7M parameter
Transformer baseline for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class S6Block(nn.Module):
    """
    Simplified Selective State Space block inspired by Mamba.
    
    Key properties:
    - Fixed-size state h_t (d_state dimensional)
    - Selection mechanism (data-dependent state transitions)
    - Linear-time complexity O(L) instead of O(L²)
    
    This is the core where thermodynamic loss should shine:
    the state h_t IS the Markovian summary.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(d_model, d_model * 2)
        
        # Convolution (local context, like Mamba's conv1d)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model
        )
        
        # Selection mechanism: input-dependent state transitions
        self.x_proj = nn.Linear(d_model, d_state * 2)  # Projects to B, C
        
        # State transition parameters (learnable, but modulated by input)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
            final_state: [batch, d_state] - for halt prediction
        """
        batch, L, D = x.shape

        # Input-dependent gating (like Mamba's "selective" mechanism)
        x_and_gate = self.in_proj(x)  # [batch, L, 2*D]
        x_ssm, gate = x_and_gate.chunk(2, dim=-1)

        # Apply convolution (in channel-first format)
        x_conv = rearrange(x_ssm, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :L]  # Trim padding
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = F.silu(x_conv)

        # Selection: compute input-dependent B, C matrices
        BC = self.x_proj(x_conv)  # [batch, L, 2*d_state]
        B_mat, C_mat = BC.chunk(2, dim=-1)  # Each [batch, L, d_state]

        # SSM: The core state-space computation
        # This is where the "Markovian compression" happens
        A = -torch.exp(self.A_log.float())  # [D, d_state]

        # Discretize (simplified - actual Mamba uses more sophisticated discretization)
        A_discrete = torch.exp(A)  # [D, d_state]

        # Scan through sequence (this is the O(L) recurrent computation)
        # State is per-channel: [batch, d_model, d_state]
        h = torch.zeros(batch, self.d_model, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        states = []  # Track state evolution for entropy measurement

        for t in range(L):
            # State transition: h_t = A * h_{t-1} + B_t * x_t
            # A_discrete: [d_model, d_state], h: [batch, d_model, d_state]
            # B_mat[:, t, :]: [batch, d_state], x_conv[:, t, :]: [batch, d_model]
            h = A_discrete.unsqueeze(0) * h + x_conv[:, t, :].unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)

            # Output: y_t = sum_n(C_t[n] * h_t[d, n])
            # C_mat[:, t, :]: [batch, d_state], h: [batch, d_model, d_state]
            y = torch.sum(C_mat[:, t, :].unsqueeze(1) * h, dim=-1)  # [batch, d_model]

            # Skip connection (the D parameter)
            y = y + self.D * x_conv[:, t, :]

            outputs.append(y)
            # Store mean state across channels for entropy tracking: [batch, d_state]
            states.append(h.mean(dim=1).clone())

        # Stack outputs
        y = torch.stack(outputs, dim=1)  # [batch, L, d_model]

        # Apply gate
        y = y * F.silu(gate)

        # Output projection
        y = self.out_proj(y)

        # Return mean state across channels for halt head: [batch, d_state]
        final_state = h.mean(dim=1)
        return y, final_state


class MambaBlock(nn.Module):
    """
    Full Mamba-style block with normalization and residual.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S6Block(d_model, d_state)
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            output: [batch, seq_len, d_model]
            final_state: [batch, d_state]
        """
        residual = x
        x = self.norm(x)
        ssm_out, final_state = self.ssm(x)
        return residual + ssm_out, final_state


class PNA_SSM(nn.Module):
    """
    Complete PNA-SSM model for thermodynamic loss experiments.
    
    Matched to ~7M parameters (same as Transformer baseline):
    - 6 layers (vs 4 in Transformer - SSM layers are cheaper)
    - d_model = 512
    - d_state = 16 (small state for Markovian compression)
    
    Key difference from Transformer:
    - No quadratic attention
    - Fixed-size recurrent state per layer
    - State transitions are input-dependent (selective)
    """
    def __init__(self, vocab_size: int, d_model: int = 512, 
                 n_layers: int = 6, d_state: int = 16,
                 max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        
        # Output heads
        self.norm = nn.LayerNorm(d_model)
        
        # Head 1: Next token prediction
        self.token_head = nn.Linear(d_model, vocab_size)
        
        # Head 2: Halt confidence
        # Key insight: This uses the FINAL STATE from the last SSM layer
        # The state h_t should naturally encode "how certain are we about the answer"
        self.halt_head = nn.Sequential(
            nn.Linear(d_state, 64),  # State is only d_state dimensional
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [batch, seq_len] - token IDs
        
        Returns:
            dict with:
            - logits: [batch, seq_len, vocab_size]
            - halt_confidence: [batch, seq_len, 1]
            - state_entropy: [batch, seq_len] - NEW: track state uncertainty
        """
        B, L = x.shape
        
        # Embed
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)
        
        # Process through SSM layers
        state_entropies = []
        for layer in self.layers:
            h, final_state = layer(h)
            
            # Measure state entropy at each position
            # This is unique to SSMs - we can track the state's "information content"
            # For the halt head: use final_state from last layer
        
        h = self.norm(h)
        
        # Token prediction head
        logits = self.token_head(h)
        
        # Halt prediction head
        # Re-run last layer to get states at each position (for training)
        # In actual inference, only need final state
        halt_confidences = []
        for t in range(L):
            # Get state at position t from last layer
            # (This is a simplification - in practice you'd track states during forward)
            # For now, use the sequence output as a proxy
            state_proxy = h[:, t, :self.d_state]  # First d_state dims
            halt_conf = self.halt_head(state_proxy)
            halt_confidences.append(halt_conf)
        
        halt_confidence = torch.stack(halt_confidences, dim=1)  # [B, L, 1]
        
        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
            "final_state": final_state  # For analysis
        }


class StateEntropyTracker:
    """
    NEW: Measure the entropy of the SSM state itself.
    
    This is a unique capability of SSMs that Transformers don't have:
    we can measure how much "uncertainty" is encoded in the state h_t.
    
    If the PNA-SSM hypothesis is correct, thermodynamic loss should
    cause the state entropy to collapse in a step-function pattern,
    even more cleanly than the token entropy does.
    """
    def __init__(self, d_state: int):
        self.d_state = d_state
    
    def compute_state_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Approximate entropy of the state vector.
        
        One approach: treat state as a distribution after softmax.
        Another: measure variance (higher variance = more uncertainty).
        
        Args:
            state: [batch, d_state]
        
        Returns:
            entropy: [batch] - scalar entropy per example
        """
        # Normalize state to probability distribution
        probs = F.softmax(state, dim=-1)
        
        # Shannon entropy
        entropy = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1)
        
        return entropy


def count_parameters(model):
    """Verify parameter count matches Transformer baseline."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test instantiation
    vocab_size = 50  # Char-level for parity task
    model = PNA_SSM(vocab_size)
    
    print(f"Total parameters: {count_parameters(model):,}")
    # Should be ~7M to match Transformer baseline
    
    # Test forward pass
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    outputs = model(x)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Halt confidence shape: {outputs['halt_confidence'].shape}")
    print(f"Final state shape: {outputs['final_state'].shape}")
