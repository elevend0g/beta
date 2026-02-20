"""
Phase 13b: Hard Gating — Fixing the Teacher-Student Disconnect

Phase 13 introduced architectural gates (OperationDetector + StateGate) that worked
perfectly under teacher forcing (85.2% op detection) but collapsed to 0.0% during
autoregressive generation — the "Wormhole Effect."

Root causes identified:
  1. MEASUREMENT BUG: op_probs were only captured for generated token positions,
     but operations live in the PROMPT. Position mapping was off → false 0%.
  2. DISTRIBUTION SHIFT: The Op Detector only trained on teacher-forced states.
     During generation, hidden states drift into regions invisible to the detector.
  3. SHORTCUT: The Halt Head found a path that bypasses the Op Detector's features
     while encoding enough signal for ~67% accuracy.

Phase 13b fixes:
  - Capture op_probs for ALL positions (including prompt) during generation
  - Scheduled Sampling: gradually replace ground-truth tokens with model predictions
    during training, forcing the backbone to produce decodable states under drift
  - State Decodability Loss: penalize high entropy in Op Detector output, forcing
    confident classifications at every operation position
  - Halt-Decodability Coupling: penalize halt confidence when Op Detector is
    uncertain, closing the "wormhole" shortcut
"""

import os
import sys
import re
import json
import time
import copy
import math
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from einops import rearrange

sns.set_theme(style="whitegrid", font_scale=1.1)

sys.path.insert(0, str(Path(__file__).parent))

from dataset import VOCAB, VOCAB_SIZE, ID_TO_TOKEN, tokenize, detokenize
from models import PNA_SSM, S6Block, MambaBlock, count_parameters
from losses import ThermodynamicLoss
from train import get_device, get_cosine_schedule, evaluate, compute_halt_f1
from compressible_task import (
    CompressibleArithmeticDataset, CompressibleGenerator,
    detect_oscillation, classify_convergence, analyze_convergence_vs_halt,
    compute_tier_metrics, TIER_NAMES, TIER_COLORS,
)
from rule_initialization import (
    RIMDatasetWrapper, GeodesicPurity, find_op_token_positions, classify_ops,
    OP_PAD, OP_REAL, OP_IDENTITY, OP_CANCEL_START, OP_CANCEL_END, OP_STAR_ZERO,
    OP_NAMES, MAX_OPS,
)

N_OP_TYPES = 6  # PAD, REAL, IDENTITY, CANCEL_START, CANCEL_END, STAR_ZERO


# ============================================================================
# Operation Detector
# ============================================================================

class OperationDetector(nn.Module):
    """
    Per-timestep classifier predicting operation type from hidden states.

    Input: hidden states [B, L, d_model] from intermediate layer
    Output: logits [B, L, N_OP_TYPES]
    """

    def __init__(self, d_model, n_op_types=N_OP_TYPES, d_hidden=128):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, n_op_types),
        )

    def forward(self, hidden):
        """Returns [B, L, n_op_types] logits."""
        return self.detector(hidden)


# ============================================================================
# State Gate
# ============================================================================

class StateGate(nn.Module):
    """
    Applies operation-dependent gating to SSM state updates.

    Gate behaviors (learned strengths, biased initialization):
      - IDENTITY: lerp toward h_old (preserve state)
      - CANCEL_END: lerp toward checkpoint (restore pre-cancel state)
      - STAR_ZERO: lerp toward collapse target (force low entropy)
      - PAD/REAL/CANCEL_START: no intervention
    """

    def __init__(self, d_model, d_state, n_op_types=N_OP_TYPES):
        super().__init__()
        self.d_state = d_state

        # Per-type gate strengths: sigmoid(logit)
        # PAD=0, REAL=1: -5 → ~0 (no gate)
        # IDENTITY=2, CANCEL_END=4, STAR_ZERO=5: +3 → ~0.95 (strong gate)
        # CANCEL_START=3: -5 → ~0 (no gate, just saves checkpoint)
        self.gate_logits = nn.Parameter(torch.tensor(
            [-5.0, -5.0, 3.0, -5.0, 3.0, 3.0]
        ))

        # Learned low-entropy collapse target for *0
        self.collapse_target = nn.Parameter(torch.zeros(d_state))

    def forward(self, h_new, h_old, op_probs, checkpoint_state):
        """
        Args:
            h_new: [B, d_model, d_state] — state AFTER SSM update
            h_old: [B, d_model, d_state] — state BEFORE SSM update
            op_probs: [B, n_op_types] — soft operation detection probabilities
            checkpoint_state: [B, d_model, d_state] — saved state for cancellation

        Returns:
            h_gated: [B, d_model, d_state]
        """
        gate_strengths = torch.sigmoid(self.gate_logits)  # [n_op_types]

        # Identity gate: preserve old state when identity op detected
        p_identity = op_probs[:, OP_IDENTITY]  # [B]
        g_identity = gate_strengths[OP_IDENTITY]
        identity_weight = (p_identity * g_identity).unsqueeze(-1).unsqueeze(-1)

        # Cancel gate: restore checkpoint when cancel_end detected
        p_cancel = op_probs[:, OP_CANCEL_END]  # [B]
        g_cancel = gate_strengths[OP_CANCEL_END]
        cancel_weight = (p_cancel * g_cancel).unsqueeze(-1).unsqueeze(-1)

        # Collapse gate: force toward collapse target when star_zero detected
        p_collapse = op_probs[:, OP_STAR_ZERO]  # [B]
        g_collapse = gate_strengths[OP_STAR_ZERO]
        collapse_weight = (p_collapse * g_collapse).unsqueeze(-1).unsqueeze(-1)

        # Apply gates (nearly mutually exclusive via softmax)
        h_gated = h_new

        # Identity: preserve old state
        h_gated = (1 - identity_weight) * h_gated + identity_weight * h_old

        # Cancel: restore checkpoint
        h_gated = (1 - cancel_weight) * h_gated + cancel_weight * checkpoint_state

        # Collapse: force toward collapse target
        collapse_expanded = self.collapse_target.unsqueeze(0).unsqueeze(0).expand_as(h_gated)
        h_gated = (1 - collapse_weight) * h_gated + collapse_weight * collapse_expanded

        return h_gated


# ============================================================================
# Gated S6Block
# ============================================================================

class S6BlockGated(S6Block):
    """
    S6Block with per-step state gating based on detected operations.
    Gate is applied AFTER each state update, modifying h before output.
    """

    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__(d_model, d_state, d_conv)
        self.state_gate = StateGate(d_model, d_state)

    def forward(self, x, op_probs=None):
        """
        Args:
            x: [B, L, d_model]
            op_probs: [B, L, n_op_types] or None
        Returns:
            (output [B, L, d_model], states [B, L, d_state])
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
        checkpoint = torch.zeros_like(h)
        outputs = []
        states = []

        for t in range(L):
            h_old = h.clone()

            # Standard SSM update
            h = A_discrete.unsqueeze(0) * h + x_conv[:, t, :].unsqueeze(-1) * B_mat[:, t, :].unsqueeze(1)

            # Apply gate if operation detection is available
            if op_probs is not None:
                op_t = op_probs[:, t, :]  # [B, n_op_types]

                # Soft checkpoint: save state when CANCEL_START detected
                p_cs = op_t[:, OP_CANCEL_START].unsqueeze(-1).unsqueeze(-1)  # [B,1,1]
                checkpoint = p_cs * h_old + (1 - p_cs) * checkpoint

                # Apply state gate
                h = self.state_gate(h, h_old, op_t, checkpoint)

            y = torch.sum(C_mat[:, t, :].unsqueeze(1) * h, dim=-1)
            y = y + self.D * x_conv[:, t, :]
            outputs.append(y)
            states.append(h.mean(dim=1))  # [batch, d_state]

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(gate)
        y = self.out_proj(y)

        all_states = torch.stack(states, dim=1)  # [batch, L, d_state]
        return y, all_states


class MambaBlockGated(nn.Module):
    """Mamba block with gated S6Block."""

    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S6BlockGated(d_model, d_state)

    def forward(self, x, op_probs=None):
        residual = x
        x = self.norm(x)
        ssm_out, states = self.ssm(x, op_probs=op_probs)
        return residual + ssm_out, states


# ============================================================================
# PNA_SSM_HardGated
# ============================================================================

class PNA_SSM_HardGated(PNA_SSM):
    """
    PNA_SSM with hard gating: learned operation detection + architectural state gates.

    Layers 0-(gate_after_layer-1): standard MambaBlock
    OperationDetector runs on hidden states after layer (gate_after_layer-1)
    Layers gate_after_layer to (n_layers-1): MambaBlockGated with state gates
    """

    def __init__(self, vocab_size, d_model=512, n_layers=6, d_state=16,
                 max_seq_len=256, gate_after_layer=3):
        super().__init__(vocab_size, d_model, n_layers, d_state, max_seq_len)

        self.gate_after_layer = gate_after_layer

        # Operation detector
        self.op_detector = OperationDetector(d_model, N_OP_TYPES, d_hidden=128)

        # Replace layers after gate_after_layer with gated versions
        for i in range(gate_after_layer, n_layers):
            self.layers[i] = MambaBlockGated(d_model, d_state)

    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_encoding(positions)

        all_states = None
        op_logits = None
        op_probs = None

        for i, layer in enumerate(self.layers):
            if i == self.gate_after_layer:
                # Run operation detector on current hidden states
                op_logits = self.op_detector(h)  # [B, L, N_OP_TYPES]
                op_probs = F.softmax(op_logits, dim=-1)

            if i >= self.gate_after_layer:
                h, all_states = layer(h, op_probs=op_probs)
            else:
                h, all_states = layer(h)

        h = self.norm(h)
        logits = self.token_head(h)
        halt_confidence = self.halt_head(all_states)

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
            "states_sequence": all_states,
            "final_state": all_states[:, -1, :],
            "op_logits": op_logits,
            "op_probs": op_probs,
        }


# ============================================================================
# Per-Token Operation Labels
# ============================================================================

def build_per_token_op_labels(op_types, op_before_pos, op_after_pos, n_ops, seq_len):
    """
    Convert per-operation metadata to per-token labels for op_detector supervision.

    For each operation, tokens from (before_pos+1) to after_pos (inclusive) get
    labeled with the operation type. All other positions get OP_PAD.

    Args:
        op_types: [B, MAX_OPS]
        op_before_pos: [B, MAX_OPS]
        op_after_pos: [B, MAX_OPS]
        n_ops: [B]
        seq_len: int

    Returns:
        labels: [B, seq_len] tensor of op type labels
    """
    B = op_types.size(0)
    device = op_types.device
    labels = torch.zeros(B, seq_len, dtype=torch.long, device=device)

    for b in range(B):
        n = n_ops[b].item()
        for i in range(min(n, MAX_OPS)):
            ot = op_types[b, i].item()
            if ot == OP_PAD:
                continue
            bp = op_before_pos[b, i].item()
            ap = op_after_pos[b, i].item()
            # Label the operator and operand token positions
            for pos in range(bp + 1, min(ap + 1, seq_len)):
                labels[b, pos] = ot

    return labels


# ============================================================================
# State Decodability Losses (NEW — Phase 13b)
# ============================================================================

def compute_decodability_loss(op_logits, op_labels):
    """
    State Decodability Loss: penalize high entropy in Op Detector predictions
    at operation positions, forcing the backbone to produce unambiguous states.

    L_decodability = E_{t in op_positions}[ H(softmax(op_logits_t)) ]

    Low entropy = detector is confident = state is decodable.
    """
    op_probs = F.softmax(op_logits, dim=-1)           # [B, L, N_OP_TYPES]
    log_probs = F.log_softmax(op_logits, dim=-1)       # [B, L, N_OP_TYPES]
    entropy = -(op_probs * log_probs).sum(dim=-1)       # [B, L]

    # Only penalize at operation positions (non-PAD)
    op_mask = (op_labels > 0).float()
    if op_mask.sum() > 0:
        return (entropy * op_mask).sum() / op_mask.sum(), entropy
    else:
        return torch.tensor(0.0, device=op_logits.device), entropy


def compute_halt_coupling_loss(halt_confidence, op_entropy):
    """
    Halt-Decodability Coupling: penalize halt confidence when the Op Detector
    is uncertain. Closes the "wormhole" shortcut where the model halts
    confidently while in an uninterpretable state.

    L_coupling = E_t[ halt_conf_t * normalized_entropy_t ]

    Gradients flow through BOTH halt_conf and entropy:
    - Halt Head learns: "don't halt when state is unreadable"
    - Backbone learns: "make states readable when you want to halt"
    """
    halt_conf = halt_confidence.squeeze(-1)             # [B, L]
    max_entropy = math.log(N_OP_TYPES)
    norm_entropy = op_entropy / max_entropy              # [0, 1]

    # Both terms receive gradients — bidirectional pressure
    return (halt_conf * norm_entropy).mean()


# ============================================================================
# Training (MODIFIED — Phase 13b: Scheduled Sampling + Decodability)
# ============================================================================

def train_hardgated_one_epoch(model, dataloader, loss_fn, optimizer,
                               scheduler, device, epoch=1, total_epochs=40,
                               op_loss_weight=0.5,
                               decodability_weight=0.3,
                               coupling_weight=0.2,
                               ss_max_prob=0.5,
                               ss_warmup=10):
    """
    Train HardGated model for one epoch with:
    - Standard CE + halt losses
    - Op Detector auxiliary loss (teacher-forced)
    - State Decodability entropy regularization
    - Halt-Decodability coupling
    - Scheduled Sampling: second forward pass with model's own predictions
      mixed into the input, closing the train/inference distribution gap
    """
    model.train()
    metrics_accum = defaultdict(float)
    n_batches = 0

    # Scheduled sampling probability: linearly increase over warmup
    # epoch 1 → 0.0, epoch ss_warmup+1 → ss_max_prob
    ss_prob = min(max(epoch - 1, 0) / max(ss_warmup, 1), 1.0) * ss_max_prob

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['targets'].to(device)
        reasoning_mask = batch['reasoning_mask'].to(device)
        result_pos = batch['result_pos'].to(device)
        op_types = batch['op_types'].to(device)
        op_before_pos = batch['op_before_pos'].to(device)
        op_after_pos = batch['op_after_pos'].to(device)
        n_ops = batch['n_ops'].to(device)

        B, L = input_ids.shape

        # ============================================================
        # Phase 1: Standard teacher-forced forward pass
        # ============================================================
        outputs = model(input_ids)

        # Standard thermodynamic loss (CE + halt)
        loss_dict = loss_fn(
            logits=outputs['logits'],
            targets=targets,
            halt_confidence=outputs['halt_confidence'],
            states_sequence=outputs.get('states_sequence'),
            reasoning_mask=reasoning_mask,
            result_token_positions=result_pos,
        )

        # Op detector auxiliary loss (teacher-forced)
        op_logits = outputs['op_logits']  # [B, L, N_OP_TYPES]
        op_labels = build_per_token_op_labels(
            op_types, op_before_pos, op_after_pos, n_ops, L
        )
        op_det_loss = F.cross_entropy(
            op_logits.reshape(-1, N_OP_TYPES), op_labels.reshape(-1),
            ignore_index=-100
        )

        # ============================================================
        # Phase 2: State Decodability Loss (NEW)
        # Penalize high entropy in Op Detector → force confident states
        # ============================================================
        decodability_loss, op_entropy = compute_decodability_loss(
            op_logits, op_labels
        )

        # ============================================================
        # Phase 3: Halt-Decodability Coupling (NEW)
        # Penalize halt confidence when Op Detector is uncertain
        # ============================================================
        coupling_loss = compute_halt_coupling_loss(
            outputs['halt_confidence'], op_entropy
        )

        # ============================================================
        # Phase 4: Scheduled Sampling (NEW)
        # Re-run with model's own predictions mixed in, compute Op
        # Detector loss on the "drifted" states to close the
        # teacher/inference distribution gap
        # ============================================================
        ss_loss = torch.tensor(0.0, device=device)
        ss_op_acc = 0.0

        if ss_prob > 0:
            with torch.no_grad():
                # pred_tokens[:, t] = model's prediction for position t+1
                pred_tokens = outputs['logits'].argmax(dim=-1)  # [B, L]

            # Create mixed input: at position t+1, sometimes use pred_tokens[:, t]
            ss_mask = (torch.rand(B, L - 1, device=device) < ss_prob)
            ss_mask[:, 0] = False  # Never replace the token right after BOS
            mixed_input = input_ids.clone()
            mixed_input[:, 1:] = torch.where(
                ss_mask,
                pred_tokens[:, :-1],   # model's prediction for each position
                input_ids[:, 1:]       # ground truth
            )

            # Forward pass with mixed (partially autoregressive) input
            outputs_ss = model(mixed_input)

            # Op Detector must still work on drifted states
            ss_op_logits = outputs_ss['op_logits']
            ss_op_det_loss = F.cross_entropy(
                ss_op_logits.reshape(-1, N_OP_TYPES),
                op_labels.reshape(-1),
                ignore_index=-100
            )

            # Decodability on drifted states
            ss_decodability, ss_entropy = compute_decodability_loss(
                ss_op_logits, op_labels
            )

            # Halt coupling on drifted states
            ss_coupling = compute_halt_coupling_loss(
                outputs_ss['halt_confidence'], ss_entropy
            )

            ss_loss = (ss_op_det_loss
                       + decodability_weight * ss_decodability
                       + coupling_weight * ss_coupling)

            # Track SS op detector accuracy
            with torch.no_grad():
                ss_preds = ss_op_logits.argmax(dim=-1)
                mask = op_labels > 0
                if mask.sum() > 0:
                    ss_op_acc = (ss_preds[mask] == op_labels[mask]).float().mean().item()

        # ============================================================
        # Total Loss
        # ============================================================
        total_loss = (loss_dict['total']
                      + op_loss_weight * op_det_loss
                      + decodability_weight * decodability_loss
                      + coupling_weight * coupling_loss
                      + op_loss_weight * ss_loss)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Op detector accuracy (teacher-forced)
        with torch.no_grad():
            op_preds = op_logits.argmax(dim=-1)  # [B, L]
            mask = op_labels > 0  # only score non-PAD positions
            if mask.sum() > 0:
                op_acc = (op_preds[mask] == op_labels[mask]).float().mean().item()
            else:
                op_acc = 0.0

        metrics_accum['total_loss'] += total_loss.item()
        metrics_accum['ce_loss'] += loss_dict['ce_loss']
        metrics_accum['halt_loss'] += loss_dict['halt_loss']
        metrics_accum['op_det_loss'] += op_det_loss.item()
        metrics_accum['op_det_acc'] += op_acc
        metrics_accum['decodability_loss'] += decodability_loss.item()
        metrics_accum['coupling_loss'] += coupling_loss.item()
        metrics_accum['ss_loss'] += ss_loss.item()
        metrics_accum['ss_op_acc'] += ss_op_acc
        metrics_accum['ss_prob'] += ss_prob
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in metrics_accum.items()}


def train_hardgated_model(train_ds, val_ds, device, results_dir,
                           epochs=40, batch_size=32, lr=1e-3, patience=5):
    """Train PNA_SSM_HardGated on compressible arithmetic."""
    print("\n" + "=" * 60)
    print("Training PNA_SSM_HardGated on Compressible Arithmetic")
    print("=" * 60)

    model = PNA_SSM_HardGated(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                               max_seq_len=64, gate_after_layer=3).to(device)
    n_params = count_parameters(model)
    print(f"  Architecture: PNA_SSM_HardGated, Params: {n_params:,}")
    print(f"  Loss: L_ce + 0.1*L_halt + 0.5*L_op + 0.3*L_decodability"
          f" + 0.2*L_coupling + 0.5*L_ss")

    # Wrap datasets
    rim_train = RIMDatasetWrapper(train_ds)
    rim_val = RIMDatasetWrapper(val_ds)
    print(f"  Data: {len(rim_train)} train, {len(rim_val)} val")

    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), weight_decay=0.01)
    train_loader = DataLoader(rim_train, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(rim_val, batch_size=batch_size, shuffle=False)

    total_steps = epochs * len(train_loader)
    scheduler = get_cosine_schedule(optimizer, total_steps)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_hardgated_one_epoch(
            model, train_loader, loss_fn, optimizer,
            scheduler, device,
            epoch=epoch, total_epochs=epochs,
            op_loss_weight=0.5,
            decodability_weight=0.3,
            coupling_weight=0.2,
            ss_max_prob=0.5,
            ss_warmup=10,
        )
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - t0

        val_loss = val_metrics.get('total_loss', val_metrics.get('ce_loss', 0))
        val_acc = val_metrics.get('accuracy', 0)

        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['total_loss'],
            'val_loss': val_loss,
            'val_acc': val_acc,
            'ce_loss': train_metrics['ce_loss'],
            'halt_loss': train_metrics['halt_loss'],
            'op_det_loss': train_metrics['op_det_loss'],
            'op_det_acc': train_metrics['op_det_acc'],
            'decodability_loss': train_metrics['decodability_loss'],
            'coupling_loss': train_metrics['coupling_loss'],
            'ss_loss': train_metrics['ss_loss'],
            'ss_op_acc': train_metrics['ss_op_acc'],
            'ss_prob': train_metrics['ss_prob'],
            'time': elapsed,
        })

        print(f"  Epoch {epoch:3d} | loss={train_metrics['total_loss']:.4f} "
              f"ce={train_metrics['ce_loss']:.4f} halt={train_metrics['halt_loss']:.4f} "
              f"op_det={train_metrics['op_det_loss']:.4f} op_acc={train_metrics['op_det_acc']:.1%} "
              f"decode={train_metrics['decodability_loss']:.4f} "
              f"ss_acc={train_metrics['ss_op_acc']:.1%} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.1%} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    ckpt_path = os.path.join(results_dir, 'hard_gating_model.pt')
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Saved checkpoint: {ckpt_path}")

    return model, history


# ============================================================================
# Hard Gated Generator (MODIFIED — Phase 13b: capture ALL op_probs)
# ============================================================================

class HardGatedGenerator(CompressibleGenerator):
    """
    Extends CompressibleGenerator to capture op_probs and gate activations.

    Phase 13b fix: captures op_probs for ALL positions (including prompt)
    on the first forward pass, so operations in the expression are measurable.
    """

    def generate(self, expression):
        """Generate with op detection tracking."""
        prompt_text = f"Input:{expression} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)

        halt_confidences = []
        state_entropies = []
        state_vectors = []
        op_probs_list = []  # FIXED: now aligned with generated_ids positions
        stop_reason = "max_length"

        self.model.eval()
        with torch.no_grad():
            for step in range(self.max_length):
                input_tensor = torch.tensor(
                    [generated_ids], dtype=torch.long, device=self.device
                )

                max_pos = getattr(self.model, 'max_seq_len', 256)
                if hasattr(self.model, 'pos_encoding'):
                    max_pos = self.model.pos_encoding.num_embeddings
                if input_tensor.size(1) > max_pos:
                    input_tensor = input_tensor[:, -max_pos:]

                outputs = self.model(input_tensor)
                logits = outputs["logits"][:, -1, :]

                # Halt confidence
                halt_conf = 0.0
                if outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                # State entropy
                if outputs.get("states_sequence") is not None:
                    last_state = outputs["states_sequence"][:, -1, :]
                    state_vec = last_state.squeeze(0).cpu()
                    state_vectors.append(state_vec)

                    energy = last_state ** 2
                    probs = energy / (energy.sum(dim=-1, keepdim=True) + 1e-9)
                    ent = -(probs * torch.log2(probs + 1e-9)).sum(dim=-1).item()
                    state_entropies.append(ent)
                else:
                    state_entropies.append(0.0)

                # --------------------------------------------------------
                # FIXED (Phase 13b): Capture op_probs for ALL positions
                # on the first step (prompt), then only the new position
                # on subsequent steps.
                #
                # After this loop: op_probs_list[i] = op_prob at position i
                # in the generated_ids sequence. This ensures operation
                # tokens in the PROMPT are measured by the Op Detector.
                # --------------------------------------------------------
                if outputs.get("op_probs") is not None:
                    if step == 0:
                        # First step: model processes entire prompt
                        # Capture op_probs for ALL prompt positions
                        all_op_probs = outputs["op_probs"][0].cpu()  # [L, N_OP_TYPES]
                        for t in range(all_op_probs.size(0)):
                            op_probs_list.append(all_op_probs[t])
                    else:
                        # Subsequent steps: only capture the new position
                        op_prob = outputs["op_probs"][:, -1, :].squeeze(0).cpu()
                        op_probs_list.append(op_prob)

                # Greedy decoding
                next_token = logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif halt_conf > self.halt_threshold:
                    stop_reason = "halt_confidence"
                    break

        text = detokenize(generated_ids)
        parsed_answer = self._parse_arithmetic_result(text)
        reasoning_tokens = self._count_reasoning_tokens(generated_ids)

        return {
            'generated_text': text,
            'generated_ids': generated_ids,
            'parsed_answer': parsed_answer,
            'reasoning_tokens': reasoning_tokens,
            'total_tokens': len(generated_ids),
            'halt_confidences': halt_confidences,
            'state_entropies': state_entropies,
            'state_vectors': state_vectors,
            'stop_reason': stop_reason,
            'op_probs': op_probs_list,
            'prompt_len': len(prompt_ids),
        }


# ============================================================================
# Generation + Evaluation (MODIFIED — Phase 13b: fixed position mapping)
# ============================================================================

def evaluate_op_detector_generation(gen_results, test_ds):
    """
    Evaluate op detector accuracy during autoregressive generation.

    Phase 13b fix: op_probs_list[i] now maps directly to position i in
    generated_ids (no prompt_len offset needed), since we capture all
    positions including the prompt.
    """
    true_labels = []
    pred_labels = []

    for i, r in enumerate(gen_results):
        if not r.get('op_probs'):
            continue

        expression = r['expression']
        tokens = r['generated_ids']
        op_probs = r['op_probs']

        # Get ground-truth op positions in the generated sequence
        op_info = find_op_token_positions(tokens, expression)

        for op_type, before_pos, after_pos in op_info:
            if op_type in (OP_PAD,):
                continue

            # Check each position in the op range
            for pos in range(before_pos + 1, after_pos + 1):
                # FIXED (Phase 13b): op_probs[pos] maps directly to position
                # pos in generated_ids — no offset needed
                if 0 <= pos < len(op_probs):
                    op_prob_at_pos = op_probs[pos]
                    # Handle both list (from .tolist()) and tensor
                    if isinstance(op_prob_at_pos, (list, np.ndarray)):
                        pred = int(np.argmax(op_prob_at_pos))
                    else:
                        pred = op_prob_at_pos.argmax().item()
                    true_labels.append(op_type)
                    pred_labels.append(pred)

    if not true_labels:
        return {
            'overall_accuracy': 0.0,
            'per_type_f1': {},
            'confusion_matrix': np.zeros((N_OP_TYPES, N_OP_TYPES)),
            'n_samples': 0,
        }

    true_arr = np.array(true_labels)
    pred_arr = np.array(pred_labels)

    overall_acc = (true_arr == pred_arr).mean()

    # Per-type F1
    per_type_f1 = {}
    for ot in range(N_OP_TYPES):
        if ot == OP_PAD:
            continue
        tp = ((true_arr == ot) & (pred_arr == ot)).sum()
        fp = ((true_arr != ot) & (pred_arr == ot)).sum()
        fn = ((true_arr == ot) & (pred_arr != ot)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_type_f1[OP_NAMES.get(ot, str(ot))] = {
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'n': int((true_arr == ot).sum()),
        }

    # Confusion matrix
    cm = np.zeros((N_OP_TYPES, N_OP_TYPES), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t < N_OP_TYPES and p < N_OP_TYPES:
            cm[t, p] += 1

    return {
        'overall_accuracy': float(overall_acc),
        'per_type_f1': per_type_f1,
        'confusion_matrix': cm,
        'n_samples': len(true_labels),
    }


def compute_gate_activation_stats(gen_results):
    """
    Analyze gate activation patterns during generation.

    Phase 13b fix: direct position mapping (no offset), float() for type safety.
    """
    gate_stats = defaultdict(list)

    for r in gen_results:
        if not r.get('op_probs'):
            continue

        expression = r['expression']
        tokens = r['generated_ids']
        op_probs = r['op_probs']

        op_info = find_op_token_positions(tokens, expression)

        # FIXED (Phase 13b): direct position mapping, no prompt_len offset
        # Build ground-truth per-position
        gt_at_pos = {}
        for op_type, before_pos, after_pos in op_info:
            for pos in range(before_pos + 1, after_pos + 1):
                if 0 <= pos < len(op_probs):
                    gt_at_pos[pos] = op_type

        for pos_idx, op_prob in enumerate(op_probs):
            gt = gt_at_pos.get(pos_idx, OP_PAD)
            # Gate activation strengths
            for op_idx, name in OP_NAMES.items():
                # FIXED: use float() to handle both tensor and plain float
                gate_stats[f'{name}_activation'].append(float(op_prob[op_idx]))
            gate_stats['gt_type'].append(gt)

    if not gate_stats.get('gt_type'):
        return {'correlations': {}, 'mean_activations': {}, 'mean_correlation': 0.0}

    # Compute correlation: when ground truth is IDENTITY, how strongly does identity gate fire?
    gt_types = np.array(gate_stats['gt_type'])
    correlations = {}
    for op_idx, name in OP_NAMES.items():
        if op_idx == OP_PAD:
            continue
        activations = np.array(gate_stats[f'{name}_activation'])
        is_this_type = (gt_types == op_idx).astype(float)
        if is_this_type.sum() > 0 and is_this_type.std() > 0 and activations.std() > 0:
            corr = np.corrcoef(activations, is_this_type)[0, 1]
            correlations[name] = float(corr)

    mean_activations = {}
    for name in OP_NAMES.values():
        vals = gate_stats.get(f'{name}_activation', [])
        if vals:
            mean_activations[name] = float(np.mean(vals))

    return {
        'correlations': correlations,
        'mean_activations': mean_activations,
        'mean_correlation': float(np.mean(list(correlations.values()))) if correlations else 0.0,
    }


def evaluate_purity_corrected(model, generated_ids, expression, device, purity_eval):
    """
    Correct purity evaluation: run one final forward pass on the complete
    generated sequence to get states aligned with token positions.

    During autoregressive generation, state_vectors[i] = state at generation
    step i, NOT at token position i. find_op_token_positions returns token
    positions. This mismatch caused Phase 12's false 16.6% purity reading.

    Fix: feed the complete generated sequence through the model in one pass.
    Now states_sequence[0, pos, :] = state at token position pos.
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)

        max_pos = getattr(model, 'max_seq_len', 256)
        if hasattr(model, 'pos_encoding'):
            max_pos = model.pos_encoding.num_embeddings
        if input_tensor.size(1) > max_pos:
            input_tensor = input_tensor[:, :max_pos]

        outputs = model(input_tensor)
        full_states = outputs['states_sequence'][0].cpu()  # [L, d_state]

    # Now indexing is correct: full_states[pos] = state at token pos
    position_states = [full_states[i] for i in range(full_states.size(0))]

    purity_result = purity_eval.evaluate_trajectory(
        position_states, expression, generated_ids[:full_states.size(0)]
    )

    # Also capture op_probs from the full forward pass (properly aligned)
    op_probs_full = None
    if outputs.get('op_probs') is not None:
        op_probs_full = outputs['op_probs'][0].cpu()  # [L, N_OP_TYPES]

    return purity_result, op_probs_full


def run_hardgated_generation(model, test_ds, device):
    """Run autoregressive generation with corrected purity evaluation."""
    generator = HardGatedGenerator(model, device=device)
    purity_eval = GeodesicPurity()

    results = []
    for i in tqdm(range(len(test_ds.examples)), desc="Generating"):
        ex = test_ds.examples[i]
        expression = ex['expression']
        ground_truth = ex['answer']
        tier = ex['tier']

        gen_result = generator.generate(expression)

        # Convergence analysis (uses generation-step states — correct for this)
        osc_result = detect_oscillation(gen_result['state_vectors'])
        convergence = classify_convergence(gen_result, osc_result)

        # CORRECTED purity: single forward pass on complete sequence
        # gives states aligned with token positions
        purity_result, op_probs_full = evaluate_purity_corrected(
            model, gen_result['generated_ids'], expression, device, purity_eval
        )

        is_correct = (gen_result['parsed_answer'] == ground_truth)

        # Constraint violation at halt
        constraint_at_halt = 0.0
        if gen_result['state_vectors']:
            last_state = gen_result['state_vectors'][-1]
            if len(gen_result['state_vectors']) > 1:
                prev_state = gen_result['state_vectors'][-2]
                constraint_at_halt = (last_state - prev_state).norm().item()

        # Use full-forward-pass op_probs (properly aligned) if available,
        # else fall back to incremental op_probs from generation
        if op_probs_full is not None:
            op_probs_out = [op_probs_full[t].tolist() for t in range(op_probs_full.size(0))]
        else:
            op_probs_out = [p.tolist() if hasattr(p, 'tolist') else p
                            for p in gen_result.get('op_probs', [])]

        results.append({
            'expression': expression,
            'tier': tier,
            'ground_truth': ground_truth,
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
            'generated_ids': gen_result['generated_ids'],
            'geodesic_purity': purity_result['purity'],
            'n_constraints': purity_result['n_constraints'],
            'n_satisfied': purity_result.get('n_satisfied', 0),
            'constraint_at_halt': constraint_at_halt,
            'prompt_len': gen_result.get('prompt_len', 0),
            'op_probs': op_probs_out,
        })

    return results


# ============================================================================
# Teacher-Forced Op Detector Evaluation
# ============================================================================

def evaluate_op_detector_teacher_forced(model, dataloader, device):
    """Evaluate op detector accuracy on teacher-forced data."""
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            op_types = batch['op_types'].to(device)
            op_before_pos = batch['op_before_pos'].to(device)
            op_after_pos = batch['op_after_pos'].to(device)
            n_ops = batch['n_ops'].to(device)

            outputs = model(input_ids)
            op_logits = outputs['op_logits']  # [B, L, N_OP_TYPES]

            op_labels = build_per_token_op_labels(
                op_types, op_before_pos, op_after_pos, n_ops, input_ids.size(1)
            )

            op_preds = op_logits.argmax(dim=-1)  # [B, L]
            mask = op_labels > 0

            if mask.sum() > 0:
                all_true.extend(op_labels[mask].cpu().tolist())
                all_pred.extend(op_preds[mask].cpu().tolist())

    if not all_true:
        return {'accuracy': 0.0, 'confusion_matrix': np.zeros((N_OP_TYPES, N_OP_TYPES))}

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    acc = (true_arr == pred_arr).mean()

    cm = np.zeros((N_OP_TYPES, N_OP_TYPES), dtype=int)
    for t, p in zip(all_true, all_pred):
        if t < N_OP_TYPES and p < N_OP_TYPES:
            cm[t, p] += 1

    # Per-type F1
    per_type_f1 = {}
    for ot in range(N_OP_TYPES):
        if ot == OP_PAD:
            continue
        tp = ((true_arr == ot) & (pred_arr == ot)).sum()
        fp = ((true_arr != ot) & (pred_arr == ot)).sum()
        fn = ((true_arr == ot) & (pred_arr != ot)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_type_f1[OP_NAMES.get(ot, str(ot))] = {
            'precision': float(prec), 'recall': float(rec),
            'f1': float(f1), 'n': int((true_arr == ot).sum()),
        }

    return {
        'accuracy': float(acc),
        'confusion_matrix': cm,
        'per_type_f1': per_type_f1,
        'n_samples': len(all_true),
    }


# ============================================================================
# Figures
# ============================================================================

def plot_op_detector_performance(tf_eval, gen_op_eval, history, fig_dir):
    """
    Fig 22: Op Detector Performance.
    Left: Confusion matrix (teacher-forced)
    Center: Op detector loss & accuracy over training (TF + SS)
    Right: Per-operation-type F1 (teacher-forced vs autoregressive)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: Confusion matrix ---
    cm = tf_eval['confusion_matrix']
    # Only show non-PAD classes
    active_types = [OP_REAL, OP_IDENTITY, OP_CANCEL_START, OP_CANCEL_END, OP_STAR_ZERO]
    active_names = [OP_NAMES[t] for t in active_types]
    cm_active = cm[np.ix_(active_types, active_types)]

    # Normalize rows
    row_sums = cm_active.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_active / row_sums, 0)

    sns.heatmap(cm_norm, ax=ax1, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=active_names, yticklabels=active_names,
                cbar_kws={'label': 'Rate'})
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Op Detector: Teacher-Forced\nConfusion Matrix")

    # --- Center: Training curves (TF acc + SS acc) ---
    if history:
        epochs = [h['epoch'] for h in history]
        op_accs = [h['op_det_acc'] * 100 for h in history]
        ss_accs = [h.get('ss_op_acc', 0) * 100 for h in history]
        decode_losses = [h.get('decodability_loss', 0) for h in history]

        ax2.plot(epochs, op_accs, 'b-', linewidth=2, label='Op Acc (TF)')
        ax2.plot(epochs, ss_accs, 'r-', linewidth=2, label='Op Acc (SS)')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Op Detector Accuracy (%)")
        ax2.legend(loc='center left', fontsize=9)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(epochs, decode_losses, 'g--', linewidth=1.5, alpha=0.7,
                      label='Decodability Loss')
        ax2_twin.set_ylabel("Decodability Loss", color='green')
        ax2_twin.tick_params(axis='y', labelcolor='green')
        ax2_twin.legend(loc='center right', fontsize=9)
    ax2.set_title("Op Detector: TF vs Scheduled Sampling")

    # --- Right: Per-type F1 comparison ---
    tf_f1 = tf_eval.get('per_type_f1', {})
    gen_f1 = gen_op_eval.get('per_type_f1', {})

    types_to_show = ['real', 'identity', 'cancel_start', 'cancel_end', 'star_zero']
    types_present = [t for t in types_to_show if t in tf_f1 or t in gen_f1]

    if types_present:
        x = np.arange(len(types_present))
        width = 0.35

        tf_vals = [tf_f1.get(t, {}).get('f1', 0) for t in types_present]
        gen_vals = [gen_f1.get(t, {}).get('f1', 0) for t in types_present]

        ax3.bar(x - width/2, tf_vals, width, color='#3498db', alpha=0.8,
                edgecolor='black', label='Teacher-Forced')
        ax3.bar(x + width/2, gen_vals, width, color='#e74c3c', alpha=0.8,
                edgecolor='black', label='Autoregressive')

        for i in range(len(types_present)):
            ax3.annotate(f'{tf_vals[i]:.2f}', (i - width/2, tf_vals[i] + 0.02),
                         ha='center', fontsize=8)
            ax3.annotate(f'{gen_vals[i]:.2f}', (i + width/2, gen_vals[i] + 0.02),
                         ha='center', fontsize=8)

        ax3.set_xticks(x)
        ax3.set_xticklabels(types_present, rotation=20, ha='right')
        ax3.set_ylabel("F1 Score")
        ax3.set_ylim(0, 1.15)
        ax3.legend(fontsize=9)
    ax3.set_title("Per-Type Detection F1")

    fig.suptitle("Phase 13b: Hard Gating — Operation Detector Performance",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig22_hard_gating_op_detector.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_gate_activations(gen_results, gate_stats, fig_dir):
    """
    Fig 23: Gate Activation Analysis.
    Left: Example Tier 2 trajectory with gate overlay
    Center: Geodesic purity comparison (Phase 11 vs 12 vs 13b)
    Right: Gate-operation correlation bar chart
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # --- Left: Example trajectory ---
    tier2_results = [r for r in gen_results if r['tier'] == 2 and r.get('op_probs')]
    if tier2_results:
        ex = tier2_results[0]
        op_probs = ex['op_probs']
        if op_probs:
            steps = list(range(len(op_probs)))
            gate_names = ['identity', 'cancel_end', 'star_zero']
            gate_indices = [OP_IDENTITY, OP_CANCEL_END, OP_STAR_ZERO]
            gate_colors = ['#2ecc71', '#3498db', '#e74c3c']

            for name, idx, color in zip(gate_names, gate_indices, gate_colors):
                activations = [float(op_probs[s][idx]) for s in range(len(op_probs))]
                ax1.plot(steps, activations, color=color, linewidth=1.5,
                         alpha=0.8, label=name)

            # Mark prompt/generated boundary
            prompt_len = ex.get('prompt_len', 0)
            if prompt_len > 0 and prompt_len < len(steps):
                ax1.axvline(x=prompt_len, color='gray', linestyle=':', alpha=0.5,
                            label='prompt|gen')

            # Also plot halt confidence
            if ex.get('halt_confidences'):
                # halt_confidences are indexed by generation step, offset from prompt
                gen_start = prompt_len
                halt_steps = list(range(gen_start, gen_start + len(ex['halt_confidences'])))
                ax1.plot(halt_steps[:len(steps) - gen_start],
                         ex['halt_confidences'][:len(steps) - gen_start],
                         'k--', linewidth=1, alpha=0.5, label='halt_conf')

            ax1.set_xlabel("Position in Sequence")
            ax1.set_ylabel("Gate Activation")
            ax1.set_title(f"Tier 2 Trajectory\n{ex['expression'][:30]}")
            ax1.legend(fontsize=7, loc='upper right')
            ax1.set_ylim(-0.05, 1.05)

    # --- Center: Geodesic purity comparison ---
    # Phase 13b data
    purity_by_tier = defaultdict(list)
    for r in gen_results:
        if r['n_constraints'] > 0:
            purity_by_tier[r['tier']].append(r['geodesic_purity'])

    # Phase 11 baseline
    baseline_purities = {1: 0.0, 2: 0.0}
    # Phase 12 RIM baseline
    rim_purities = {1: 0.0, 2: 0.346}

    tiers = [1, 2]
    x = np.arange(len(tiers))
    width = 0.25

    p11_vals = [baseline_purities.get(t, 0) for t in tiers]
    p12_vals = [rim_purities.get(t, 0) for t in tiers]
    p13_vals = [np.mean(purity_by_tier.get(t, [0])) for t in tiers]

    ax2.bar(x - width, p11_vals, width, color='#95a5a6', alpha=0.7,
            edgecolor='black', label='Phase 11')
    ax2.bar(x, p12_vals, width, color='#f39c12', alpha=0.7,
            edgecolor='black', label='Phase 12 (RIM)')
    ax2.bar(x + width, p13_vals, width, color='#2ecc71', alpha=0.8,
            edgecolor='black', label='Phase 13b (Hard)')

    for i in range(len(tiers)):
        ax2.annotate(f'{p13_vals[i]:.2f}', (i + width, p13_vals[i] + 0.02),
                     ha='center', fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Tier {t}' for t in tiers])
    ax2.set_ylabel("Mean Geodesic Purity")
    ax2.set_title("Geodesic Purity: 3-Way Comparison")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.1)

    # --- Right: Gate-operation correlations ---
    correlations = gate_stats.get('correlations', {})
    if correlations:
        names = list(correlations.keys())
        vals = [correlations[n] for n in names]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]
        ax3.barh(names, vals, color=colors, alpha=0.8, edgecolor='black')
        ax3.axvline(x=0, color='black', linewidth=0.5)
        ax3.set_xlabel("Correlation with Ground Truth")
        mean_corr = np.mean(vals)
        ax3.set_title(f"Gate-Op Correlation\n(mean={mean_corr:.3f})")
    else:
        ax3.text(0.5, 0.5, "No correlation data", ha='center', va='center',
                 transform=ax3.transAxes)
        ax3.set_title("Gate-Op Correlation")

    fig.suptitle("Phase 13b: Hard Gating — Gate Activation Analysis",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig23_hard_gating_activations.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_comparison(gen_results, history, fig_dir):
    """
    Fig 24: Summary Comparison (Phase 11 vs 12 vs 13b).
    Left: Accuracy by tier
    Center: Geodesic purity by tier
    Right: Training curves (including SS accuracy gap)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Phase 11 baselines (from TODO.md)
    p11_acc = {0: 0.398, 1: 0.418, 2: 0.652}
    p12_acc = {0: 0.426, 1: 0.456, 2: 0.664}

    # Phase 13b per-tier accuracy
    tier_metrics = compute_tier_metrics(gen_results)
    p13_acc = {t: tier_metrics[t]['accuracy'] for t in tier_metrics}

    tiers = [0, 1, 2]
    x = np.arange(len(tiers))
    width = 0.25

    # --- Left: Accuracy by tier ---
    ax1.bar(x - width, [p11_acc.get(t, 0) * 100 for t in tiers], width,
            color='#95a5a6', alpha=0.7, edgecolor='black', label='Phase 11')
    ax1.bar(x, [p12_acc.get(t, 0) * 100 for t in tiers], width,
            color='#f39c12', alpha=0.7, edgecolor='black', label='Phase 12 (RIM)')
    ax1.bar(x + width, [p13_acc.get(t, 0) * 100 for t in tiers], width,
            color='#2ecc71', alpha=0.8, edgecolor='black', label='Phase 13b (Hard)')

    for i, t in enumerate(tiers):
        v = p13_acc.get(t, 0) * 100
        ax1.annotate(f'{v:.1f}%', (i + width, v + 1), ha='center', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'T{t} ({TIER_NAMES[t]})' for t in tiers])
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Accuracy by Tier")
    ax1.legend(fontsize=9)

    # --- Center: Geodesic purity by tier ---
    p11_pur = {1: 0.0, 2: 0.0}
    p12_pur = {1: 0.0, 2: 0.346}

    purity_by_tier = defaultdict(list)
    for r in gen_results:
        if r['n_constraints'] > 0:
            purity_by_tier[r['tier']].append(r['geodesic_purity'])

    pur_tiers = [1, 2]
    x2 = np.arange(len(pur_tiers))

    ax2.bar(x2 - width, [p11_pur.get(t, 0) * 100 for t in pur_tiers], width,
            color='#95a5a6', alpha=0.7, edgecolor='black', label='Phase 11')
    ax2.bar(x2, [p12_pur.get(t, 0) * 100 for t in pur_tiers], width,
            color='#f39c12', alpha=0.7, edgecolor='black', label='Phase 12 (RIM)')
    p13_pur_vals = [np.mean(purity_by_tier.get(t, [0])) * 100 for t in pur_tiers]
    ax2.bar(x2 + width, p13_pur_vals, width,
            color='#2ecc71', alpha=0.8, edgecolor='black', label='Phase 13b (Hard)')

    for i in range(len(pur_tiers)):
        ax2.annotate(f'{p13_pur_vals[i]:.1f}%', (i + width, p13_pur_vals[i] + 1),
                     ha='center', fontsize=8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels([f'Tier {t}' for t in pur_tiers])
    ax2.set_ylabel("Geodesic Purity (%)")
    ax2.set_title("Geodesic Purity by Tier")
    ax2.legend(fontsize=9)

    # --- Right: Training curves (TF acc vs SS acc gap) ---
    if history:
        epochs = [h['epoch'] for h in history]
        tf_accs = [h['op_det_acc'] * 100 for h in history]
        ss_accs = [h.get('ss_op_acc', 0) * 100 for h in history]
        gaps = [tf - ss for tf, ss in zip(tf_accs, ss_accs)]

        ax3.plot(epochs, tf_accs, 'b-', linewidth=2, label='Op Acc (TF)')
        ax3.plot(epochs, ss_accs, 'r-', linewidth=2, label='Op Acc (SS)')
        ax3.fill_between(epochs, ss_accs, tf_accs, alpha=0.15, color='orange',
                         label='TF-SS Gap')
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Op Detector Accuracy (%)")
        ax3.legend(fontsize=9)
        ax3.set_title("Teacher-Student Gap\n(should shrink over training)")
    else:
        ax3.set_title("Training Curves")

    fig.suptitle("Phase 13b vs Phase 12 vs Phase 11: Hard Gating Summary",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(fig_dir, 'fig24_hard_gating_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 13b: Hard Gating")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-n', type=int, default=8000)
    parser.add_argument('--val-n', type=int, default=1000)
    parser.add_argument('--test-n', type=int, default=1000)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--fig-dir', type=str, default='figures')
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    print(f"\n{'=' * 60}")
    print("Phase 13b: Hard Gating Experiment (Decodability Fix)")
    print(f"{'=' * 60}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # ---- Create Datasets ----
    print("\n--- Creating Datasets ---")
    train_ds = CompressibleArithmeticDataset(
        num_samples=args.train_n, min_ops=3, max_ops=6,
        max_seq_len=64, seed=42
    )
    val_ds = CompressibleArithmeticDataset(
        num_samples=args.val_n, min_ops=3, max_ops=6,
        max_seq_len=64, seed=123
    )
    test_ds = CompressibleArithmeticDataset(
        num_samples=args.test_n, min_ops=3, max_ops=8,
        max_seq_len=64, seed=456
    )

    for name, ds in [('Train', train_ds), ('Val', val_ds), ('Test', test_ds)]:
        tier_counts = defaultdict(int)
        for ex in ds.examples:
            tier_counts[ex['tier']] += 1
        print(f"  {name}: {len(ds)} examples, tiers: {dict(sorted(tier_counts.items()))}")

    # ---- Train or Load Model ----
    ckpt_path = os.path.join(args.results_dir, 'hard_gating_model.pt')
    if args.skip_training and os.path.exists(ckpt_path):
        print(f"\n--- Loading pretrained model from {ckpt_path} ---")
        model = PNA_SSM_HardGated(VOCAB_SIZE, d_model=512, n_layers=6, d_state=16,
                                   max_seq_len=64, gate_after_layer=3).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        history = []
    else:
        model, history = train_hardgated_model(
            train_ds, val_ds, device, args.results_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience
        )

    # ---- Teacher-forced Evaluation ----
    print("\n--- Teacher-forced Evaluation ---")
    loss_fn = ThermodynamicLoss(alpha=0.0, beta=0.1, gamma=0.0,
                                pad_token_id=VOCAB['<PAD>'])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    halt_f1 = compute_halt_f1(model, test_loader, device)
    print(f"  Test accuracy: {test_metrics['accuracy']:.1%}")
    print(f"  Halt F1: {halt_f1:.4f}")

    # Op detector evaluation (teacher-forced)
    print("\n--- Op Detector Evaluation (Teacher-Forced) ---")
    rim_test = RIMDatasetWrapper(test_ds)
    rim_test_loader = DataLoader(rim_test, batch_size=args.batch_size, shuffle=False)
    tf_op_eval = evaluate_op_detector_teacher_forced(model, rim_test_loader, device)
    print(f"  Op detector accuracy: {tf_op_eval['accuracy']:.1%}")
    for name, metrics in tf_op_eval.get('per_type_f1', {}).items():
        print(f"    {name}: F1={metrics['f1']:.3f} (n={metrics['n']})")

    # ---- Autoregressive Generation ----
    print("\n--- Autoregressive Generation ---")
    gen_results = run_hardgated_generation(model, test_ds, device)

    # ---- Tier Metrics ----
    print("\n--- Per-Tier Metrics ---")
    tier_metrics = compute_tier_metrics(gen_results)
    for tier in sorted(tier_metrics.keys()):
        m = tier_metrics[tier]
        print(f"\n  Tier {tier} ({TIER_NAMES.get(tier, '?')}):")
        print(f"    N={m['n']}, Accuracy={m['accuracy']:.1%}")
        print(f"    Reasoning tokens: {m['mean_reasoning_tokens']:.1f} ± {m['std_reasoning_tokens']:.1f}")
        print(f"    Mean halt time: {m['mean_halt_time']:.1f} ± {m['std_halt_time']:.1f}")
        print(f"    Convergence: {dict(m['convergence'])}")

    # ---- Geodesic Purity ----
    print("\n--- Geodesic Purity ---")
    purity_by_tier = defaultdict(list)
    for r in gen_results:
        if r['n_constraints'] > 0:
            purity_by_tier[r['tier']].append(r['geodesic_purity'])

    for tier in sorted(purity_by_tier.keys()):
        vals = purity_by_tier[tier]
        print(f"  Tier {tier}: mean purity = {np.mean(vals):.3f} ± {np.std(vals):.3f} "
              f"(n={len(vals)})")

    all_purities = [r['geodesic_purity'] for r in gen_results if r['n_constraints'] > 0]
    overall_purity = np.mean(all_purities) if all_purities else 0
    print(f"  Overall: mean purity = {overall_purity:.3f}")

    # ---- Op Detector During Generation ----
    print("\n--- Op Detector Evaluation (Autoregressive) ---")
    gen_op_eval = evaluate_op_detector_generation(gen_results, test_ds)
    print(f"  Op detector accuracy (autoregressive): {gen_op_eval['overall_accuracy']:.1%}")
    print(f"  Samples evaluated: {gen_op_eval.get('n_samples', 0)}")
    for name, metrics in gen_op_eval.get('per_type_f1', {}).items():
        print(f"    {name}: F1={metrics['f1']:.3f} (n={metrics['n']})")

    # ---- Gate Activation Stats ----
    print("\n--- Gate Activation Analysis ---")
    gate_stats = compute_gate_activation_stats(gen_results)
    print(f"  Mean gate-op correlation: {gate_stats.get('mean_correlation', 0):.3f}")
    for name, corr in gate_stats.get('correlations', {}).items():
        print(f"    {name}: r={corr:.3f}")
    for name, act in gate_stats.get('mean_activations', {}).items():
        print(f"    {name} mean activation: {act:.4f}")

    # ---- Basin Analysis ----
    print("\n--- Convergence vs Halt Analysis ---")
    basin_metrics = analyze_convergence_vs_halt(gen_results)

    conv = basin_metrics['convergence']
    cyc = basin_metrics['cycling']
    print(f"\n  True Convergence (n={conv['n']}):")
    print(f"    Accuracy: {conv['accuracy']:.1%}")
    print(f"    Mean tokens: {conv['mean_tokens']:.1f}")

    print(f"\n  State Cycling (n={cyc['n']}):")
    print(f"    Accuracy: {cyc['accuracy']:.1%}")
    print(f"    Mean tokens: {cyc['mean_tokens']:.1f}")

    print(f"\n  Accuracy gap: {basin_metrics['accuracy_gap']:+.1%}")

    # ---- Teacher-Student Gap Diagnostic ----
    print("\n--- Teacher-Student Gap Diagnostic ---")
    tf_acc = tf_op_eval['accuracy']
    ar_acc = gen_op_eval['overall_accuracy']
    gap = tf_acc - ar_acc
    print(f"  Op Detector TF accuracy:  {tf_acc:.1%}")
    print(f"  Op Detector AR accuracy:  {ar_acc:.1%}")
    print(f"  Gap (TF - AR):            {gap:+.1%}")
    if gap > 0.3:
        print(f"  WARNING: Large TF-AR gap ({gap:.1%}). State decodability may need more training.")
    elif gap < 0.1:
        print(f"  OK: TF-AR gap is small ({gap:.1%}). Wormhole likely closed.")

    # ---- Figures ----
    print("\n--- Generating Figures ---")
    plot_op_detector_performance(tf_op_eval, gen_op_eval, history, args.fig_dir)
    plot_gate_activations(gen_results, gate_stats, args.fig_dir)
    plot_summary_comparison(gen_results, history, args.fig_dir)

    # ---- Save Results ----
    purity_summary = {}
    for tier in sorted(purity_by_tier.keys()):
        vals = purity_by_tier[tier]
        purity_summary[str(tier)] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals)),
            'n': len(vals),
        }

    output = {
        'teacher_forced': {
            'accuracy': test_metrics.get('accuracy', 0),
            'halt_f1': halt_f1,
        },
        'op_detector_teacher_forced': {
            'accuracy': tf_op_eval['accuracy'],
            'per_type_f1': tf_op_eval.get('per_type_f1', {}),
        },
        'op_detector_autoregressive': {
            'accuracy': gen_op_eval['overall_accuracy'],
            'per_type_f1': gen_op_eval.get('per_type_f1', {}),
            'n_samples': gen_op_eval.get('n_samples', 0),
        },
        'teacher_student_gap': {
            'tf_accuracy': tf_acc,
            'ar_accuracy': ar_acc,
            'gap': gap,
        },
        'gate_activation': {
            'correlations': gate_stats.get('correlations', {}),
            'mean_correlation': gate_stats.get('mean_correlation', 0),
            'mean_activations': gate_stats.get('mean_activations', {}),
        },
        'tier_metrics': {
            str(t): {
                'n': m['n'],
                'accuracy': m['accuracy'],
                'mean_reasoning_tokens': m['mean_reasoning_tokens'],
                'std_reasoning_tokens': m['std_reasoning_tokens'],
                'mean_halt_time': m['mean_halt_time'],
                'convergence': dict(m['convergence']),
            }
            for t, m in tier_metrics.items()
        },
        'geodesic_purity': purity_summary,
        'overall_purity': float(overall_purity),
        'basin_analysis': {
            'convergence': {
                'n': conv['n'],
                'accuracy': conv['accuracy'],
                'mean_tokens': conv['mean_tokens'],
            },
            'cycling': {
                'n': cyc['n'],
                'accuracy': cyc['accuracy'],
                'mean_tokens': cyc['mean_tokens'],
            },
            'accuracy_gap': basin_metrics['accuracy_gap'],
        },
        'training_epochs': len(history),
        'history': history,
    }

    results_path = os.path.join(args.results_dir, 'hard_gating_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved results: {results_path}")

    # ---- Success Criteria ----
    print("\n" + "=" * 60)
    print("Success Criteria (Phase 13b)")
    print("=" * 60)

    criteria = [
        ("Geodesic Purity (overall)", overall_purity, 0.50, ">50%"),
        ("Op Detector Acc (autoregressive)", gen_op_eval['overall_accuracy'], 0.70, ">70%"),
        ("Tier 2 Accuracy", tier_metrics.get(2, {}).get('accuracy', 0), 0.70, ">70%"),
        ("True Convergence Rate", conv['n'] / max(conv['n'] + cyc['n'], 1), 0.15, ">15%"),
        ("Gate-Op Correlation", gate_stats.get('mean_correlation', 0), 0.50, ">0.5"),
        ("TF-AR Gap < 15%", 1.0 - gap, 0.85, "gap<15%"),
    ]

    n_pass = 0
    for name, value, threshold, target in criteria:
        status = "PASS" if value >= threshold else "FAIL"
        if status == "PASS":
            n_pass += 1
        print(f"  {name}: {value:.3f} (target {target}) — {status}")

    print(f"\n  Score: {n_pass}/{len(criteria)} criteria passed")


if __name__ == '__main__':
    main()