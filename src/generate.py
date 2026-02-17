"""
Autoregressive generation for PNA-SSM evaluation.
Tests whether thermodynamic training produces shorter reasoning chains
when models generate freely (not teacher-forced).
"""

import sys
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from dataset import VOCAB, ID_TO_TOKEN, VOCAB_SIZE, tokenize, detokenize


@dataclass
class GenerationResult:
    input_bits: str
    generated_text: str
    generated_tokens: List[int]

    # Parsing
    parsed_answer: Optional[int]
    is_valid_syntax: bool

    # Efficiency metrics
    reasoning_token_count: int
    total_token_count: int
    halt_position: Optional[int]
    halt_confidence_trajectory: List[float] = field(default_factory=list)

    # Correctness
    is_correct: bool = False
    ground_truth: int = 0

    # Stop reason
    stop_reason: str = "max_length"


@dataclass
class GenerationConfig:
    max_length: int = 200
    halt_confidence_threshold: float = 0.95
    use_halt_head: bool = True  # False for groups A/C (no halt training)
    temperature: float = 0.0    # Greedy decoding


class FreeGenerator:
    """Autoregressive generation for PNA evaluation."""

    def __init__(self, model, config: GenerationConfig, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.halt_id = VOCAB['<HALT>']
        self.eos_id = VOCAB['<EOS>']
        self.pad_id = VOCAB['<PAD>']

    def generate(self, input_bits: str) -> GenerationResult:
        """Generate reasoning chain autoregressively."""
        # Build prompt: <BOS>Input:XXXX
        prompt_text = f"Input:{input_bits} "
        prompt_ids = [VOCAB['<BOS>']] + tokenize(prompt_text)
        generated_ids = list(prompt_ids)
        halt_confidences = []
        stop_reason = "max_length"

        self.model.eval()
        with torch.no_grad():
            for step in range(self.config.max_length):
                input_tensor = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

                # Clamp to max_seq_len if needed
                max_pos = self.model.pos_encoding.num_embeddings if hasattr(self.model, 'pos_encoding') else 256
                if input_tensor.size(1) > max_pos:
                    input_tensor = input_tensor[:, -max_pos:]

                outputs = self.model(input_tensor)
                logits = outputs["logits"][:, -1, :]  # [1, vocab_size]

                # Halt confidence
                halt_conf = 0.0
                if self.config.use_halt_head and outputs.get("halt_confidence") is not None:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                halt_confidences.append(halt_conf)

                # Greedy or temperature sampling
                if self.config.temperature == 0.0:
                    next_token = logits.argmax(dim=-1).item()
                else:
                    probs = F.softmax(logits / self.config.temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()

                generated_ids.append(next_token)

                # Check stop conditions
                if next_token == self.halt_id:
                    stop_reason = "halt_token"
                    break
                elif next_token == self.eos_id:
                    stop_reason = "eos"
                    break
                elif self.config.use_halt_head and halt_conf > self.config.halt_confidence_threshold:
                    stop_reason = "halt_confidence"
                    break

        # Decode and parse
        generated_text = detokenize(generated_ids)
        ground_truth = compute_parity(input_bits)
        parsed = parse_output(generated_text, input_bits)

        # Count reasoning tokens
        reasoning_count = count_reasoning_tokens(generated_ids)

        # Find halt position
        halt_pos = None
        if self.halt_id in generated_ids:
            halt_pos = generated_ids.index(self.halt_id)

        return GenerationResult(
            input_bits=input_bits,
            generated_text=generated_text,
            generated_tokens=generated_ids,
            parsed_answer=parsed["answer"],
            is_valid_syntax=parsed["valid"],
            reasoning_token_count=reasoning_count,
            total_token_count=len(generated_ids),
            halt_position=halt_pos,
            halt_confidence_trajectory=halt_confidences,
            is_correct=(parsed["answer"] == ground_truth) if parsed["answer"] is not None else False,
            ground_truth=ground_truth,
            stop_reason=stop_reason,
        )


def compute_parity(bits_str: str) -> int:
    """Compute XOR parity of a binary string."""
    return sum(int(b) for b in bits_str if b in '01') % 2


def parse_output(text: str, input_bits: str) -> dict:
    """
    Extract the final answer from free-form generation.
    Handles: "Result:1<HALT>", step-by-step XOR, or last digit fallback.
    """
    # Try structured format first: Result:0 or Result:1
    result_match = re.search(r'Result:([01])', text)
    if result_match:
        return {"answer": int(result_match.group(1)), "valid": True}

    # Try finding last digit after reasoning
    input_marker = f"Input:{input_bits}"
    if input_marker in text:
        reasoning_section = text.split(input_marker, 1)[-1]
    else:
        reasoning_section = text

    # Look for "=0" or "=1" patterns from XOR steps
    digit_matches = re.findall(r'=([01])', reasoning_section)
    if digit_matches:
        return {"answer": int(digit_matches[-1]), "valid": False}

    # Fallback: last 0 or 1 in the string
    all_digits = re.findall(r'([01])', reasoning_section)
    if all_digits:
        return {"answer": int(all_digits[-1]), "valid": False}

    return {"answer": None, "valid": False}


def count_reasoning_tokens(token_ids: List[int]) -> int:
    """
    Count tokens in the reasoning region.
    Reasoning = everything after Input:XXXX (space token) until Result: or <HALT> or end.
    """
    input_tok = VOCAB['Input:']
    result_tok = VOCAB['Result:']
    halt_tok = VOCAB['<HALT>']
    space_tok = VOCAB[' ']

    # Find end of input section (Input: + digits + space)
    start = None
    for i, t in enumerate(token_ids):
        if t == input_tok:
            # Skip past input bits and the space after them
            j = i + 1
            while j < len(token_ids) and token_ids[j] in (VOCAB.get('0', -1), VOCAB.get('1', -1)):
                j += 1
            if j < len(token_ids) and token_ids[j] == space_tok:
                j += 1
            start = j
            break

    if start is None:
        return 0

    # Find end of reasoning
    end = len(token_ids)
    for i in range(start, len(token_ids)):
        if token_ids[i] in (result_tok, halt_tok):
            end = i
            break

    return end - start


def load_model(group: str, checkpoint_path: str, device='cpu'):
    """Load a trained model from checkpoint."""
    from models import create_model
    model = create_model(group, VOCAB_SIZE, max_seq_len=256, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_test_inputs(n_in_dist=400, n_ood=100, seed=2024):
    """
    Generate test set for autoregressive evaluation.
    400 in-distribution (2-8 bits) + 100 out-of-distribution (9-10 bits).
    """
    import random
    rng = random.Random(seed)
    inputs = []

    # In-distribution: 2-8 bits
    for _ in range(n_in_dist):
        n_bits = rng.randint(2, 8)
        bits = ''.join(str(rng.randint(0, 1)) for _ in range(n_bits))
        inputs.append(bits)

    # Out-of-distribution: 9-10 bits
    for _ in range(n_ood):
        n_bits = rng.choice([9, 10])
        bits = ''.join(str(rng.randint(0, 1)) for _ in range(n_bits))
        inputs.append(bits)

    return inputs


if __name__ == "__main__":
    # Quick smoke test with a single example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='D')
    parser.add_argument('--checkpoint', default='results/group_D_model.pt')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    model = load_model(args.group, args.checkpoint, args.device)
    use_halt = args.group in ('B', 'D')
    config = GenerationConfig(use_halt_head=use_halt)
    gen = FreeGenerator(model, config, device=args.device)

    test_bits = ["1101", "10110", "11", "1010101"]
    for bits in test_bits:
        result = gen.generate(bits)
        gt = compute_parity(bits)
        print(f"Input: {bits} | GT: {gt} | Parsed: {result.parsed_answer} | "
              f"Correct: {result.is_correct} | Valid: {result.is_valid_syntax} | "
              f"Reasoning tokens: {result.reasoning_token_count} | "
              f"Stop: {result.stop_reason}")
        print(f"  Generated: {result.generated_text[:120]}")
        print()
