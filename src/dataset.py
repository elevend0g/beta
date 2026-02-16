"""
Dataset generation for PNA-SSM experiment.
Symbolic parity (2-8 bits) + multi-step arithmetic.
"""

import random
import torch
from torch.utils.data import Dataset


# Token vocabulary for the reasoning tasks
VOCAB = {
    '<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<HALT>': 3,
    'Input:': 4, 'Result:': 5,
    '0': 6, '1': 7, '2': 8, '3': 9, '4': 10,
    '5': 11, '6': 12, '7': 13, '8': 14, '9': 15,
    '+': 16, '-': 17, '*': 18, '=': 19, '^': 20,
    '(': 21, ')': 22, ',': 23, ' ': 24,
}
# Reverse mapping
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def tokenize(text, vocab=VOCAB):
    """Convert text to token IDs. Unknown chars map to space."""
    tokens = []
    i = 0
    while i < len(text):
        # Try multi-char tokens first
        matched = False
        for length in [7, 6, 5]:  # 'Result:', 'Input:'
            if i + length <= len(text) and text[i:i+length] in vocab:
                tokens.append(vocab[text[i:i+length]])
                i += length
                matched = True
                break
        if not matched:
            ch = text[i]
            tokens.append(vocab.get(ch, vocab[' ']))
            i += 1
    return tokens


def detokenize(ids, id_to_token=ID_TO_TOKEN):
    """Convert token IDs back to text."""
    return ''.join(id_to_token.get(i, '?') for i in ids)


def generate_parity_example(n_bits, include_reasoning=True):
    """
    Generate a parity task example.
    Input: binary string, Output: XOR parity (0 or 1).
    Optionally includes step-by-step reasoning.
    """
    bits = [random.randint(0, 1) for _ in range(n_bits)]
    answer = 0
    for b in bits:
        answer ^= b

    bits_str = ''.join(str(b) for b in bits)

    if include_reasoning:
        # Build step-by-step XOR reasoning
        steps = []
        running = bits[0]
        for i in range(1, len(bits)):
            result = running ^ bits[i]
            steps.append(f"{running}^{bits[i]}={result}")
            running = result
        reasoning = ' '.join(steps)
        text = f"Input:{bits_str} {reasoning} Result:{answer}"
    else:
        text = f"Input:{bits_str} Result:{answer}"

    return text, answer


def generate_arithmetic_example():
    """
    Generate a simple arithmetic example (addition/subtraction of small numbers).
    """
    op = random.choice(['+', '-'])
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    if op == '+':
        answer = a + b
    else:
        answer = a - b

    text = f"Input:{a}{op}{b} {a}{op}{b}={answer} Result:{answer}"
    return text, answer


class ReasoningDataset(Dataset):
    """Dataset for parity + arithmetic reasoning tasks."""

    def __init__(self, n_examples, bit_range=(2, 8), max_seq_len=64,
                 include_arithmetic=True, seed=42):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.examples = []

        rng = random.Random(seed)
        old_state = random.getstate()
        random.seed(seed)

        for _ in range(n_examples):
            if include_arithmetic and rng.random() < 0.3:
                text, answer = generate_arithmetic_example()
            else:
                n_bits = rng.randint(bit_range[0], bit_range[1])
                text, answer = generate_parity_example(n_bits)

            tokens = [VOCAB['<BOS>']] + tokenize(text) + [VOCAB['<HALT>'], VOCAB['<EOS>']]

            # Find Result: token position
            result_pos = None
            for i, t in enumerate(tokens):
                if t == VOCAB['Result:']:
                    result_pos = i
                    break
            if result_pos is None:
                result_pos = len(tokens) - 3

            # Build reasoning mask (1 for reasoning tokens between Input: and Result:)
            reasoning_mask = [0] * len(tokens)
            input_pos = None
            for i, t in enumerate(tokens):
                if t == VOCAB['Input:']:
                    input_pos = i
                    break
            if input_pos is not None and result_pos is not None:
                for i in range(input_pos + 1, result_pos):
                    reasoning_mask[i] = 1

            self.examples.append({
                'tokens': tokens,
                'result_pos': result_pos,
                'reasoning_mask': reasoning_mask,
                'text': text,
            })

        random.setstate(old_state)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        tokens = ex['tokens']
        mask = ex['reasoning_mask']

        # Pad or truncate
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
            mask = mask[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(tokens)
            tokens = tokens + [VOCAB['<PAD>']] * pad_len
            mask = mask + [0] * pad_len

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        reasoning_mask = torch.tensor(mask[:-1], dtype=torch.float)
        result_pos = min(ex['result_pos'], self.max_seq_len - 2)

        return {
            'input_ids': input_ids,
            'targets': targets,
            'reasoning_mask': reasoning_mask,
            'result_pos': torch.tensor(result_pos, dtype=torch.long),
        }


def create_datasets(train_n=8000, val_n=1000, test_n=1000, bit_range=(2, 8),
                    max_seq_len=64):
    """Create train/val/test splits."""
    train = ReasoningDataset(train_n, bit_range=bit_range, max_seq_len=max_seq_len, seed=42)
    val = ReasoningDataset(val_n, bit_range=bit_range, max_seq_len=max_seq_len, seed=123)
    test = ReasoningDataset(test_n, bit_range=bit_range, max_seq_len=max_seq_len, seed=456)
    return train, val, test


if __name__ == "__main__":
    train, val, test = create_datasets()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Vocab size: {VOCAB_SIZE}")
    sample = train[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample text: {train.examples[0]['text']}")
