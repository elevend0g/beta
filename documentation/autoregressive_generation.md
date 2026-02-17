# Autoregressive Generation Experiment Design

## Core Question

**Does thermodynamic training produce shorter reasoning chains when models generate freely (not teacher-forced)?**

---

## Experimental Protocol

### 1. Generation Setup

**Starting Point**: Give each model only the input prompt:
```
Input:1101 
```

**Generation Parameters**:
- **Sampling strategy**: Greedy decoding (argmax) for deterministic comparison
- **Max length**: 200 tokens (generous — teacher-forced traces are ~30 tokens for 8-bit)
- **Stop conditions**: 
  1. Model emits `<HALT>` token
  2. Model emits `<EOS>` (end of sequence) 
  3. Reaches max length
  4. Halt confidence head spikes above 0.95 (for groups B/D only)

**Key Decision**: Use the halt confidence head as an *auxiliary* stop signal. If halt_conf > 0.95, stop generation even if the model hasn't emitted `<HALT>` token. This tests if the model learned halt detection but couldn't express it properly in generation.

### 2. Test Set Design

**Size**: 500 examples
- 400 in-distribution (2-8 bits, matching training)
- 100 out-of-distribution (9-10 bits, testing generalization)

**Why 500**: Large enough for statistical power, small enough to run in <1 hour per model.

### 3. Metrics

For each generated sequence, extract:

```python
@dataclass
class GenerationResult:
    # What the model generated
    input_bits: str                    # "1101"
    generated_text: str                # Full output
    generated_tokens: List[int]        # Token IDs
    
    # Parsing
    parsed_answer: Optional[int]       # Extracted final answer (0 or 1)
    is_valid_syntax: bool              # Did it follow the format?
    
    # Efficiency metrics
    reasoning_token_count: int         # Tokens between Input: and Result:
    total_token_count: int             # Full sequence length
    halt_position: Optional[int]       # Where <HALT> appeared (-1 if never)
    halt_confidence_trajectory: List[float]  # Per-token halt_conf values
    
    # Correctness
    is_correct: bool                   # Answer matches ground truth
    ground_truth: int                  # Expected parity
    
    # Stop reason
    stop_reason: str                   # "halt_token", "halt_confidence", "max_length", "eos"
```

**Primary Comparisons**:

| Metric | What It Tests | Success Criterion |
|--------|---------------|-------------------|
| **Accuracy** | Does free generation maintain correctness? | All groups ≥95% |
| **Mean Reasoning Length** | Token efficiency | Group D < Groups A/B/C by ≥20% |
| **Valid Syntax Rate** | Generation quality | ≥90% for thermodynamic groups |
| **Halt Placement Precision** | Calibration | Group D stops within 2 tokens of correct position |
| **Token Variance** | Adaptive reasoning | Group D shows high variance (short for easy, long for hard) |

### 4. Implementation

```python
# src/generate.py

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class GenerationConfig:
    max_length: int = 200
    halt_confidence_threshold: float = 0.95
    use_halt_head: bool = True  # False for groups A/C (no halt training)
    temperature: float = 0.0    # Greedy decoding
    
class FreeGenerator:
    """Autoregressive generation for PNA evaluation."""
    
    def __init__(self, model, tokenizer, config: GenerationConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.halt_id = tokenizer.token_to_id("<HALT>")
        self.eos_id = tokenizer.token_to_id("<EOS>")
    
    def generate(self, input_bits: str) -> GenerationResult:
        """
        Generate reasoning chain autoregressively.
        
        Returns full trace with metrics.
        """
        # Start with input prompt
        prompt = f"Input:{input_bits} "
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.model.device)
        
        generated_ids = input_ids[0].tolist()
        halt_confidences = []
        
        for step in range(self.config.max_length):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs["logits"][:, -1, :]  # [1, vocab_size]
                
                # Halt confidence (if model has halt head)
                if self.config.use_halt_head and "halt_confidence" in outputs:
                    halt_conf = outputs["halt_confidence"][:, -1, 0].item()
                    halt_confidences.append(halt_conf)
                else:
                    halt_conf = 0.0
                    halt_confidences.append(0.0)
            
            # Sample next token (greedy for now)
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
            elif halt_conf > self.config.halt_confidence_threshold:
                stop_reason = "halt_confidence"
                break
            
            # Append for next iteration
            input_ids = torch.tensor([generated_ids]).to(self.model.device)
        else:
            stop_reason = "max_length"
        
        # Decode and parse
        generated_text = self.tokenizer.decode(generated_ids)
        parsed = self._parse_output(generated_text, input_bits)
        
        return GenerationResult(
            input_bits=input_bits,
            generated_text=generated_text,
            generated_tokens=generated_ids,
            parsed_answer=parsed["answer"],
            is_valid_syntax=parsed["valid"],
            reasoning_token_count=self._count_reasoning_tokens(generated_ids),
            total_token_count=len(generated_ids),
            halt_position=self._find_halt_position(generated_ids),
            halt_confidence_trajectory=halt_confidences,
            is_correct=parsed["answer"] == self._compute_ground_truth(input_bits),
            ground_truth=self._compute_ground_truth(input_bits),
            stop_reason=stop_reason
        )
    
    def _parse_output(self, text: str, input_bits: str) -> dict:
        """
        Extract the final answer from free-form generation.
        
        Handles various formats:
        - "Result:1<HALT>"  (ideal)
        - "...therefore the answer is 1"  (natural language)
        - "1^1=0 0^0=0 0^1=1"  (missing Result: label)
        """
        import re
        
        # Try structured format first
        result_match = re.search(r'Result:([01])', text)
        if result_match:
            return {"answer": int(result_match.group(1)), "valid": True}
        
        # Try finding last digit after reasoning
        # Look for pattern like "= 0" or "is 1" after the input
        reasoning_section = text.split(input_bits)[-1]
        digit_matches = re.findall(r'[=is]\s*([01])', reasoning_section)
        if digit_matches:
            return {"answer": int(digit_matches[-1]), "valid": False}
        
        # Fallback: last digit in the string
        all_digits = re.findall(r'([01])', reasoning_section)
        if all_digits:
            return {"answer": int(all_digits[-1]), "valid": False}
        
        return {"answer": None, "valid": False}
    
    def _count_reasoning_tokens(self, token_ids: List[int]) -> int:
        """
        Count tokens in the reasoning region.
        
        Reasoning region = everything after "Input:XXXX " until "Result:" or <HALT>
        """
        # Find start: first token after input bits
        text = self.tokenizer.decode(token_ids)
        if "Input:" not in text:
            return 0
        
        input_end = text.index("Input:") + len("Input:") + text[text.index("Input:"):].index(" ") + 1
        
        # Find end: "Result:" or <HALT>
        if "Result:" in text[input_end:]:
            reasoning_end = input_end + text[input_end:].index("Result:")
        elif "<HALT>" in text[input_end:]:
            reasoning_end = input_end + text[input_end:].index("<HALT>")
        else:
            reasoning_end = len(text)
        
        reasoning_text = text[input_end:reasoning_end]
        return len(self.tokenizer.encode(reasoning_text))
    
    def _find_halt_position(self, token_ids: List[int]) -> Optional[int]:
        """Return index of <HALT> token, or None if not present."""
        try:
            return token_ids.index(self.halt_id)
        except ValueError:
            return None
    
    def _compute_ground_truth(self, input_bits: str) -> int:
        """Compute correct parity."""
        return sum(int(b) for b in input_bits) % 2
```

### 5. Evaluation Script

```python
# src/eval_generation.py

from generate import FreeGenerator, GenerationConfig
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

def evaluate_generation(model_path: str, test_set: List[str], 
                       config: GenerationConfig) -> dict:
    """
    Run free generation evaluation on a test set.
    
    Returns comprehensive metrics for publication.
    """
    model, tokenizer = load_model_and_tokenizer(model_path)
    generator = FreeGenerator(model, tokenizer, config)
    
    results = []
    for input_bits in test_set:
        result = generator.generate(input_bits)
        results.append(result)
    
    # Aggregate metrics
    metrics = {
        "accuracy": np.mean([r.is_correct for r in results]),
        "valid_syntax_rate": np.mean([r.is_valid_syntax for r in results]),
        
        # Token efficiency
        "mean_reasoning_tokens": np.mean([r.reasoning_token_count for r in results]),
        "median_reasoning_tokens": np.median([r.reasoning_token_count for r in results]),
        "std_reasoning_tokens": np.std([r.reasoning_token_count for r in results]),
        
        # Halt behavior
        "halt_token_rate": np.mean([r.halt_position is not None for r in results]),
        "mean_halt_position": np.mean([r.halt_position for r in results if r.halt_position]),
        
        # Stop reasons
        "stop_reasons": {
            reason: sum(1 for r in results if r.stop_reason == reason)
            for reason in ["halt_token", "halt_confidence", "eos", "max_length"]
        },
        
        # Stratified by length
        "by_input_length": defaultdict(dict)
    }
    
    # Stratify by input length (2-8 bits vs 9-10 bits)
    for length in range(2, 11):
        subset = [r for r in results if len(r.input_bits) == length]
        if subset:
            metrics["by_input_length"][length] = {
                "accuracy": np.mean([r.is_correct for r in subset]),
                "mean_tokens": np.mean([r.reasoning_token_count for r in subset]),
                "halt_rate": np.mean([r.halt_position is not None for r in subset])
            }
    
    return metrics, results

def compare_groups(results_by_group: dict) -> dict:
    """
    Statistical comparison between groups.
    
    Tests:
    1. Token efficiency: Is Group D significantly shorter than A/B/C?
    2. Accuracy parity: No significant difference in correctness
    3. Halt calibration: Group D halts closer to optimal position
    """
    from scipy import stats
    
    # Extract reasoning token counts
    tokens_A = [r.reasoning_token_count for r in results_by_group["A"]]
    tokens_B = [r.reasoning_token_count for r in results_by_group["B"]]
    tokens_C = [r.reasoning_token_count for r in results_by_group["C"]]
    tokens_D = [r.reasoning_token_count for r in results_by_group["D"]]
    
    # T-tests
    comparison = {
        "D_vs_A": {
            "mean_reduction": (np.mean(tokens_A) - np.mean(tokens_D)) / np.mean(tokens_A),
            "t_statistic": stats.ttest_ind(tokens_D, tokens_A),
            "effect_size": (np.mean(tokens_A) - np.mean(tokens_D)) / np.std(tokens_A)
        },
        "D_vs_B": {
            "mean_reduction": (np.mean(tokens_B) - np.mean(tokens_D)) / np.mean(tokens_B),
            "t_statistic": stats.ttest_ind(tokens_D, tokens_B),
        },
        "D_vs_C": {
            "mean_reduction": (np.mean(tokens_C) - np.mean(tokens_D)) / np.mean(tokens_C),
            "t_statistic": stats.ttest_ind(tokens_D, tokens_C),
        }
    }
    
    return comparison
```

### 6. What We're Looking For

**Scenario 1: Strong Confirmation (Best Case)**
- Group D generates **25-40% fewer tokens** than A/B/C
- Accuracy remains ≥95% across all groups
- Halt placement in D is within 2 tokens of optimal
- Effect size (Cohen's d) > 0.5, p < 0.01

**Interpretation**: Thermodynamic training + SSMs produces genuine reasoning compression. The model learned to take shortcuts.

---

**Scenario 2: Moderate Confirmation**
- Group D generates **10-20% fewer tokens**
- Halt calibration is much better (D stops at right place, A/B/C overshoot)
- High variance in D's token counts (adaptive reasoning: short for easy, long for hard)

**Interpretation**: Thermodynamic training doesn't dramatically compress reasoning, but it produces better-calibrated models that know when to stop.

---

**Scenario 3: Null Result (What You Currently Have)**
- Token counts are statistically identical across groups
- Halt precision is still better in D (99% F1)
- Accuracy identical

**Interpretation**: The task's deterministic structure resists compression even in free generation. Thermodynamic training improves *halt learning* but not *token efficiency* for this domain. **This is still publishable** — it validates the boundary condition and demonstrates that halt learning is a distinct phenomenon.

---

**Scenario 4: Surprising Finding**
- Groups B and D both generate fewer tokens than A and C
- No SSM-specific advantage (C and D similar)

**Interpretation**: Thermodynamic training matters more than architecture choice for this task. This would suggest the halt signal is the primary mechanism, not state compression.

### 7. Timeline

**Day 1-2**: Implement `generate.py` and `eval_generation.py`
**Day 3**: Run all 4 groups on 500-example test set (~2 hours compute)
**Day 4**: Analyze results, run statistical tests, generate comparison plots
**Day 5**: Write up findings, update figures

### 8. Expected Outputs

**Quantitative**:
- Table: Mean reasoning tokens (± std) per group
- Statistical significance tests (t-tests, effect sizes)
- Accuracy by input length (generalization curve)
- Halt placement error distribution

**Qualitative**:
- 20 example traces (5 per group) showing free generation
- Highlighting: Where do models differ in their reasoning paths?

**Figures**:
- **fig7_generation_comparison.png**: Box plots of reasoning length by group
- **fig8_halt_placement.png**: Histogram of halt position error
- **fig9_adaptive_reasoning.png**: Token count vs problem difficulty (scatter)

---

## Key Insight: What If Tokens Are Identical?

If free generation produces identical token counts across groups, **you still have a major result**: Thermodynamic training teaches models to *detect reasoning completion* without changing *how they reason*. This is valuable for:

1. **Confidence calibration**: Halt confidence is a learned uncertainty signal
2. **Cost control**: Stop generation when halt_conf > threshold (saves tokens)
3. **Theoretical insight**: Measurement-as-collapse detection is a learnable skill distinct from reasoning efficiency

