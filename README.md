# Evaluating and Enhancing the Faithfulness of Large Reasoning Models to System Prompts under Adversarial Prompt Injection

## Overview

This repository contains implementation code for evaluating Large Reasoning Models (LRMs) under adversarial prompt injection attacks and training defense mechanisms. The code evaluates 4 models across 4 benchmarks using 3 types of reasoning-based distractors, and implements defense training using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO).

## Project Structure

```
project_code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── mmlu_redux/                        # MMLU-Redux evaluation
├── MATH/                              # MATH-500 evaluation
├── instruction_following/             # IFEval evaluation
├── LRM_as_a_judge/                    # JudgeLM evaluation
└── LLaMA-Factory/                     # Defense training framework
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA 12.8 (or compatible)
- PyTorch 2.0+
- vLLM for inference

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import vllm, torch, transformers; print('Ready!')"
```

## Running Evaluations

### General Command Structure

All evaluation scripts follow this pattern:

```bash
python <script_name> \
    --model <model_name> \
    --injection_type <distractor_type> \
    --injection_position <position> \
    --temperature <temp> \
    --tensor_parallel_size <num_gpus>
```

### Parameters

| Parameter                | Options                                    | Description                                    |
| ------------------------ | ------------------------------------------ | ---------------------------------------------- |
| `--model`                | Model path or HuggingFace ID               | Model to evaluate                              |
| `--injection_type`       | `math_aime`, `code`, `zebra_logic`, `none` | Distractor type (`none` for baseline)          |
| `--injection_position`   | `start`, `middle`, `end`                   | Injection position                             |
| `--temperature`          | Float (0.0-1.0)                            | Sampling temperature (0.0 for reproducibility) |
| `--tensor_parallel_size` | Integer                                    | Number of GPUs for inference                   |

### Distractor Types

- **`math_aime`**: AIME 2025 competition-level math problems
- **`code`**: LiveCodeBench competitive programming challenges
- **`zebra_logic`**: ZebraLogic multi-step deduction puzzles
- **`none`**: No distractor (clean baseline)

### Supported Models

- `Qwen/Qwen3-4B`
- `Qwen/Qwen3-8B`
- `microsoft/phi-4-reasoning-mini`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

## Benchmark-Specific Instructions

### 1. MMLU-Redux

3,000 multiple-choice questions across 57 subjects.

```bash
cd mmlu_redux

# Clean baseline
python eval_mmlu_redux_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type none \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With AIME distractor
python eval_mmlu_redux_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type math_aime \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With Code distractor
python eval_mmlu_redux_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type code \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With Logic distractor
python eval_mmlu_redux_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type zebra_logic \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8
```

### 2. MATH-500

500 competition-level mathematical word problems.

```bash
cd MATH/eval

# Clean baseline
python eval_math500_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type none \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With distractor
python eval_math500_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type math_aime \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8
```

### 3. IFEval (Instruction Following)

541 prompts with verifiable constraints.

```bash
cd instruction_following

# Clean baseline
python infer_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type none \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With distractor
python infer_offline.py \
    --model Qwen/Qwen3-8B \
    --injection_type zebra_logic \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8

# Evaluate results
python evaluation_main.py \
    --input_data outputs/qwen3_8b_clean.jsonl \
    --input_response_data outputs/qwen3_8b_clean.jsonl
```

### 4. JudgeLM (LRM-as-a-Judge)

1,000 pairwise comparison tasks.

```bash
cd LRM_as_a_judge

# Clean baseline
python eval_judgelm.py \
    --model Qwen/Qwen3-8B \
    --injection_type none \
    --temperature 0.0 \
    --tensor_parallel_size 8

# With distractor
python eval_judgelm.py \
    --model Qwen/Qwen3-8B \
    --injection_type code \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8
```

## Defense Training

### Quick Start

```bash
cd LLaMA-Factory

# Edit train.sh to select your training strategy
bash train.sh
```

### Training Strategies

Three defense approaches are available:

| Strategy               | Description                                 |
| ---------------------- | ------------------------------------------- |
| **SFT-only**           | Supervised fine-tuning on faithful examples |
| **DPO-only**           | Direct preference optimization              |
| **Sequential SFT+DPO** | SFT followed by DPO (recommended)           |

### Training Data

- **SFT Dataset**: `data/sft_mixed_openthought_12000_distract_1200.json`
  - 13,200 examples (12K clean + 1.2K adversarial)
- **DPO Dataset**: `data/distract_dpo.json`
  - Preference pairs (faithful vs. distracted responses)
s

After training, run evaluations using benchmark scripts:

```bash
cd ../mmlu_redux
python eval_mmlu_redux_offline.py \
    --model output_model_path/qwen_3_8b_lora_sft_dpo \
    --injection_type math_aime \
    --injection_position end \
    --temperature 0.0 \
    --tensor_parallel_size 8
```

## Output Format

Evaluation scripts output `.jsonl` files with each line containing:

```json
{
  "question": "Original question",
  "prompt": "Full prompt with injection",
  "response": "Model response",
  "prediction": "Extracted answer",
  "ground_truth": "Correct answer",
  "correct": true/false,
  "distractor_type": "math_aime",
  "position": "end"
}
```

## Computing Metrics

Calculate accuracy from output files:

```bash
python -c "
import json
correct = sum(1 for line in open('outputs/file.jsonl')
              if json.loads(line).get('correct', False))
total = sum(1 for _ in open('outputs/file.jsonl'))
print(f'Accuracy: {correct/total*100:.2f}% ({correct}/{total})')
"
```

## Troubleshooting

### Out of Memory

```bash
--tensor_parallel_size 1  # Use fewer GPUs
--max_tokens 16384        # Reduce generation length
```


## Notes

- All code uses open-source models only (no API required)
- Use `temperature=0.0` for reproducible results
- Default injection position is `end`
- Training requires significant compute

## Contact

**Zhehao Zhang**: zhang.16420@osu.edu
