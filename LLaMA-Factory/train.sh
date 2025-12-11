#!/bin/bash

################################################################################
# Defense Training Script for Course Project
#
# This script implements adversarial fine-tuning defense experiments
# as described in the course project report. Due to computational constraints,
# we focused defense experiments on Qwen-3-4B and Qwen-3-8B models.
#
# Training Strategies Evaluated (Report Section 3.2.2):
#   1. SFT-only: Supervised fine-tuning on faithful behavior examples
#   2. DPO-only: Direct preference optimization with base model as reference
#   3. Sequential SFT+DPO: Combined approach (best performing)
#
# Key Findings from Report (Section 4.2):
#   - SFT-only: Recovered 49.7-59.9 percentage points (essential for defense)
#   - DPO-only: Minimal improvement (-0.15 to +3.3 points)
#   - Sequential SFT+DPO: Best results with 59.2-59.9 point improvements
#
# Model Focus: Qwen-3-4B and Qwen-3-8B (computational constraints)
# Training Config: LoRA (r=64, α=128), 4-bit quantization (nf4), bf16
# Datasets:
#   - SFT: sft_mixed_openthought_12000_distract_1200.json (13.2K examples)
#   - DPO: distract_dpo.json (preference pairs)
#
# Hardware: Single A100 40GB GPU (EC2 p4d.24xlarge instance)
################################################################################

# Stop the script if any command fails
set -e

# --- Configuration Files ---
# Configuration files for defense training experiments
# Uncomment the training strategy you want to run

CONFIG_FILES=(
    # ========== Qwen-3-4B Defense Training ==========
    # Strategy 1: SFT-only (57.6-59.9 point improvement over attacked base)
    #"examples/train_distractor/qwen3_4b_lora_sft.yaml"

    # Strategy 2: DPO-only (minimal improvement: -0.15 to +3.3 points)
    #"examples/train_distractor/qwen_3_4b_lora_dpo.yaml"

    # Strategy 3: Sequential SFT+DPO (BEST: 59.2-59.9 point improvement)
    #   - MMLU-Redux: 62.6% (+59.2 over attacked, 73.4% recovery)
    #   - MATH-500: 76.4% (+49.7 over attacked)
    #   - IFEval: 44.5% (+4.5 over attacked, 10% recovery)
    #   - JudgeLM: 61.8% (+59.9 over attacked, 86.4% recovery)
    "examples/train_distractor/qwen_3_4b_lora_sft_dpo.yaml"

    # ========== Qwen-3-8B Defense Training ==========
    # Most vulnerable model in baseline (3.45% on MMLU under AIME attack)
    # Sequential SFT+DPO achieved best defense results in report

    # Strategy 1: SFT-only
    #"examples/train_distractor/qwen_3_8b_lora_sft.yaml"

    # Strategy 2: DPO-only
    #"examples/train_distractor/qwen_3_8b_lora_dpo.yaml"

    # Strategy 3: Sequential SFT+DPO (primary results reported in paper)
    #"examples/train_distractor/qwen_3_8b_lora_sft_dpo.yaml"

    # ========== Note on Other Models ==========
    # Phi-4-mini and DeepSeek-R1-Distill-Llama-8B were evaluated for
    # vulnerability (Phase 1) but NOT included in defense training (Phase 2)
    # due to computational constraints mentioned in report Section 3.2.2
)

# --- Training Loop ---
echo "=========================================="
echo "  Defense Training - Course Project"
echo "=========================================="
echo ""
echo "Models evaluated in report:"
echo "  - Vulnerability (Phase 1): DeepSeek-8B, Qwen-3-4B, Qwen-3-8B, Phi-4-mini"
echo "  - Defense (Phase 2): Qwen-3-4B, Qwen-3-8B only"
echo ""
echo "Training strategies:"
echo "  1. SFT-only: Essential baseline (49.7-59.9 point recovery)"
echo "  2. DPO-only: Minimal effect (≤3.3 point improvement)"
echo "  3. Sequential SFT+DPO: Best performance (59.2-59.9 point recovery)"
echo ""
echo "=========================================="
echo ""

for config_file in "${CONFIG_FILES[@]}"
do
    echo "##################################################"
    echo "###                                            ###"
    echo "###      STARTING TRAINING                     ###"
    echo "###      Config: $(basename $config_file)"
    echo "###                                            ###"
    echo "##################################################"
    echo ""

    # Check if the config file exists
    if [ -f "$config_file" ]; then
        # Extract training info from filename
        if [[ "$config_file" == *"sft_dpo"* ]]; then
            echo "Training Strategy: Sequential SFT+DPO (Best in report)"
            echo "Expected Improvement: 59.2-59.9 percentage points"
        elif [[ "$config_file" == *"sft"* ]]; then
            echo "Training Strategy: SFT-only"
            echo "Expected Improvement: 49.7-59.9 percentage points"
        elif [[ "$config_file" == *"dpo"* ]]; then
            echo "Training Strategy: DPO-only"
            echo "Expected Improvement: Minimal (-0.15 to +3.3 points)"
        fi
        echo ""

        # Run the training command
        echo "Executing: llamafactory-cli train $config_file"
        echo ""
        llamafactory-cli train "$config_file"

        echo ""
        echo "##################################################"
        echo "###                                            ###"
        echo "###      COMPLETED TRAINING                    ###"
        echo "###      Config: $(basename $config_file)"
        echo "###                                            ###"
        echo "##################################################"
        echo ""
    else
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!!!                                            !!!"
        echo "!!!      ERROR: Config file not found          !!!"
        echo "!!!      $config_file"
        echo "!!!                                            !!!"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        exit 1
    fi
done

echo "=========================================="
echo "  ALL TRAINING JOBS COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate defended models using evaluation scripts in:"
echo "     - mmlu_redux/run_eval_offline.sh"
echo "     - MATH/run_eval_offline.sh"
echo "     - instruction_following/run_infer_offline.sh"
echo "     - LRM_as_a_judge/run_eval_offline_with_distraction.sh"
echo ""
echo "  2. Compare results against report baselines (Table 2, Section 4.2)"
echo "=========================================="
