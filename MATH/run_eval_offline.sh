#!/bin/bash

# Course Project: Evaluate models on MATH-500 benchmark
# Models from the course report

# Base models to evaluate
MODELS=(
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "microsoft/phi-4-reasoning-mini"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Distraction types (only those in course report)
DISTRACTIONS=(
    "none"
    "math_aime"
    "code"
    "zebra_logic"
)

# Injection position (end position most effective)
POSITION="end"

# Run evaluations
for MODEL in "${MODELS[@]}"; do
    echo "================================"
    echo "Evaluating model: $MODEL"
    echo "================================"

    for DISTRACTION in "${DISTRACTIONS[@]}"; do
        echo "Running with distraction: $DISTRACTION"

        if [ "$DISTRACTION" == "none" ]; then
            # Clean baseline (no distraction)
            python eval/eval_math500_offline.py \
                --model "$MODEL" \
                --injection_type none \
                --temperature 0.0 \
                --tensor_parallel_size 8
        else
            # With distraction
            python eval/eval_math500_offline.py \
                --model "$MODEL" \
                --injection_type "$DISTRACTION" \
                --injection_position "$POSITION" \
                --temperature 0.0 \
                --tensor_parallel_size 8
        fi
    done
done

echo ""
echo "All evaluations completed!"
echo "Results saved to eval/outputs/"
