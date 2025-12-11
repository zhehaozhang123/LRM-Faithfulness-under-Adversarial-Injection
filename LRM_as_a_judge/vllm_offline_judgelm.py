import json
import argparse
import os
from pathlib import Path
from typing import Optional, Dict
from utils import (
    infer_reasoning_offline,
    label_from_scores,
    compute_classification_metrics,
    extract_scores_from_llm_output,
    count_tokens_openai
)
from prompt.prompt_judgelm import system_prompt, user_prompt
from distraction_injection import inject_distraction_prompt
from tqdm import tqdm
from vllm import LLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="sampled_data/judgelm/judgelm_val_5k_gpt4_clean_sampled_200.jsonl", help="input dataset file")
    parser.add_argument('--persuasion_file', type=str, help="JSON file containing pre-generated persuasion attacks")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Local model path or Hugging Face model ID"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32768,
        help="Max tokens to generate"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.95,
        help="The fraction of GPU memory to be used for the model executor"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5000,
        help="Batch size for inference"
    )
    # Add new distraction injection arguments
    parser.add_argument(
        "--injection_type",
        type=str,
        choices=['direct', 'math_aime', 'arithmetic', 'zebra_logic', 'dyck', 'none', 'code'],
        default='none',
        help="Type of distraction to inject into prompts"
    )
    parser.add_argument(
        "--direct_type",
        type=str,
        choices=['naive', 'escape', 'context', 'completion'],
        default='naive',
        help="Type of direct prompt injection to use when injection_type is 'direct'"
    )
    parser.add_argument(
        "--injection_position",
        type=str,
        choices=['start', 'middle', 'end'],
        default='end',
        help="Position to inject the distraction in the prompt"
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="saved_prompts",
        help="Directory to save/load prompts"
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    
    # Create output filename with attack information
    attack_info = ""
    if args.injection_type != 'none':
        attack_info = f"_{args.injection_type}"
        if args.injection_type == 'direct':
            attack_info += f"_{args.direct_type}"
        attack_info += f"_{args.injection_position}"
    
    output_file = f"outputs/{Path(args.input_file).stem}_{args.model.replace('/', '_')}{attack_info}.jsonl"
    
    # Check if metrics file already exists - if so, skip inference
    metrics_file = output_file.replace(".jsonl", "_metrics.json")
    if os.path.exists(metrics_file):
        print(f"Metrics file already exists: {metrics_file}")
        print("Skipping inference as results are already available.")
        print("To rerun inference, please delete the metrics file first.")
        return
    
    # Create prompts filename
    os.makedirs(args.prompts_dir, exist_ok=True)
    prompts_file = f"{args.prompts_dir}/judgelm_prompts{attack_info}.jsonl"

    # Initialize tokenizer and model once
    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine_args = {
        "model": args.model,
        "device": "cuda",
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": True,
    }
    llm = LLM(**engine_args)

    # -- RESUMABLE LOGIC --
    if os.path.exists(output_file):
        print(f"Resuming from {output_file}")
        records = []
        with open(output_file, "r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line))
        n_finished = sum('pred_scores' in rec for rec in records)
    else:
        print("Loading merged JudgeLM JSONL...")
        merged_path = args.input_file
        records = []
        with open(merged_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Preparing merged dataset"):
                ex = json.loads(line)
                records.append({
                    "question_body": ex["question_body"],
                    "answer1_body": ex["answer1_body"],
                    "answer2_body": ex["answer2_body"],
                    "gt_scores": ex["score"],
                })
        n_finished = 0

    print(f"{n_finished}/{len(records)} already judged. {len(records)-n_finished} to go.")

    # Check if we have saved prompts
    saved_prompts = {}
    if os.path.exists(prompts_file):
        print(f"Found saved prompts file: {prompts_file}")
        print("Loading saved prompts to reuse...")
        with open(prompts_file, "r") as f:
            for line in f:
                prompt_record = json.loads(line)
                key = (prompt_record["question"], prompt_record["answer1"], prompt_record["answer2"])
                saved_prompts[key] = prompt_record["prompt"]
        print(f"Loaded {len(saved_prompts)} saved prompts")

    # -- Inference loop with batching --
    try:
        # Get unfinished records
        unfinished_records = [r for r in records if 'pred_scores' not in r]
        
        # Process in batches
        for i in tqdm(range(0, len(unfinished_records), args.batch_size), desc="Judging"):
            batch_records = unfinished_records[i:i + args.batch_size]
            
            # Prepare prompts for the batch
            prompts = []
            for rec in batch_records:
                # Format base prompt
                base_prompt = user_prompt.format(
                    question=rec["question_body"],
                    answer_1=rec["answer1_body"],
                    answer_2=rec["answer2_body"]
                )
                
                # Check if we have a saved prompt
                key = (rec["question_body"], rec["answer1_body"], rec["answer2_body"])
                if key in saved_prompts:
                    user_query = saved_prompts[key]
                    print(f"Reusing saved prompt")
                else:
                    # Apply distraction injection if specified
                    injection_type = None if args.injection_type == 'none' else args.injection_type
                    user_query = inject_distraction_prompt(
                        base_prompt,
                        injection_type,
                        position=args.injection_position,
                        dataset='judgelm',
                        direct_type=args.direct_type
                    )
                    
                    # Save the generated prompt
                    prompt_record = {
                        "question": rec["question_body"],
                        "answer1": rec["answer1_body"],
                        "answer2": rec["answer2_body"],
                        "base_prompt": base_prompt,
                        "prompt": user_query,
                        "injection_info": {
                            "type": args.injection_type,
                            "direct_type": args.direct_type if args.injection_type == "direct" else None,
                            "position": args.injection_position if args.injection_type != "none" else None
                        }
                    }
                    # Append to prompts file
                    with open(prompts_file, "a") as f:
                        f.write(json.dumps(prompt_record) + "\n")
                
                prompts.append(tokenizer.apply_chat_template(
                    [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_query}],
                    tokenize=False,
                    add_generation_prompt=True
                ))
                
                # Store the prompts for saving later
                rec["user_prompt_to_judge"] = user_query
                rec["system_prompt_to_judge"] = system_prompt

            # Run batch inference
            results = infer_reasoning_offline(
                llm=llm,
                prompt=prompts,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            # Process results
            for j, rec in enumerate(batch_records):
                reasoning, output = results[j]
                try:
                    pred_scores = extract_scores_from_llm_output(output)
                    rec["llm_reasoning"] = reasoning
                    rec["llm_output"] = output
                    rec["pred_scores"] = pred_scores
                    rec["injection_info"] = {
                        "type": args.injection_type,
                        "direct_type": args.direct_type if args.injection_type == "direct" else None,
                        "position": args.injection_position if args.injection_type != "none" else None
                    }
                except Exception as e:
                    print(f"[{i+j}] Extraction error: {e}\nOutput: {output}")
                    continue

            # Save progress after each batch
            with open(output_file, "w", encoding="utf-8") as fout:
                for item in records:
                    fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    except KeyboardInterrupt:
        print("\nExecution interrupted! Progress safely stored.")
        print(f"{sum('pred_scores' in r for r in records)}/{len(records)} already judged. {len(records)-sum('pred_scores' in r for r in records)} to go.")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Progress safely stored.")
        print(f"{sum('pred_scores' in r for r in records)}/{len(records)} already judged. {len(records)-sum('pred_scores' in r for r in records)} to go.")

    print(f"\nFinal predictions written to {output_file}")

    # -- Metric computation --
    completed_records = [r for r in records if "pred_scores" in r]
    if completed_records:
        # Overall metrics
        total = len(completed_records)
        gt_labels = [label_from_scores(r["gt_scores"][0], r["gt_scores"][1]) for r in completed_records]
        pred_labels = [label_from_scores(r["pred_scores"][0], r["pred_scores"][1]) for r in completed_records]
        metrics = compute_classification_metrics(gt_labels, pred_labels)
        
        # Calculate token usage
        total_tokens = 0
        for record in completed_records:
            combined_text = record["llm_reasoning"] + " " + record["llm_output"]
            total_tokens += count_tokens_openai(combined_text)
        avg_tokens = total_tokens / total if total > 0 else 0

        # Save metrics summary
        metrics_file = output_file.replace(".jsonl", "_metrics.json")
        metrics_summary = {
            "overall": {
                "agreement": metrics["agreement"] * 100,
                "precision": metrics["precision"] * 100,
                "recall": metrics["recall"] * 100,
                "f1": metrics["f1"] * 100,
                "total": total,
                "total_examples": len(records),
                "total_tokens": total_tokens,
                "avg_tokens_per_example": avg_tokens
            }
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        print(f"\nMetrics summary saved to {metrics_file}")
        
        # Print metrics
        print(f"\nOverall Results:")
        print(f"Agreement: {metrics['agreement']*100:.2f}% | "
              f"Precision: {metrics['precision']*100:.2f}% | "
              f"Recall: {metrics['recall']*100:.2f}% | "
              f"F1: {metrics['f1']*100:.2f}%")

if __name__ == "__main__":
    main()
