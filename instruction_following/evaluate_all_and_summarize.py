#!/usr/bin/env python3
"""
Script to evaluate all files in outputs_llm directory and summarize results for easy comparison.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re

def parse_filename(filename):
    """
    Parse filename to extract model, distraction type, direct type, and position.
    
    Expected format for outputs_if_mitigation: ifeval__data_bfcl_gorilla_berkeley-function-call-leaderboard_models_<model>_<training_type>__<distraction_type>_<position>.jsonl
    For baseline: ifeval__data_bfcl_gorilla_berkeley-function-call-leaderboard_models_<model>_<training_type>_.jsonl
    """
    basename = os.path.basename(filename)
    
    # Remove the prefix and suffix
    name = basename.replace('ifeval__data_bfcl_gorilla_berkeley-function-call-leaderboard_models_', '').replace('.jsonl', '')
    
    # Handle baseline case (no distraction) - files ending with just training type
    if name.endswith('_'):
        # Extract model and training type
        parts = name.rstrip('_').split('_')
        if len(parts) >= 2:
            training_type = parts[-1]
            model = '_'.join(parts[:-1])
        else:
            model = name.rstrip('_')
            training_type = 'unknown'
        
        return {
            'model': f"{model}_{training_type}",
            'distraction_type': 'none',
            'direct_type': None,
            'position': None
        }
    
    # Handle distraction cases - format: model_training_type__distraction_type_position
    if '__' in name:
        model_training_part, distraction_part = name.split('__', 1)
        
        # Extract model and training type
        model_training_parts = model_training_part.split('_')
        if len(model_training_parts) >= 2:
            training_type = model_training_parts[-1]
            model = '_'.join(model_training_parts[:-1])
        else:
            model = model_training_part
            training_type = 'unknown'
        
        # Extract distraction type and position
        distraction_parts = distraction_part.split('_')
        if len(distraction_parts) >= 2:
            # Handle multi-word distraction types like 'math_aime' and 'zebra_logic'
            if distraction_parts[0] == 'math' and len(distraction_parts) >= 3:
                distraction_type = 'math_aime'
                position = distraction_parts[2]
            elif distraction_parts[0] == 'zebra' and len(distraction_parts) >= 3:
                distraction_type = 'zebra_logic'
                position = distraction_parts[2]
            else:
                distraction_type = distraction_parts[0]
                position = distraction_parts[1]
        else:
            distraction_type = distraction_part
            position = 'unknown'
        
        return {
            'model': f"{model}_{training_type}",
            'distraction_type': distraction_type,
            'direct_type': None,
            'position': position
        }
    
    # Fallback for unexpected format
    return {
        'model': name,
        'distraction_type': 'unknown',
        'direct_type': None,
        'position': 'unknown'
    }

def run_evaluation(input_file, output_dir):
    """Run evaluation_main.py for a single file."""
    try:
        # Run the evaluation using the module approach from parent directory
        # Extract just the directory name from the full path
        output_dir_name = os.path.basename(output_dir)
        cmd = [
            'python3', '-m', 'instruction_following_eval.evaluation_main',
            '--input_data=./instruction_following_eval/data/input_data.jsonl',
            f'--input_response_data=./instruction_following_eval/outputs_if_mitigation/{os.path.basename(input_file)}',
            f'--output_dir=./instruction_following_eval/results/{output_dir_name}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following')
        
        if result.returncode != 0:
            print(f"Error evaluating {input_file}: {result.stderr}")
            return None
        
        # Extract accuracy from the output
        lines = result.stdout.split('\n')
        strict_accuracy = None
        loose_accuracy = None
        
        for line in lines:
            if 'Accuracy:' in line and 'Generating eval_results_strict' in result.stdout:
                # Find the strict accuracy line
                if 'Generating eval_results_strict' in result.stdout and 'Accuracy:' in line:
                    try:
                        strict_accuracy = float(line.split('Accuracy:')[1].strip())
                    except:
                        pass
            elif 'Accuracy:' in line and 'Generating eval_results_loose' in result.stdout:
                # Find the loose accuracy line
                if 'Generating eval_results_loose' in result.stdout and 'Accuracy:' in line:
                    try:
                        loose_accuracy = float(line.split('Accuracy:')[1].strip())
                    except:
                        pass
        
        return {
            'strict_accuracy': strict_accuracy,
            'loose_accuracy': loose_accuracy
        }
        
    except Exception as e:
        print(f"Exception evaluating {input_file}: {e}")
        return None

def load_accuracy_from_results(results_dir, filename_prefix):
    """Load accuracy from existing results files."""
    strict_file = os.path.join(results_dir, f"{filename_prefix}_strict.jsonl")
    loose_file = os.path.join(results_dir, f"{filename_prefix}_loose.jsonl")
    
    strict_accuracy = None
    loose_accuracy = None
    
    # Load strict accuracy
    if os.path.exists(strict_file):
        try:
            with open(strict_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Calculate accuracy from the results
                    total = len(lines)
                    passed = sum(1 for line in lines if json.loads(line.strip()).get('follow_all_instructions', False))
                    strict_accuracy = (passed / total) * 100 if total > 0 else 0
        except Exception as e:
            print(f"Error loading strict results from {strict_file}: {e}")
    
    # Load loose accuracy
    if os.path.exists(loose_file):
        try:
            with open(loose_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Calculate accuracy from the results
                    total = len(lines)
                    passed = sum(1 for line in lines if json.loads(line.strip()).get('follow_all_instructions', False))
                    loose_accuracy = (passed / total) * 100 if total > 0 else 0
        except Exception as e:
            print(f"Error loading loose results from {loose_file}: {e}")
    
    return strict_accuracy, loose_accuracy

def main():
    # Set up paths
    outputs_dir = Path('/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following/instruction_following_eval/outputs_if_mitigation')
    results_dir = Path('/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following/instruction_following_eval/results')
    
    if not outputs_dir.exists():
        print(f"Error: Outputs directory {outputs_dir} does not exist")
        return
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)
    
    # Get all jsonl files
    jsonl_files = list(outputs_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print("No jsonl files found in outputs_if_mitigation directory")
        return
    
    print(f"Found {len(jsonl_files)} files to evaluate")
    
    # Collect results
    results = []
    
    for filepath in jsonl_files:
        print(f"Processing {filepath.name}...")
        
        # Parse filename to extract configuration
        config = parse_filename(filepath.name)
        
        # Create a unique results directory for this file
        file_results_dir = results_dir / f"results_{filepath.stem}"
        file_results_dir.mkdir(exist_ok=True)
        
        # Check if results already exist
        strict_accuracy, loose_accuracy = load_accuracy_from_results(str(file_results_dir), "eval_results")
        
        if strict_accuracy is None or loose_accuracy is None:
            # Run evaluation
            print(f"  Running evaluation for {filepath.name}...")
            eval_results = run_evaluation(filepath, str(file_results_dir))
            
            if eval_results:
                strict_accuracy = eval_results['strict_accuracy']
                loose_accuracy = eval_results['loose_accuracy']
            else:
                print(f"  Failed to evaluate {filepath.name}")
                continue
        else:
            print(f"  Using existing results for {filepath.name}")
        
        if strict_accuracy is not None and loose_accuracy is not None:
            results.append({
                'model': config['model'],
                'distraction_type': config['distraction_type'],
                'direct_type': config['direct_type'],
                'position': config['position'],
                'strict_accuracy': strict_accuracy,
                'loose_accuracy': loose_accuracy,
                'filename': filepath.name
            })
    
    if not results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("INSTRUCTION FOLLOWING EVALUATION RESULTS SUMMARY (IF MITIGATION)")
    print("=" * 80)
    print(f"Total experiments: {len(results)}")
    print(f"Models analyzed: {len(df['model'].unique())}")
    print()
    
    # Generate tables for each model
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        
        print("=" * 80)
        print(f"TABLE FOR MODEL: {model}")
        print("=" * 80)
        
        # Get baseline accuracy
        baseline_df = model_df[model_df['distraction_type'] == 'none']
        baseline_strict = baseline_df['strict_accuracy'].iloc[0] if not baseline_df.empty else None
        baseline_loose = baseline_df['loose_accuracy'].iloc[0] if not baseline_df.empty else None
        
        if baseline_strict is not None and baseline_loose is not None:
            print(f"Baseline Accuracy (no injection): Strict={baseline_strict:.1f}%, Loose={baseline_loose:.1f}%")
            print("-" * 80)
        
        # Create results table
        print(f"{'Configuration':<25} {'Start':<12} {'Middle':<12} {'End':<12} {'Average':<12}")
        print(f"{'':25} {'S':<6} {'L':<6} {'S':<6} {'L':<6} {'S':<6} {'L':<6} {'S':<6} {'L':<6}")
        print("-" * 80)
        
        # Process each distraction type
        for distraction_type in sorted(model_df['distraction_type'].unique()):
            if distraction_type == 'none':
                continue
                
            type_df = model_df[model_df['distraction_type'] == distraction_type]
            
            if distraction_type == 'direct':
                # Direct injection - group by direct_type
                for direct_type in sorted(type_df['direct_type'].unique()):
                    if pd.isna(direct_type):
                        continue
                    direct_df = type_df[type_df['direct_type'] == direct_type]
                    
                    config_name = f"direct_{direct_type}"
                    
                    # Get accuracies for each position
                    start_strict = direct_df[direct_df['position'] == 'start']['strict_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'start'].empty else None
                    start_loose = direct_df[direct_df['position'] == 'start']['loose_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'start'].empty else None
                    middle_strict = direct_df[direct_df['position'] == 'middle']['strict_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'middle'].empty else None
                    middle_loose = direct_df[direct_df['position'] == 'middle']['loose_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'middle'].empty else None
                    end_strict = direct_df[direct_df['position'] == 'end']['strict_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'end'].empty else None
                    end_loose = direct_df[direct_df['position'] == 'end']['loose_accuracy'].iloc[0] if not direct_df[direct_df['position'] == 'end'].empty else None
                    
                    # Calculate averages
                    strict_values = [x for x in [start_strict, middle_strict, end_strict] if x is not None]
                    loose_values = [x for x in [start_loose, middle_loose, end_loose] if x is not None]
                    
                    avg_strict = sum(strict_values) / len(strict_values) if strict_values else None
                    avg_loose = sum(loose_values) / len(loose_values) if loose_values else None
                    
                    # Format strings
                    start_str = f"{start_strict:.1f}/{start_loose:.1f}" if start_strict is not None and start_loose is not None else "N/A"
                    middle_str = f"{middle_strict:.1f}/{middle_loose:.1f}" if middle_strict is not None and middle_loose is not None else "N/A"
                    end_str = f"{end_strict:.1f}/{end_loose:.1f}" if end_strict is not None and end_loose is not None else "N/A"
                    avg_str = f"{avg_strict:.1f}/{avg_loose:.1f}" if avg_strict is not None and avg_loose is not None else "N/A"
                    
                    print(f"{config_name:<25} {start_str:<12} {middle_str:<12} {end_str:<12} {avg_str:<12}")
            else:
                # Regular distraction types
                start_strict = type_df[type_df['position'] == 'start']['strict_accuracy'].iloc[0] if not type_df[type_df['position'] == 'start'].empty else None
                start_loose = type_df[type_df['position'] == 'start']['loose_accuracy'].iloc[0] if not type_df[type_df['position'] == 'start'].empty else None
                middle_strict = type_df[type_df['position'] == 'middle']['strict_accuracy'].iloc[0] if not type_df[type_df['position'] == 'middle'].empty else None
                middle_loose = type_df[type_df['position'] == 'middle']['loose_accuracy'].iloc[0] if not type_df[type_df['position'] == 'middle'].empty else None
                end_strict = type_df[type_df['position'] == 'end']['strict_accuracy'].iloc[0] if not type_df[type_df['position'] == 'end'].empty else None
                end_loose = type_df[type_df['position'] == 'end']['loose_accuracy'].iloc[0] if not type_df[type_df['position'] == 'end'].empty else None
                
                # Calculate averages
                strict_values = [x for x in [start_strict, middle_strict, end_strict] if x is not None]
                loose_values = [x for x in [start_loose, middle_loose, end_loose] if x is not None]
                
                avg_strict = sum(strict_values) / len(strict_values) if strict_values else None
                avg_loose = sum(loose_values) / len(loose_values) if loose_values else None
                
                # Format strings
                start_str = f"{start_strict:.1f}/{start_loose:.1f}" if start_strict is not None and start_loose is not None else "N/A"
                middle_str = f"{middle_strict:.1f}/{middle_loose:.1f}" if middle_strict is not None and middle_loose is not None else "N/A"
                end_str = f"{end_strict:.1f}/{end_loose:.1f}" if end_strict is not None and end_loose is not None else "N/A"
                avg_str = f"{avg_strict:.1f}/{avg_loose:.1f}" if avg_strict is not None and avg_loose is not None else "N/A"
                
                print(f"{distraction_type:<25} {start_str:<12} {middle_str:<12} {end_str:<12} {avg_str:<12}")
        
        print()  # Add spacing between tables
    
    # Save detailed results to CSV
    csv_file = results_dir / "detailed_results_if_mitigation.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")

if __name__ == "__main__":
    main()
