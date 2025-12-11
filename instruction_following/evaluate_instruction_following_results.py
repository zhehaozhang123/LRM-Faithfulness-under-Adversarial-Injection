#!/usr/bin/env python3
"""
Comprehensive script to evaluate instruction following results and generate summaries.
This script processes all result files in the instruction_following_output directory.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import shutil
from collections import defaultdict

def parse_filename(filename):
    """Parse filename to extract model, distraction type, direct type, and position."""
    basename = os.path.basename(filename)
    name = basename.replace('ifeval_', '').replace('.jsonl', '')
    
    # Handle baseline case (no distraction)
    if not any(keyword in name for keyword in ['arithmetic', 'code', 'dyck', 'math_aime', 'zebra_logic', 'direct']):
        return {
            'model': name,
            'distraction_type': 'none',
            'direct_type': None,
            'position': None
        }
    
    # Split by underscore to parse components
    parts = name.split('_')
    
    # Known distraction types and positions
    distraction_keywords = ['arithmetic', 'code', 'dyck', 'math_aime', 'zebra_logic', 'direct']
    position_keywords = ['start', 'middle', 'end']
    direct_types = ['naive', 'escape', 'context', 'completion']
    
    # Look for multi-word distraction patterns first
    model_parts = []
    distraction_start_idx = None
    distraction_type = None
    
    for i in range(len(parts) - 1):
        if parts[i] == 'math' and parts[i + 1] == 'aime':
            distraction_start_idx = i
            distraction_type = 'math_aime'
            break
        elif parts[i] == 'zebra' and parts[i + 1] == 'logic':
            distraction_start_idx = i
            distraction_type = 'zebra_logic'
            break
    
    if distraction_start_idx is None:
        # Look for other distraction keywords
        for i, part in enumerate(parts):
            if part in distraction_keywords:
                distraction_start_idx = i
                distraction_type = part
                break
            model_parts.append(part)
    
    if distraction_start_idx is None:
        # If no explicit distraction keyword found, look for position keywords
        for i, part in enumerate(parts):
            if part in position_keywords:
                distraction_start_idx = i
                break
        
        if distraction_start_idx is None:
            # Fallback: assume everything except last part is model
            model = '_'.join(parts[:-1])
            distraction_type = parts[-1]
            position = None
            direct_type = None
        else:
            # Model is everything before position, distraction type is the part before position
            model = '_'.join(parts[:distraction_start_idx-1])
            distraction_type = parts[distraction_start_idx-1]
            position = parts[distraction_start_idx]
            direct_type = None
    else:
        if distraction_type in ['math_aime', 'zebra_logic']:
            # For multi-word distraction types, model is everything before the first word
            model = '_'.join(parts[:distraction_start_idx])
            position = parts[distraction_start_idx + 2] if distraction_start_idx + 2 < len(parts) else None
            direct_type = None
        else:
            model = '_'.join(model_parts)
            distraction_type = parts[distraction_start_idx]
            
            # Handle direct injection case
            if distraction_type == 'direct':
                if distraction_start_idx + 2 < len(parts):
                    direct_type = parts[distraction_start_idx + 1]
                    position = parts[distraction_start_idx + 2]
                elif distraction_start_idx + 1 < len(parts):
                    direct_type = parts[distraction_start_idx + 1]
                    position = None
                else:
                    direct_type = None
                    position = None
            else:
                direct_type = None
                position = parts[distraction_start_idx + 1] if distraction_start_idx + 1 < len(parts) else None
    
    return {
        'model': model,
        'distraction_type': distraction_type,
        'direct_type': direct_type,
        'position': position
    }

def run_evaluation(input_file, output_dir):
    """Run evaluation_main.py for a single file."""
    try:
        # Run evaluation from the parent directory to fix import issues
        cmd = [
            'python3', '-m', 'instruction_following_eval.evaluation_main',
            '--input_data=./instruction_following_eval/data/input_data.jsonl',
            f'--input_response_data={input_file}',
            f'--output_dir={output_dir}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              cwd='/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following')
        
        if result.returncode != 0:
            print(f"Error evaluating {input_file}: {result.stderr}")
            return None
        
        # Extract accuracy from the output
        lines = result.stdout.split('\n')
        strict_accuracy = None
        loose_accuracy = None
        
        for line in lines:
            if 'Accuracy:' in line:
                try:
                    accuracy = float(line.split('Accuracy:')[1].strip())
                    if 'strict' in result.stdout.lower() and strict_accuracy is None:
                        strict_accuracy = accuracy
                    elif 'loose' in result.stdout.lower() and loose_accuracy is None:
                        loose_accuracy = accuracy
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
    input_dir = Path('/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following/instruction_following_eval/instruction_following_output')
    results_dir = Path('/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following/instruction_following_eval/results')
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(exist_ok=True)
    
    # Get all jsonl files
    jsonl_files = list(input_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print("No jsonl files found in instruction_following_output directory")
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
            eval_results = run_evaluation(str(filepath), str(file_results_dir))
            
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
            print(f"  Success: Strict={strict_accuracy:.1f}%, Loose={loose_accuracy:.1f}%")
    
    if not results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("INSTRUCTION FOLLOWING EVALUATION RESULTS SUMMARY")
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
    
    # Create a comparison table across all models
    print("=" * 80)
    print("CROSS-MODEL COMPARISON (Average Across All Distraction Types)")
    print("=" * 80)
    
    comparison_data = []
    for model in sorted(df['model'].unique()):
        model_df = df[df['model'] == model]
        
        # Calculate average across all distraction types (excluding baseline)
        non_baseline_df = model_df[model_df['distraction_type'] != 'none']
        
        if not non_baseline_df.empty:
            avg_strict = non_baseline_df['strict_accuracy'].mean()
            avg_loose = non_baseline_df['loose_accuracy'].mean()
            
            # Get baseline for comparison
            baseline_df = model_df[model_df['distraction_type'] == 'none']
            baseline_strict = baseline_df['strict_accuracy'].iloc[0] if not baseline_df.empty else None
            baseline_loose = baseline_df['loose_accuracy'].iloc[0] if not baseline_df.empty else None
            
            comparison_data.append({
                'model': model,
                'baseline_strict': baseline_strict,
                'baseline_loose': baseline_loose,
                'avg_strict': avg_strict,
                'avg_loose': avg_loose,
                'strict_drop': baseline_strict - avg_strict if baseline_strict is not None else None,
                'loose_drop': baseline_loose - avg_loose if baseline_loose is not None else None
            })
    
    if comparison_data:
        print(f"{'Model':<40} {'Baseline':<15} {'Avg w/ Distraction':<20} {'Performance Drop':<20}")
        print(f"{'':40} {'S':<7} {'L':<7} {'S':<7} {'L':<7} {'S':<7} {'L':<7}")
        print("-" * 80)
        
        for data in comparison_data:
            baseline_str = f"{data['baseline_strict']:.1f}/{data['baseline_loose']:.1f}" if data['baseline_strict'] is not None else "N/A"
            avg_str = f"{data['avg_strict']:.1f}/{data['avg_loose']:.1f}"
            drop_str = f"{data['strict_drop']:.1f}/{data['loose_drop']:.1f}" if data['strict_drop'] is not None else "N/A"
            
            print(f"{data['model']:<40} {baseline_str:<15} {avg_str:<20} {drop_str:<20}")
    
    # Save detailed results to CSV
    csv_file = results_dir / "instruction_following_detailed_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")

if __name__ == "__main__":
    main()
