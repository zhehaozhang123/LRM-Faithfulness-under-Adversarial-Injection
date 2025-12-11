#!/usr/bin/env python3
"""
Simple script to check and display the evaluation results.
"""

import json
import os
from pathlib import Path
import pandas as pd

def parse_filename(filename):
    """Parse filename to extract model, distraction type, direct type, and position."""
    basename = os.path.basename(filename)
    
    # Handle the new naming convention from outputs_if_mitigation
    if 'results_ifeval__data_bfcl_gorilla_berkeley-function-call-leaderboard_models_' in basename:
        name = basename.replace('results_ifeval__data_bfcl_gorilla_berkeley-function-call-leaderboard_models_', '').replace('.jsonl', '')
        
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
    
    # Original parsing for old format
    name = basename.replace('results_ifeval_', '').replace('.jsonl', '')
    
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

def compute_accuracy(result_file):
    """Compute accuracy from a result file."""
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
        
        total = len(lines)
        passed = 0
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                if data.get('follow_all_instructions', False):
                    passed += 1
            except:
                continue
        
        accuracy = (passed / total) * 100 if total > 0 else 0
        return accuracy, total, passed
        
    except Exception as e:
        print(f"Error processing {result_file}: {e}")
        return None, None, None

def main():
    # Set up paths
    results_dir = Path('/home/ec2-user/zhehaozhang/LRM_distraction/instruction_following/instruction_following_eval/results')
    
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    # Get all result subdirectories
    result_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('results_ifeval_')]
    
    if not result_dirs:
        print("No result directories found")
        return
    
    print(f"Found {len(result_dirs)} result directories")
    
    # Collect results
    results = []
    
    for result_dir in result_dirs:
        print(f"Processing {result_dir.name}...")
        
        # Parse filename to extract configuration
        config = parse_filename(result_dir.name)
        
        # Check for strict and loose results
        strict_file = result_dir / "eval_results_strict.jsonl"
        loose_file = result_dir / "eval_results_loose.jsonl"
        
        strict_accuracy = None
        loose_accuracy = None
        
        if strict_file.exists():
            strict_accuracy, total, passed = compute_accuracy(strict_file)
            if strict_accuracy is not None:
                print(f"  Strict: {strict_accuracy:.1f}% ({passed}/{total})")
        
        if loose_file.exists():
            loose_accuracy, total, passed = compute_accuracy(loose_file)
            if loose_accuracy is not None:
                print(f"  Loose: {loose_accuracy:.1f}% ({passed}/{total})")
        
        if strict_accuracy is not None or loose_accuracy is not None:
            results.append({
                'model': config['model'],
                'distraction_type': config['distraction_type'],
                'direct_type': config['direct_type'],
                'position': config['position'],
                'strict_accuracy': strict_accuracy,
                'loose_accuracy': loose_accuracy,
                'directory': result_dir.name
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
    csv_file = results_dir / "evaluation_summary_if_mitigation.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to: {csv_file}")

if __name__ == "__main__":
    main()


