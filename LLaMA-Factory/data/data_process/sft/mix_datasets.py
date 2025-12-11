#!/usr/bin/env python3
"""
Script to sample and mix the Hugging Face dataset with the combined SFT dataset.
Usage: python mix_datasets.py --hf_count 5000 --sft_count 10000 --output mixed_dataset.json
"""

import json
import random
import argparse
import re
from pathlib import Path
from datasets import load_dataset

def convert_openthoughts_tags(text):
    """Convert OpenThoughts tags to the desired format."""
    if not text:
        return text
    
    # Replace thought tags
    text = re.sub(r'<\|begin_of_thought\|>', '<think>', text)
    text = re.sub(r'<\|end_of_thought\|>', '</think>', text)
    
    # Replace solution tags (if you want to keep them, otherwise remove these lines)
    text = re.sub(r'<\|begin_of_solution\|>', '<answer>', text)
    text = re.sub(r'<\|end_of_solution\|>', '</answer>', text)
    
    return text

def convert_hf_to_sft_format(hf_sample):
    """Convert Hugging Face format to SFT format."""
    # Extract system message
    system = hf_sample.get('system', '')
    
    # Extract conversations
    conversations = hf_sample.get('conversations', [])
    
    if not conversations:
        return None
    
    # Find user and assistant messages
    user_messages = []
    assistant_messages = []
    
    for conv in conversations:
        if conv.get('from') == 'user':
            user_messages.append(conv.get('value', ''))
        elif conv.get('from') == 'assistant':
            assistant_messages.append(conv.get('value', ''))
    
    # Combine user messages as instruction
    instruction = '\n'.join(user_messages) if user_messages else ''
    
    # Use the first assistant message as output and convert tags
    output = assistant_messages[0] if assistant_messages else ''
    output = convert_openthoughts_tags(output)
    
    # Also convert tags in instruction if needed
    instruction = convert_openthoughts_tags(instruction)
    
    # Convert system message tags if needed
    system = convert_openthoughts_tags(system)
    
    # Create SFT format
    sft_sample = {
        'instruction': instruction,
        'input': '',  # Empty input as per SFT format
        'output': output,
        'system': system
    }
    
    return sft_sample

def load_sft_dataset(file_path):
    """Load the combined SFT dataset."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sample_and_mix_datasets(hf_samples, sft_samples, hf_count, sft_count, output_path):
    """Sample and mix datasets."""
    
    print(f"Sampling {hf_count} samples from HF dataset...")
    hf_sampled = random.sample(hf_samples, min(hf_count, len(hf_samples)))
    
    print(f"Sampling {sft_count} samples from SFT dataset...")
    sft_sampled = random.sample(sft_samples, min(sft_count, len(sft_samples)))
    
    print("Converting HF samples to SFT format and updating tags...")
    converted_hf = []
    tag_conversions = 0
    
    for sample in hf_sampled:
        converted = convert_hf_to_sft_format(sample)
        if converted and converted['instruction'].strip() and converted['output'].strip():
            # Count tag conversions for logging
            if ('<|begin_of_thought|>' in str(sample) or '<|end_of_thought|>' in str(sample) or
                '<|begin_of_solution|>' in str(sample) or '<|end_of_solution|>' in str(sample)):
                tag_conversions += 1
            converted_hf.append(converted)
    
    print(f"Successfully converted {len(converted_hf)} HF samples")
    print(f"Converted tags in {tag_conversions} samples")
    
    # Combine datasets
    mixed_dataset = converted_hf + sft_sampled
    
    # Shuffle the mixed dataset
    print("Shuffling mixed dataset...")
    random.shuffle(mixed_dataset)
    
    # Save the mixed dataset
    print(f"Saving mixed dataset with {len(mixed_dataset)} samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mixed_dataset, f, ensure_ascii=False, indent=2)
    
    return len(mixed_dataset), len(converted_hf), len(sft_sampled)

def main():
    """Main function to mix datasets."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Mix Hugging Face and SFT datasets')
    parser.add_argument('--hf_count', type=int, default=5000, 
                       help='Number of samples to sample from HF dataset (default: 5000)')
    parser.add_argument('--sft_count', type=int, default=10000, 
                       help='Number of samples to sample from SFT dataset (default: 10000)')
    parser.add_argument('--output', type=str, default='mixed_dataset.json',
                       help='Output filename (default: mixed_dataset.json)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Paths (relative to script location)
    script_dir = Path(__file__).parent
    sft_file = script_dir / "combined_sft_dataset.json"
    output_file = script_dir / args.output
    
    print(f"Configuration:")
    print(f"  - HF samples: {args.hf_count}")
    print(f"  - SFT samples: {args.sft_count}")
    print(f"  - Output file: {output_file}")
    print(f"  - Random seed: {args.seed}")
    print()
    
    print("Loading Hugging Face dataset...")
    try:
        hf_dataset = load_dataset("open-thoughts/OpenThoughts-114k")
        hf_samples = list(hf_dataset['train'])  # Convert to list for sampling
        print(f"Loaded {len(hf_samples)} samples from HF dataset")
    except Exception as e:
        print(f"Error loading HF dataset: {e}")
        return
    
    print("Loading SFT dataset...")
    try:
        sft_samples = load_sft_dataset(sft_file)
        print(f"Loaded {len(sft_samples)} samples from SFT dataset")
    except Exception as e:
        print(f"Error loading SFT dataset: {e}")
        return
    
    # Validate sample counts
    if args.hf_count > len(hf_samples):
        print(f"Warning: Requested {args.hf_count} HF samples, but only {len(hf_samples)} available. Using all available.")
        args.hf_count = len(hf_samples)
    
    if args.sft_count > len(sft_samples):
        print(f"Warning: Requested {args.sft_count} SFT samples, but only {len(sft_samples)} available. Using all available.")
        args.sft_count = len(sft_samples)
    
    # Sample and mix datasets
    total_samples, actual_hf, actual_sft = sample_and_mix_datasets(
        hf_samples, 
        sft_samples, 
        args.hf_count, 
        args.sft_count, 
        output_file
    )
    
    print(f"\n✅ Successfully created mixed dataset!")
    print(f"📁 Output saved to: {output_file}")
    print(f"📊 Dataset composition:")
    print(f"   - HF samples: {actual_hf}")
    print(f"   - SFT samples: {actual_sft}")
    print(f"   - Total: {total_samples}")

if __name__ == "__main__":
    main()