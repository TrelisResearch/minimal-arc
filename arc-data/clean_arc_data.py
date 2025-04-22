import json
import os
from pathlib import Path
from collections import defaultdict

def clean_arc_data(input_file, output_file):
    """Clean ARC data by keeping only examples with single test inputs/outputs."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Track removed examples
    removed_examples = defaultdict(list)
    
    # Filter out examples with multiple tests
    cleaned_data = {}
    for task_id, task in data.items():
        # For challenge files
        if 'test' in task:
            if len(task['test']) == 1:
                cleaned_data[task_id] = task
            else:
                removed_examples['multiple_tests'].append((task_id, len(task['test'])))
        # For solution files (which are just lists of outputs)
        else:
            if len(task) == 1:
                cleaned_data[task_id] = task
            else:
                removed_examples['multiple_solutions'].append((task_id, len(task)))

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Print statistics
    print(f"\nProcessing results for {os.path.basename(input_file)}:")
    print(f"Original examples: {len(data)}")
    print(f"Cleaned examples: {len(cleaned_data)}")
    print(f"Total removed: {len(data) - len(cleaned_data)}")
    
    if removed_examples['multiple_tests']:
        print("\nExamples removed due to multiple tests:")
        for task_id, num_tests in removed_examples['multiple_tests']:
            print(f"  - Task {task_id}: {num_tests} tests")
    
    if removed_examples['multiple_solutions']:
        print("\nExamples removed due to multiple solutions:")
        for task_id, num_solutions in removed_examples['multiple_solutions']:
            print(f"  - Task {task_id}: {num_solutions} solutions")
    
    print("\n" + "="*50)
    
    return len(data), len(cleaned_data)

def main():
    # Create output directory
    output_dir = Path('arc-data-cleaned')
    output_dir.mkdir(exist_ok=True)
    
    # List of files to process
    files_to_clean = [
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        'arc-agi_training_challenges.json',
        'arc-agi_training_solutions.json',
        'arc-agi_test_challenges.json'
    ]
    
    # Track overall statistics
    total_stats = {
        'original': 0,
        'cleaned': 0
    }
    
    # Process each file
    for filename in files_to_clean:
        input_path = Path('arc-data') / filename
        output_path = output_dir / filename
        
        if input_path.exists():
            print(f"\nProcessing {filename}...")
            orig_count, cleaned_count = clean_arc_data(input_path, output_path)
            total_stats['original'] += orig_count
            total_stats['cleaned'] += cleaned_count
            print(f"Saved cleaned data to {output_path}")
        else:
            print(f"Warning: {filename} not found")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total original examples: {total_stats['original']}")
    print(f"Total cleaned examples: {total_stats['cleaned']}")
    print(f"Total removed: {total_stats['original'] - total_stats['cleaned']}")

if __name__ == "__main__":
    main()