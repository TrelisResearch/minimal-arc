"""
ARC DSL Dataset Runner.

This script runs the DSL solver on multiple ARC tasks from a dataset file.
"""
import argparse
import time
import sys
import os
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from dsl.dsl_utils.primitives import ALL_PRIMITIVES, print_primitives_summary
from dsl.dsl_utils.types import Grid
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input
from dsl.io.visualizer import visualize_task
from dsl.parallel.job_queue import process_tasks_parallel


def save_visualization(task, prediction, save_dir, task_id):
    """
    Save a visualization of the task and prediction.
    
    Args:
        task: The task data
        prediction: The prediction grid
        save_dir: Directory to save the visualization
        task_id: The task ID
    """
    save_path = os.path.join(save_dir, f"{task_id}.png")
    visualize_task(task, prediction, save_path)


def main():
    """Main entry point for the dataset runner."""
    parser = argparse.ArgumentParser(description='Run the ARC DSL solver on a dataset')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing task IDs')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth (default: 4)')
    parser.add_argument('--timeout', type=float, default=3.0, help='Search timeout in seconds (default: 15.0)')
    parser.add_argument('--parallel', type=int, default=mp.cpu_count() - 1, help='Number of parallel processes (default: CPU count - 1)')
    parser.add_argument('--data-path', type=str, help='Path to the data directory')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    parser.add_argument('--results-file', type=str, default='results.json', help='File to save results (default: results.json)')
    parser.add_argument('--op-timeout', type=float, default=0.25, help='Timeout for individual operations in seconds (default: 0.25)')
    
    args = parser.parse_args()
    
    # Display welcome message with primitives summary
    print_primitives_summary()
    print()
    
    # Load task IDs from the JSON file
    with open(args.json_file, 'r') as f:
        data = json.load(f)
    
    # Extract task IDs
    if isinstance(data, list):
        # JSON file contains a list of task IDs
        task_ids = data
    elif isinstance(data, dict) and 'tasks' in data:
        # JSON file contains a dictionary with a 'tasks' key
        task_ids = data['tasks']
    else:
        # Try to extract keys from the dictionary
        task_ids = list(data.keys())
    
    print(f"Loaded {len(task_ids)} task IDs from {args.json_file}")
    
    # Set up parallel processing
    num_processes = args.parallel
    
    print(f"Running {len(task_ids)} tasks with {num_processes} parallel processes...")
    
    # Create the save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create partial functions for the loaders
    task_loader = partial(load_task, data_path=args.data_path)
    solution_loader = partial(load_solution, data_path=args.data_path)
    train_pairs_loader = load_train_pairs
    test_input_loader = load_test_input
    
    # Add debug flag
    debug = True
    
    # Process tasks in parallel using the job queue
    results = process_tasks_parallel(
        task_ids=task_ids,
        task_loader=task_loader,
        solution_loader=solution_loader,
        train_pairs_loader=train_pairs_loader,
        test_input_loader=test_input_loader,
        depth=args.depth,
        timeout=args.timeout,
        op_timeout=args.op_timeout,
        num_processes=num_processes,
        save_dir=args.save_dir,
        visualizer=save_visualization if args.save_dir else None,
        debug=debug
    )
    
    # Count successful tasks
    solved_count = sum(1 for result in results if result['solved'])
    correct_count = sum(1 for result in results if result.get('correct', False))
    
    print(f"Results: {solved_count}/{len(task_ids)} tasks solved")
    print(f"Correct: {correct_count}/{len(task_ids)} predictions match solutions")
    
    # Save results to a file
    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.results_file}")


if __name__ == '__main__':
    main()
