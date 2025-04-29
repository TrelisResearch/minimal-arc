"""
Unified ARC DSL Runner.

This script provides a unified interface for running the DSL solver on either:
1. A single ARC task
2. Multiple ARC tasks from a dataset file

It eliminates duplication between run_task.py and run_dataset.py to ensure consistent behavior.
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
import matplotlib.pyplot as plt
from tqdm import tqdm

# Fix the import path issue
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Use correct imports for the new directory structure
from dsl.dsl_utils.primitives import ALL_PRIMITIVES, TILE_PATTERN, print_primitives_summary
from dsl.dsl_utils.types import Grid
from dsl.dsl_utils.program import Program, TimeoutException
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
    """Main entry point for the unified ARC runner."""
    parser = argparse.ArgumentParser(
        description='Run the ARC DSL solver on a single task or multiple tasks from a dataset'
    )
    
    # Create a mutually exclusive group for task_id vs. dataset_file
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument('--task-id', type=str, help='The task ID to solve')
    task_group.add_argument('--dataset-file', type=str, help='Path to the JSON file containing task IDs')
    
    # Common arguments
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth (default: 4)')
    parser.add_argument('--timeout', type=float, default=60.0, help='Search timeout in seconds (default: 60.0)')
    parser.add_argument('--op-timeout', type=float, default=0.25, help='Timeout for individual operations in seconds (default: 0.25)')
    parser.add_argument('--data-path', type=str, help='Path to the data directory')
    parser.add_argument('--save-dir', type=str, help='Directory to save results and visualizations')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    # Visualization options
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument('--show', action='store_true', help='Show visualization (for single task only)')
    viz_group.add_argument('--save-viz', action='store_true', help='Save visualizations to save-dir')
    
    # Parallel processing options
    parser.add_argument('--parallel', type=int, default=max(1, mp.cpu_count() - 1), 
                        help='Number of parallel processes (default: CPU count - 1)')
    
    # Dataset-specific options
    parser.add_argument('--results-file', type=str, help='File to save results (for dataset mode)')
    
    args = parser.parse_args()
    
    # Display welcome message with primitives summary
    print_primitives_summary()
    print()
    
    # Create the save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Create partial functions for the loaders
    task_loader = partial(load_task, data_path=args.data_path)
    solution_loader = partial(load_solution, data_path=args.data_path)
    train_pairs_loader = load_train_pairs
    test_input_loader = load_test_input
    
    # Determine if we're running in single task or dataset mode
    if args.task_id:
        # Single task mode
        task_ids = [args.task_id]
        print(f"Running single task: {args.task_id}")
        
        # For single task, we can use just one process
        num_processes = 1 if args.parallel <= 1 else 1
        
        # Set up visualization
        visualizer = None
        if args.show or args.save_viz:
            if args.save_dir:
                visualizer = save_visualization
    else:
        # Dataset mode
        # Load task IDs from the JSON file
        with open(args.dataset_file, 'r') as f:
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
        
        print(f"Loaded {len(task_ids)} task IDs from {args.dataset_file}")
        
        # Use the specified number of parallel processes
        num_processes = args.parallel
        
        # Set up visualization for dataset mode
        visualizer = save_visualization if args.save_viz and args.save_dir else None
    
    print(f"Running with {num_processes} parallel processes...")
    
    # Process tasks using the unified approach
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
        save_dir=args.save_dir if args.save_viz else None,
        visualizer=visualizer,
        debug=args.debug
    )
    
    # Process results
    if len(task_ids) == 1:
        # Single task mode - show detailed results
        result = results[0]
        task_id = result['task_id']
        
        if result['solved']:
            print(f"Task {task_id}: Solution found!")
            print(f"Program: {result['program']}")
            
            if 'correct' in result and result['correct']:
                print("Prediction matches solution!")
            elif 'correct' in result:
                print("Prediction does not match solution")
        else:
            if result.get('search_exhausted', False):
                print(f"Task {task_id}: No solution found - Search space exhausted (all programs up to depth {args.depth} tried)")
            elif result.get('search_timed_out', False):
                print(f"Task {task_id}: No solution found - Search timed out after {args.timeout} seconds")
            else:
                print(f"Task {task_id}: No solution found")
        
        # For single task mode with --show, display the visualization
        if args.show:
            task = task_loader(task_id)
            prediction = None
            
            if result['solved'] and 'prediction' in result:
                prediction = result['prediction']
            
            if prediction is not None:
                plt.figure(figsize=(12, 8))
                visualize_task(task, prediction)
                plt.tight_layout()
                plt.show()
    else:
        # Dataset mode - show summary statistics
        solved_count = sum(1 for result in results if result['solved'])
        correct_count = sum(1 for result in results if result.get('correct', False))
        
        print(f"Results: {solved_count}/{len(task_ids)} tasks solved")
        print(f"Correct: {correct_count}/{len(task_ids)} predictions match solutions")
        
        # Save results to a file if specified
        if args.results_file:
            results_path = args.results_file
            if not os.path.isabs(results_path) and args.save_dir:
                results_path = os.path.join(args.save_dir, results_path)
                
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
