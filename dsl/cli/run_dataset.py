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
from multiprocessing import Pool
from tqdm import tqdm

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from dsl.dsl_utils.primitives import ALL_PRIMITIVES
from dsl.dsl_utils.types import Grid
from dsl.search.enumerator import iter_deepening
from dsl.search.verifier import verify, evaluate_program
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input, load_id_list
from dsl.io.visualizer import visualize_task, save_visualization


def solve_task(args):
    """
    Solve a single task.
    
    Args:
        args: Tuple of (task_id, depth, timeout, data_path, save_dir)
        
    Returns:
        Dictionary with results
    """
    task_id, depth, timeout, data_path, save_dir = args
    
    try:
        # Load the task
        task = load_task(task_id, data_path)
        
        # Extract training pairs
        train_pairs = load_train_pairs(task)
        test_input = load_test_input(task)
        
        # Try to load the solution
        solution = None
        try:
            solution_grid = load_solution(task_id, data_path)
            solution = Grid(solution_grid)
        except Exception:
            pass
        
        # Get shapes for heuristics
        input_shape = train_pairs[0][0].shape
        output_shape = train_pairs[0][1].shape
        
        # Start the search
        start_time = time.time()
        found_solution = False
        valid_program = None
        prediction = None
        
        # Generate and verify programs
        for program in iter_deepening(ALL_PRIMITIVES, depth, input_shape, output_shape, timeout):
            if verify(program, train_pairs):
                valid_program = program
                found_solution = True
                
                # Generate prediction for the test input
                prediction = evaluate_program(program, test_input)
                break
        
        elapsed_time = time.time() - start_time
        
        # Check if the prediction matches the solution
        correct = False
        if solution is not None and prediction is not None:
            correct = prediction == solution
        
        # Save visualization if requested
        if save_dir and found_solution:
            os.makedirs(save_dir, exist_ok=True)
            fig = visualize_task(task, prediction, solution)
            save_visualization(fig, os.path.join(save_dir, f"{task_id}.png"))
        
        return {
            'task_id': task_id,
            'found_solution': found_solution,
            'program': str(valid_program) if valid_program else None,
            'correct': correct,
            'time': elapsed_time
        }
    
    except Exception as e:
        return {
            'task_id': task_id,
            'error': str(e),
            'found_solution': False,
            'correct': False,
            'time': 0
        }


def main():
    """Main entry point for the dataset runner."""
    parser = argparse.ArgumentParser(description='Run the ARC DSL solver on multiple tasks')
    parser.add_argument('json_file', type=str, help='JSON file with task IDs')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth (default: 4)')
    parser.add_argument('--timeout', type=float, default=15.0, help='Search timeout in seconds per task (default: 15.0)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel processes (default: 1)')
    parser.add_argument('--data-path', type=str, help='Path to the data directory')
    parser.add_argument('--save-dir', type=str, help='Directory to save visualizations')
    parser.add_argument('--results-file', type=str, help='File to save results')
    
    args = parser.parse_args()
    
    # Load the task IDs
    task_ids = load_id_list(args.json_file)
    print(f"Loaded {len(task_ids)} task IDs from {args.json_file}")
    
    # Prepare arguments for parallel processing
    process_args = [(task_id, args.depth, args.timeout, args.data_path, args.save_dir) for task_id in task_ids]
    
    # Run the tasks
    results = []
    if args.parallel > 1:
        print(f"Running {len(task_ids)} tasks with {args.parallel} parallel processes...")
        with Pool(args.parallel) as pool:
            results = list(tqdm(pool.imap(solve_task, process_args), total=len(task_ids)))
    else:
        print(f"Running {len(task_ids)} tasks sequentially...")
        for task_arg in tqdm(process_args):
            results.append(solve_task(task_arg))
    
    # Summarize results
    solved_count = sum(1 for r in results if r['found_solution'])
    correct_count = sum(1 for r in results if r['correct'])
    
    print(f"Results: {solved_count}/{len(task_ids)} tasks solved")
    print(f"Correct: {correct_count}/{len(task_ids)} predictions match solutions")
    
    # Save results if requested
    if args.results_file:
        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.results_file}")


if __name__ == '__main__':
    main()
