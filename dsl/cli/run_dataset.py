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
from functools import partial
from tqdm import tqdm
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from dsl.dsl_utils.primitives import ALL_PRIMITIVES
from dsl.dsl_utils.types import Grid
from dsl.search.enumerator import iter_deepening
from dsl.search.verifier import verify, evaluate_program
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input, load_id_list
from dsl.io.visualizer import visualize_task, save_visualization


def solve_task(task_id, depth, timeout, data_path, save_dir, op_timeout):
    """
    Solve a single task.
    
    Args:
        task_id: The task ID
        depth: Maximum search depth
        timeout: Search timeout in seconds
        data_path: Path to the data directory
        save_dir: Directory to save results
        op_timeout: Timeout for individual operations in seconds
        
    Returns:
        Dictionary with results
    """
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
        found_solution = False
        valid_program = None
        prediction = None
        
        # Generate and verify programs
        for program in iter_deepening(ALL_PRIMITIVES, depth, input_shape, output_shape, timeout, True):
            try:
                if verify(program, train_pairs, op_timeout=op_timeout):
                    valid_program = program
                    found_solution = True
                    
                    # Generate prediction for the test input
                    prediction = evaluate_program(program, test_input, op_timeout=op_timeout)
                    break
            except TimeoutException:
                continue  # Skip programs that time out
            except Exception as e:
                continue  # Skip programs that raise exceptions
        
        # Check if the prediction matches the solution
        correct = False
        if solution is not None and prediction is not None:
            correct = np.array_equal(prediction.data, solution.data)
        
        # Save visualization if requested
        if save_dir and found_solution:
            save_path = os.path.join(save_dir, f"{task_id}.png")
            visualize_task(task, prediction, save_path)
        
        return {
            'task_id': task_id,
            'solved': found_solution,
            'correct': correct,
            'program': str(valid_program) if valid_program else None
        }
    except Exception as e:
        return {
            'task_id': task_id,
            'solved': False,
            'correct': False,
            'error': str(e)
        }


def evaluate_program(program, input_grid, op_timeout=0.25):
    """
    Evaluate a program on an input grid with timeout handling.
    
    Args:
        program: The program to evaluate
        input_grid: The input grid
        op_timeout: Timeout for individual operations in seconds
        
    Returns:
        The result of running the program, or None if an error occurs
    """
    try:
        return program.run(input_grid, op_timeout=op_timeout)
    except TimeoutException:
        return None
    except Exception as e:
        return None


def main():
    """Main entry point for the dataset runner."""
    parser = argparse.ArgumentParser(description='Run the ARC DSL solver on a dataset')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing task IDs')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth (default: 4)')
    parser.add_argument('--timeout', type=float, default=15.0, help='Search timeout in seconds (default: 15.0)')
    parser.add_argument('--parallel', type=int, help='Number of parallel processes (default: CPU count - 1)')
    parser.add_argument('--data-path', type=str, help='Path to the data directory')
    parser.add_argument('--save-dir', type=str, help='Directory to save results')
    parser.add_argument('--results-file', type=str, default='results.json', help='File to save results (default: results.json)')
    parser.add_argument('--op-timeout', type=float, default=0.25, help='Timeout for individual operations in seconds (default: 0.25)')
    
    args = parser.parse_args()
    
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
    if args.parallel is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    else:
        num_processes = args.parallel
    
    print(f"Running {len(task_ids)} tasks with {num_processes} parallel processes...")
    
    # Create the save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Run tasks in parallel
    with Pool(processes=num_processes) as pool:
        # Create a partial function with fixed arguments
        run_task_partial = partial(
            solve_task,
            depth=args.depth,
            timeout=args.timeout,
            data_path=args.data_path,
            save_dir=args.save_dir,
            op_timeout=args.op_timeout
        )
        
        # Map the function to the task IDs with a progress bar
        results = list(tqdm(
            pool.imap_unordered(run_task_partial, task_ids),
            total=len(task_ids),
            desc="Processing tasks"
        ))
    
    # Count successful tasks
    solved_count = sum(1 for result in results if result['solved'])
    correct_count = sum(1 for result in results if result['correct'])
    
    print(f"Results: {solved_count}/{len(task_ids)} tasks solved")
    print(f"Correct: {correct_count}/{len(task_ids)} predictions match solutions")
    
    # Save results to a file
    if args.save_dir:
        results_path = os.path.join(args.save_dir, args.results_file)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
