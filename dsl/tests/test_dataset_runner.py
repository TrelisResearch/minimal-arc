"""
Test script to verify dataset runner with our fixes.
"""
import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from dsl.parallel.job_queue import process_tasks_parallel
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input
from functools import partial

def main():
    """Test the dataset runner with a small set of tasks."""
    # Use the tasks that were previously failing
    task_ids = ["1a2e2828", "3194b014"]
    
    # Set up data path
    data_path = os.path.join(project_root, "arc-data-cleaned")
    
    # Create partial functions for the loaders
    task_loader = partial(load_task, data_path=data_path)
    solution_loader = partial(load_solution, data_path=data_path)
    train_pairs_loader = load_train_pairs
    test_input_loader = load_test_input
    
    # Process tasks with our fixed implementation
    print(f"Testing dataset runner with {len(task_ids)} tasks...")
    results = process_tasks_parallel(
        task_ids=task_ids,
        task_loader=task_loader,
        solution_loader=solution_loader,
        train_pairs_loader=train_pairs_loader,
        test_input_loader=test_input_loader,
        depth=4,
        timeout=5.0,
        op_timeout=0.25,
        num_processes=1,  # Use single process for easier debugging
        debug=True
    )
    
    # Print results
    print("\nResults summary:")
    for result in results:
        task_id = result['task_id']
        solved = result['solved']
        correct = result.get('correct', False)
        print(f"Task {task_id}: Solved: {solved}, Correct: {correct}")
    
    # Count successful tasks
    solved_count = sum(1 for result in results if result['solved'])
    correct_count = sum(1 for result in results if result.get('correct', False))
    
    print(f"\nOverall: {solved_count}/{len(task_ids)} tasks solved")
    print(f"Correct: {correct_count}/{len(task_ids)} predictions match solutions")

if __name__ == "__main__":
    main()
