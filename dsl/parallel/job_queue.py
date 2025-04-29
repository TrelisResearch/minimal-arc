"""
ARC DSL Job Queue.

This module implements a process-based parallelism approach with a job queue
for solving multiple ARC tasks efficiently.
"""
import time
import multiprocessing as mp
from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
from tqdm import tqdm

from dsl.dsl_utils.types import Grid
from dsl.dsl_utils.program import Program
from dsl.search.enumerator import iter_deepening
from dsl.search.verifier import verify
from dsl.dsl_utils.primitives import ALL_PRIMITIVES

# Define a context manager for timing out operations
class TimeoutException(Exception):
    pass

class time_limit:
    def __init__(self, limit):
        self.limit = limit

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if time.time() - self.start_time > self.limit:
            raise TimeoutException("Operation timed out")


def worker_process(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
    task_loader: Callable,
    solution_loader: Callable,
    train_pairs_loader: Callable,
    test_input_loader: Callable,
    depth: int,
    timeout: float,
    op_timeout: float,
    save_dir: Optional[str] = None,
    visualizer: Optional[Callable] = None,
    debug: bool = False
):
    """
    Worker process function that processes jobs from the queue.
    
    Args:
        job_queue: Queue of task IDs to process
        result_queue: Queue to put results in
        task_loader: Function to load a task from its ID
        solution_loader: Function to load a solution from a task ID
        train_pairs_loader: Function to load training pairs from a task
        test_input_loader: Function to load test input from a task
        depth: Maximum search depth
        timeout: Search timeout in seconds
        op_timeout: Timeout for individual operations in seconds
        save_dir: Directory to save visualizations (optional)
        visualizer: Function to visualize results (optional)
        debug: Whether to print debug information
    """
    while True:
        try:
            # Get a job from the queue
            task_id = job_queue.get(block=False)
        except Exception:
            # No more jobs
            break
            
        try:
            # Load the task
            task = task_loader(task_id)
            
            # Extract training pairs
            train_pairs = train_pairs_loader(task)
            test_input = test_input_loader(task)
            
            # Try to load the solution
            solution = None
            try:
                solution_grid = solution_loader(task_id)
                solution = Grid(solution_grid)
                if debug:
                    print(f"Loaded solution for task {task_id}: {solution.shape}")
            except Exception as e:
                if debug:
                    print(f"Failed to load solution for task {task_id}: {e}")
            
            # Get shapes for heuristics
            input_shape = train_pairs[0][0].shape
            output_shape = train_pairs[0][1].shape
            
            # Start the search
            start_time = time.time()
            end_time = start_time + timeout
            
            found_solution = False
            valid_program = None
            prediction = None
            search_exhausted = False
            search_timed_out = False
            
            # Only use the first training input for memoization to reduce memory usage
            # Create a fresh visited dictionary for each task to prevent contamination
            first_input = [train_pairs[0][0]] if train_pairs else None
            
            # Generate and verify programs
            try:
                visited = {}  # Create a fresh visited dictionary for each task
                for result in iter_deepening(ALL_PRIMITIVES, depth, input_shape, output_shape, timeout, 
                                           parallel=False, train_inputs=first_input, op_timeout=op_timeout, visited=visited):
                    program, metadata = result
                    
                    # Check if this is a status update rather than a program
                    if program is None:
                        search_exhausted = metadata.get("search_exhausted", False)
                        search_timed_out = metadata.get("search_timed_out", False)
                        if search_exhausted and debug:
                            print(f"Task {task_id}: Search space exhausted (all programs up to depth {depth} tried)")
                        if search_timed_out and debug:
                            print(f"Task {task_id}: Search timed out after {timeout} seconds")
                        break
                    
                    # Check if we've exceeded the timeout
                    current_time = time.time()
                    if current_time > end_time:
                        search_timed_out = True
                        if debug:
                            print(f"Task {task_id}: Search timed out after {timeout} seconds")
                        break
                        
                    try:
                        with time_limit(op_timeout * 2):  # Give verification a bit more time
                            if verify(program, train_pairs, op_timeout=op_timeout):
                                valid_program = program
                                found_solution = True
                                
                                # Generate prediction for the test input
                                try:
                                    with time_limit(op_timeout * 2):  # Give prediction a bit more time
                                        prediction = program.run(test_input, op_timeout=op_timeout)
                                except TimeoutException:
                                    prediction = None
                                except Exception:
                                    prediction = None
                                
                                break
                    except TimeoutException:
                        continue
                    except Exception:
                        continue
            except TimeoutException:
                search_timed_out = True
                if debug:
                    print(f"Task {task_id}: Search timed out after {timeout} seconds")
            except StopIteration:
                # End of iterator
                pass
            except Exception as e:
                if debug:
                    print(f"Task {task_id}: Unexpected error during search: {e}")
            
            # If no prediction was generated, report why
            if not found_solution and debug:
                if search_exhausted:
                    print(f"Task {task_id}: No prediction generated - Search space exhausted (all programs up to depth tried)")
                elif search_timed_out:
                    print(f"Task {task_id}: No prediction generated - Search timed out after {timeout} seconds")
                else:
                    print(f"Task {task_id}: No prediction generated")
            
            elapsed_time = time.time() - start_time
            
            # Check if the prediction matches the solution
            correct = False
            if solution is not None and prediction is not None:
                # Handle different array shapes by comparing the actual grid data
                solution_data = solution.data
                prediction_data = prediction.data
                
                # If solution has an extra dimension (e.g., (1, 4, 4)), remove it
                if solution_data.ndim > 2 and solution_data.shape[0] == 1:
                    solution_data = solution_data[0]
                
                # Compare the actual grid data
                correct = np.array_equal(prediction_data, solution_data)
                
                if debug:
                    print(f"Task {task_id}: Prediction {'matches' if correct else 'does not match'} solution")
                    if not correct:
                        print(f"Prediction shape: {prediction.shape}, Solution shape: {solution.shape}")
                        print(f"Prediction data: {prediction_data.tolist()}")
                        print(f"Solution data: {solution_data.tolist()}")
            elif debug:
                if solution is None:
                    print(f"Task {task_id}: No solution available for comparison")
                if prediction is None:
                    print(f"Task {task_id}: No prediction generated")
            
            # Save visualization if requested
            if save_dir and found_solution and visualizer:
                try:
                    visualizer(task, prediction, save_dir, task_id)
                except Exception:
                    pass
            
            # Put the result in the result queue
            result_queue.put({
                'task_id': task_id,
                'solved': found_solution,
                'correct': correct,
                'program': str(valid_program) if valid_program else None,
                'elapsed_time': elapsed_time,
                'search_exhausted': search_exhausted,
                'search_timed_out': search_timed_out
            })
        except Exception as e:
            # Put an error result in the result queue
            result_queue.put({
                'task_id': task_id,
                'solved': False,
                'correct': False,
                'error': str(e),
                'elapsed_time': 0,
                'search_exhausted': False,
                'search_timed_out': False
            })


def process_tasks_parallel(
    task_ids: List[str],
    task_loader: Callable,
    solution_loader: Callable,
    train_pairs_loader: Callable,
    test_input_loader: Callable,
    depth: int,
    timeout: float,
    op_timeout: float,
    num_processes: int,
    save_dir: Optional[str] = None,
    visualizer: Optional[Callable] = None,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """
    Process multiple tasks in parallel using a job queue.
    
    Args:
        task_ids: List of task IDs to process
        task_loader: Function to load a task from its ID
        solution_loader: Function to load a solution from a task ID
        train_pairs_loader: Function to load training pairs from a task
        test_input_loader: Function to load test input from a task
        depth: Maximum search depth
        timeout: Search timeout in seconds
        op_timeout: Timeout for individual operations in seconds
        num_processes: Number of worker processes to use
        save_dir: Directory to save visualizations (optional)
        visualizer: Function to visualize results (optional)
        debug: Whether to print debug information
        
    Returns:
        List of result dictionaries, one for each task
    """
    # Create the queues
    job_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Put all jobs in the queue
    for task_id in task_ids:
        job_queue.put(task_id)
    
    # Start the worker processes
    processes = []
    for _ in range(num_processes):
        p = mp.Process(
            target=worker_process,
            args=(
                job_queue,
                result_queue,
                task_loader,
                solution_loader,
                train_pairs_loader,
                test_input_loader,
                depth,
                timeout,
                op_timeout,
                save_dir,
                visualizer,
                debug
            )
        )
        p.start()
        processes.append(p)
    
    # Collect results with a progress bar
    results = []
    with tqdm(total=len(task_ids), desc="Processing tasks") as pbar:
        while len(results) < len(task_ids):
            try:
                result = result_queue.get(timeout=0.1)
                results.append(result)
                pbar.update(1)
            except Exception:
                # Check if all processes are done
                if all(not p.is_alive() for p in processes):
                    break
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    return results
