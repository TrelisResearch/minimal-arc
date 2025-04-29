"""
ARC DSL Task Solver.

This module provides a unified interface for solving ARC tasks,
which can be used by both individual task runners and dataset runners.
"""
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from dsl.dsl_utils.primitives import ALL_PRIMITIVES
from dsl.dsl_utils.types import Grid
from dsl.dsl_utils.program import Program, TimeoutException
from dsl.search.enumerator import iter_deepening
from dsl.search.verifier import verify


def solve_task(
    task_id: str,
    train_pairs: List[Tuple[Grid, Grid]],
    test_input: Grid,
    depth: int = 4,
    timeout: float = 15.0,
    op_timeout: float = 0.25,
    parallel: bool = True,
    num_processes: Optional[int] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Solve a single ARC task.
    
    Args:
        task_id: The task ID
        train_pairs: List of (input, output) pairs for training
        test_input: The test input grid
        depth: Maximum search depth
        timeout: Search timeout in seconds
        op_timeout: Timeout for individual operations in seconds
        parallel: Whether to use parallel search
        num_processes: Number of processes to use for parallel search
        debug: Whether to print debug information
        
    Returns:
        Dictionary with results:
        - solved: Whether a solution was found
        - program: The program that solves the task (if found)
        - prediction: The prediction for the test input (if available)
        - elapsed_time: Time taken to find the solution
        - search_exhausted: Whether the search space was exhausted
        - search_timed_out: Whether the search timed out
    """
    # Get shapes for heuristics
    input_shape = train_pairs[0][0].shape
    output_shape = train_pairs[0][1].shape
    
    # Extract training inputs for grid state memoization
    train_inputs = [pair[0] for pair in train_pairs]
    
    if debug:
        print(f"Input shape: {input_shape}, Output shape: {output_shape}")
        print(f"Searching for programs with max depth {depth}...")
    
    # Start the search
    start_time = time.time()
    found_solution = False
    valid_program = None
    prediction = None
    
    # Use a more reliable timeout approach
    end_time = start_time + timeout
    
    # Track whether search was exhausted or timed out
    search_exhausted = False
    search_timed_out = False
    
    # Check if we're running in a multiprocessing context
    # If we are, we should disable parallel search to avoid nested parallelism
    try:
        import multiprocessing
        current_process = multiprocessing.current_process()
        if current_process.daemon:
            # We're in a daemon process, so disable parallel search
            parallel = False
            if debug:
                print("Running in daemon process, disabling parallel search")
    except (ImportError, AttributeError):
        # multiprocessing not available or we're not in a multiprocessing context
        pass
    
    # Generate and verify programs
    try:
        # Create a fresh visited dictionary for this task
        visited = {}
        
        iterator = iter_deepening(ALL_PRIMITIVES, depth, input_shape, output_shape, timeout, 
                                parallel, num_processes, train_inputs=train_pairs[0][0:1], op_timeout=op_timeout,
                                visited=visited)
        
        while True:
            try:
                result = next(iterator)
                program, metadata = result
                
                # Check if this is a status update rather than a program
                if program is None:
                    search_exhausted = metadata.get("search_exhausted", False)
                    search_timed_out = metadata.get("search_timed_out", False)
                    if search_exhausted and debug:
                        print(f"Search space exhausted (all programs up to depth {depth} tried)")
                    if search_timed_out and debug:
                        print(f"Search timed out after {timeout} seconds")
                    break
                
                # Check if we've exceeded the timeout
                current_time = time.time()
                if current_time > end_time:
                    if debug:
                        print(f"Search timed out after {timeout} seconds")
                    search_timed_out = True
                    break
                    
                try:
                    if verify(program, train_pairs, op_timeout=op_timeout):
                        valid_program = program
                        found_solution = True
                        
                        if debug:
                            print(f"Found valid program: {program}")
                        
                        # Generate prediction for the test input
                        try:
                            prediction = program.run(test_input, op_timeout=op_timeout)
                            if debug and prediction is not None:
                                print(f"Generated prediction for test input")
                            elif debug:
                                print(f"Failed to generate prediction for test input")
                        except TimeoutException:
                            if debug:
                                print("Operation timed out during prediction")
                        except Exception as e:
                            if debug:
                                print(f"Error during prediction: {e}")
                        
                        break
                except TimeoutException:
                    if debug:
                        print(f"Program timed out during verification: {program}")
                    continue
                except Exception as e:
                    if debug:
                        print(f"Error during verification: {e}")
                    continue
            
            except StopIteration:
                # End of iterator
                break
    
    except KeyboardInterrupt:
        if debug:
            print("Search interrupted by user")
    except TimeoutException:
        if debug:
            print(f"Search timed out after {timeout} seconds")
        search_timed_out = True
    except Exception as e:
        if debug:
            print(f"Unexpected error during search: {e}")
    
    elapsed_time = time.time() - start_time
    
    if debug:
        print(f"Search completed in {elapsed_time:.2f} seconds")
        if not found_solution:
            if search_exhausted:
                print(f"No solution found: Search space exhausted (all programs up to depth {depth} tried)")
            elif search_timed_out:
                print(f"No solution found: Search timed out after {timeout} seconds")
            else:
                print("No solution found")
    
    return {
        'task_id': task_id,
        'solved': found_solution,
        'program': valid_program,
        'prediction': prediction,
        'elapsed_time': elapsed_time,
        'search_exhausted': search_exhausted,
        'search_timed_out': search_timed_out
    }


def evaluate_program(program: Program, input_grid: Grid, op_timeout: float = 0.25) -> Optional[Grid]:
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
    except Exception:
        return None
