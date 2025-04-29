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
    """
    # Get shapes for heuristics
    input_shape = train_pairs[0][0].shape
    output_shape = train_pairs[0][1].shape
    
    if debug:
        print(f"Input shape: {input_shape}, Output shape: {output_shape}")
        print(f"Searching for programs with max depth {depth}...")
    
    # Start the search
    start_time = time.time()
    found_solution = False
    valid_program = None
    prediction = None
    
    # Generate and verify programs
    try:
        for program in iter_deepening(ALL_PRIMITIVES, depth, input_shape, output_shape, timeout, 
                                    parallel, num_processes):
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
    except KeyboardInterrupt:
        if debug:
            print("Search interrupted by user")
    
    elapsed_time = time.time() - start_time
    
    if debug:
        print(f"Search completed in {elapsed_time:.2f} seconds")
        if not found_solution:
            print("No solution found")
    
    return {
        'task_id': task_id,
        'solved': found_solution,
        'program': valid_program,
        'prediction': prediction,
        'elapsed_time': elapsed_time
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
