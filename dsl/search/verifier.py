"""
ARC DSL Program Verifier.

This module implements verification of programs against training examples.
"""
from typing import List, Tuple, Dict, Optional, Any
import time
import numpy as np

from ..dsl_utils.program import Program, TimeoutException
from ..dsl_utils.types import Grid
from .heuristics import run_with_timeout


def verify(program: Program, examples: List[Tuple[Grid, Grid]], op_timeout: float = 0.25) -> bool:
    """
    Verify that a program produces the expected output for all examples.
    
    Args:
        program: The program to verify
        examples: List of (input, expected_output) pairs
        op_timeout: Timeout for individual operations in seconds
        
    Returns:
        True if the program produces the expected output for all examples, False otherwise
    """
    for input_grid, expected_output in examples:
        try:
            # Run the program on the input
            actual_output = program.run(input_grid, op_timeout=op_timeout)
            
            # Check if the output matches the expected output
            if actual_output is None or not np.array_equal(actual_output.data, expected_output.data):
                return False
        except TimeoutException:
            # If the program times out, it's not valid
            return False
        except Exception as e:
            # If the program raises an exception, it's not valid
            return False
    
    return True


def batch_verify(programs: List[Program], train_pairs: List[Tuple[Grid, Grid]], 
                timeout_sec: float = 0.2) -> List[bool]:
    """
    Verify multiple programs against training examples.
    
    Args:
        programs: List of programs to verify
        train_pairs: List of (input, expected_output) pairs
        timeout_sec: Timeout for each program execution in seconds
        
    Returns:
        List of boolean results (True if program passes all examples)
    """
    results = []
    
    for program in programs:
        results.append(verify(program, train_pairs, timeout_sec))
    
    return results


def find_valid_program(programs: List[Program], train_pairs: List[Tuple[Grid, Grid]], 
                      timeout_sec: float = 0.2) -> Optional[Program]:
    """
    Find the first valid program in a list.
    
    Args:
        programs: List of programs to check
        train_pairs: List of (input, expected_output) pairs
        timeout_sec: Timeout for each program execution in seconds
        
    Returns:
        The first valid program, or None if no valid program is found
    """
    for program in programs:
        if verify(program, train_pairs, timeout_sec):
            return program
    
    return None


def evaluate_program(program: Program, test_input: Grid, timeout_sec: float = 0.2) -> Optional[Grid]:
    """
    Evaluate a program on a test input.
    
    Args:
        program: The program to evaluate
        test_input: The test input grid
        timeout_sec: Timeout for program execution in seconds
        
    Returns:
        The result grid, or None if execution failed
    """
    return run_with_timeout(program, test_input, timeout_sec)


def solve_task(task: Dict[str, Any], programs: List[Program], 
              timeout_sec: float = 0.2) -> Tuple[Optional[Program], Optional[Grid]]:
    """
    Find a program that solves a task and apply it to the test input.
    
    Args:
        task: The task dictionary with 'train' and 'test' keys
        programs: List of candidate programs
        timeout_sec: Timeout for each program execution in seconds
        
    Returns:
        A tuple of (valid_program, test_prediction) or (None, None) if no solution is found
    """
    # Extract training pairs
    train_pairs = []
    for example in task['train']:
        inp = Grid(example['input'])
        out = Grid(example['output'])
        train_pairs.append((inp, out))
    
    # Find a valid program
    valid_program = find_valid_program(programs, train_pairs, timeout_sec)
    
    if valid_program is None:
        return None, None
    
    # Apply the program to the test input
    test_input = Grid(task['test'][0]['input'])
    test_prediction = evaluate_program(valid_program, test_input, timeout_sec)
    
    return valid_program, test_prediction
