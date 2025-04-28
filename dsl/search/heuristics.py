"""
ARC DSL Search Heuristics.

This module implements heuristics for pruning the search space.
"""
from typing import List, Tuple, Optional, Set
import numpy as np
import time
import signal
from contextlib import contextmanager

from ..dsl_utils.primitives import Op
from ..dsl_utils.program import Program
from ..dsl_utils.types import Grid


class TimeoutException(Exception):
    """Exception raised when a timeout occurs."""
    pass


@contextmanager
def timeout(seconds: float):
    """
    Context manager that raises a TimeoutException after the specified number of seconds.
    
    Args:
        seconds: The timeout in seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    try:
        yield
    finally:
        # Cancel the timeout
        signal.setitimer(signal.ITIMER_REAL, 0)


def run_with_timeout(program: Program, grid: Grid, timeout_sec: float = 0.2) -> Optional[Grid]:
    """
    Run a program with a timeout.
    
    Args:
        program: The program to run
        grid: The input grid
        timeout_sec: The timeout in seconds
        
    Returns:
        The result grid or None if the execution timed out
    """
    try:
        with timeout(timeout_sec):
            return program.run(grid)
    except TimeoutException:
        return None
    except Exception as e:
        # Handle other exceptions that might occur during execution
        print(f"Error executing program: {e}")
        return None


def type_check(program: Program) -> bool:
    """
    Check if a program's operation types are compatible.
    
    Args:
        program: The program to check
        
    Returns:
        True if the types are compatible, False otherwise
    """
    return program.types_ok()


def symmetry_prune(ops: List[Op]) -> bool:
    """
    Check if a sequence of operations contains redundant patterns.
    
    Args:
        ops: The sequence of operations
        
    Returns:
        True if the sequence should be pruned, False otherwise
    """
    if len(ops) < 2:
        return False
    
    # Check for consecutive involutions
    for i in range(len(ops) - 1):
        if ops[i].name == ops[i+1].name and ops[i].name in ops[i].commutes_with:
            return True
    
    # Check for rotation patterns
    if len(ops) >= 3:
        # Three consecutive 90-degree rotations (equivalent to a single 270-degree rotation)
        if all(op.name == "rot90" for op in ops[-3:]):
            return True
        
        # Three consecutive 270-degree rotations (equivalent to a single 90-degree rotation)
        if all(op.name == "rot270" for op in ops[-3:]):
            return True
        
        # Full 360-degree rotation
        if len(ops) >= 4 and all(op.name == "rot90" for op in ops[-4:]):
            return True
        if len(ops) >= 4 and all(op.name == "rot270" for op in ops[-4:]):
            return True
        
        # Consecutive flips in the same direction
        if len(ops) >= 2 and ops[-1].name == "flip_h" and ops[-2].name == "flip_h":
            return True
        if len(ops) >= 2 and ops[-1].name == "flip_v" and ops[-2].name == "flip_v":
            return True
    
    return False


def shape_heuristic(input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> Set[str]:
    """
    Use shape information to determine which operations are likely to be useful.
    
    Args:
        input_shape: The shape of the input grid
        output_shape: The shape of the expected output grid
        
    Returns:
        A set of operation names that are likely to be useful
    """
    useful_ops = set()
    
    # If shapes match, transformations are likely useful
    if input_shape == output_shape:
        useful_ops.update(["rot90", "rot180", "rot270", "flip_h", "flip_v", "transpose"])
    
    # If output is larger, tiling is likely needed
    if output_shape[0] > input_shape[0] or output_shape[1] > input_shape[1]:
        useful_ops.add("tile")
    
    # If output is smaller, cropping might be needed
    if output_shape[0] < input_shape[0] or output_shape[1] < input_shape[1]:
        useful_ops.add("crop")
    
    # If dimensions are swapped, transpose might be useful
    if input_shape[0] == output_shape[1] and input_shape[1] == output_shape[0]:
        useful_ops.add("transpose")
    
    # If no specific heuristic applies, allow all operations
    if not useful_ops:
        return set()  # Empty set means no restrictions
    
    return useful_ops


def similarity_heuristic(grid1: Grid, grid2: Grid) -> float:
    """
    Compute a similarity score between two grids.
    
    Args:
        grid1: The first grid
        grid2: The second grid
        
    Returns:
        A similarity score between 0 and 1, where 1 means identical
    """
    # If shapes don't match, similarity is low
    if grid1.shape != grid2.shape:
        return 0.0
    
    # Compute the percentage of matching cells
    total_cells = grid1.data.size
    matching_cells = np.sum(grid1.data == grid2.data)
    
    return matching_cells / total_cells
