"""
ARC DSL Program.

This module defines the Program class, which represents a sequence of operations.
"""
from typing import List, Any, Optional
import traceback
import numpy as np
import time
import signal
from contextlib import contextmanager

from .primitives import Op
from .types import Grid, ObjList, Type, Grid_T, ObjList_T, Int_T, Bool_T


class TimeoutException(Exception):
    """Exception raised when a program execution times out."""
    pass


@contextmanager
def operation_timeout(seconds: float):
    """
    Context manager to limit the execution time of an operation.
    
    Args:
        seconds: The timeout in seconds
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Operation timed out")
    
    # Set the timeout handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    
    try:
        yield
    finally:
        # Reset the alarm
        signal.setitimer(signal.ITIMER_REAL, 0)


class Program:
    """Representation of a program in the ARC DSL."""
    
    def __init__(self, ops: List[Op]):
        """Initialize a program with a list of operations."""
        self.ops = ops
    
    def run(self, input_grid: Grid, op_timeout: float = 0.25) -> Any:
        """
        Run the program on an input grid.
        
        Args:
            input_grid: The input grid
            op_timeout: Timeout for each operation in seconds
            
        Returns:
            The result of running the program
        """
        result = input_grid
        
        try:
            for op in self.ops:
                # Check if the operation expects a Grid
                if op.in_type == Grid_T and not isinstance(result, Grid):
                    print(f"Error: Operation {op.name} expects a Grid, but got {type(result)}")
                    return None
                
                # Check if the operation expects an ObjList
                if op.in_type == ObjList_T and not isinstance(result, ObjList):
                    print(f"Error: Operation {op.name} expects an ObjList, but got {type(result)}")
                    return None
                
                # Apply a timeout to each operation
                with operation_timeout(op_timeout):
                    # Apply the operation - all operations are now unary (take only one argument)
                    result = op.fn(result)
        except TimeoutException:
            raise TimeoutException(f"Program execution timed out at operation: {op.name}")
        except Exception as e:
            print(f"Error executing program: {str(e)}")
            return None
        
        return result
    
    def is_compatible(self, in_type: Type, out_type: Type) -> bool:
        """
        Check if the program is compatible with the given input and output types.
        
        Args:
            in_type: The input type
            out_type: The output type
            
        Returns:
            True if the program is compatible, False otherwise
        """
        if not self.ops:
            return False
        
        # Check if the first operation accepts the input type
        if self.ops[0].in_type != in_type:
            return False
        
        # Check if the last operation produces the output type
        if self.ops[-1].out_type != out_type:
            return False
        
        # Check if the operations are compatible with each other
        for i in range(1, len(self.ops)):
            if self.ops[i].in_type != self.ops[i-1].out_type:
                return False
        
        return True
    
    def __repr__(self) -> str:
        """String representation of the program."""
        return f"Program({', '.join(op.name for op in self.ops)})"
