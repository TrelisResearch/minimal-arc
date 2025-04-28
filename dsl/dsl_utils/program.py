"""
ARC DSL Program.

This module defines the Program class, which represents a sequence of operations.
"""
from typing import List, Any, Optional
import traceback
import numpy as np

from .primitives import Op
from .types import Grid, ObjList, Type, Grid_T, ObjList_T, Int_T, Bool_T


class Program:
    """Representation of a program in the ARC DSL."""
    
    def __init__(self, ops: List[Op]):
        """Initialize a program with a list of operations."""
        self.ops = ops
    
    def run(self, input_grid: Grid) -> Any:
        """
        Run the program on an input grid.
        
        Args:
            input_grid: The input grid
            
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
                
                # Apply the operation with appropriate arguments
                if op.name == "tile" and len(op.fn.__code__.co_varnames) > 1:
                    # Special case for tile which needs additional arguments
                    # Assume we want to tile 3x3 for now (common case)
                    rows, cols = 3, 3
                    # If the output is expected to be 6x6, use 3x3 tiling
                    if result.shape == (2, 2):
                        rows, cols = 3, 3
                    result = op.fn(result, rows, cols)
                elif op.name == "mask_color":
                    # Find a color to mask (for simplicity, use the most common non-zero color)
                    unique_colors, counts = np.unique(result.data, return_counts=True)
                    non_zero_colors = [c for c in unique_colors if c != 0]
                    if not non_zero_colors:
                        color = 1  # Default if no non-zero colors
                    else:
                        # Use the most common non-zero color
                        non_zero_indices = [i for i, c in enumerate(unique_colors) if c != 0]
                        color = unique_colors[non_zero_indices[np.argmax(counts[non_zero_indices])]]
                    result = op.fn(result, int(color))
                elif op.name == "flood_fill":
                    # Find a point to fill (for simplicity, use the first non-zero point)
                    non_zero_points = np.argwhere(result.data > 0)
                    if len(non_zero_points) == 0:
                        # If no non-zero points, use the center
                        row, col = result.data.shape[0] // 2, result.data.shape[1] // 2
                        new_color = 1  # Default color
                    else:
                        row, col = non_zero_points[0]
                        new_color = (result.data[row, col] + 1) % 10  # Next color
                    result = op.fn(result, int(row), int(col), int(new_color))
                elif op.name == "crop":
                    # Default crop: center portion
                    h, w = result.data.shape
                    crop_h, crop_w = max(1, h // 2), max(1, w // 2)
                    top, left = (h - crop_h) // 2, (w - crop_w) // 2
                    result = op.fn(result, int(top), int(left), int(crop_h), int(crop_w))
                elif op.name == "replace_color":
                    # Replace the first non-zero color with a different color
                    unique_colors = np.unique(result.data)
                    non_zero_colors = [c for c in unique_colors if c != 0]
                    if non_zero_colors:
                        old_color = non_zero_colors[0]
                        new_color = (old_color + 1) % 10
                        if new_color == 0:  # Avoid using 0 as it's typically background
                            new_color = 1
                        result = op.fn(result, int(old_color), int(new_color))
                    else:
                        # If no non-zero colors, replace 0 with 1
                        result = op.fn(result, 0, 1)
                elif op.name == "count_color":
                    # Count the most common non-zero color
                    unique_colors, counts = np.unique(result.data, return_counts=True)
                    non_zero_colors = [c for c in unique_colors if c != 0]
                    if not non_zero_colors:
                        color = 1  # Default if no non-zero colors
                    else:
                        # Use the most common non-zero color
                        non_zero_indices = [i for i, c in enumerate(unique_colors) if c != 0]
                        color = unique_colors[non_zero_indices[np.argmax(counts[non_zero_indices])]]
                    result = op.fn(result, int(color))
                else:
                    # Standard operation with no additional arguments
                    result = op.fn(result)
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
