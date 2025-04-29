"""
Color utilities for the ARC DSL.

This module provides functions for manipulating colors in ARC grids.
"""
import numpy as np
from typing import Tuple, Dict


def normalise_palette(grid: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Normalize the palette of a grid.
    
    Converts the grid so that 0 stays 0, and the first non-zero color becomes 1,
    the next becomes 2, and so on.
    
    Args:
        grid: The input grid
        
    Returns:
        A tuple of (normalized_grid, forward_mapping)
    """
    # Create a copy of the grid to modify
    norm = grid.copy()
    
    # Get unique colors and sort them
    unique_colors = sorted(np.unique(grid))
    
    # Create mapping from original colors to normalized colors
    # Keep 0 as 0, map other colors to 1, 2, 3, ...
    mapping = {}
    next_color = 1
    
    for color in unique_colors:
        if color == 0:
            mapping[int(color)] = 0  # Keep 0 as 0
        else:
            mapping[int(color)] = next_color
            next_color += 1
    
    # Apply the mapping to each color
    for old_color, new_color in mapping.items():
        mask = (grid == old_color)
        norm[mask] = new_color
        
    return norm, mapping


def denormalise(grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
    """
    Denormalize a grid using the provided mapping.
    
    Args:
        grid: The normalized grid
        mapping: The forward mapping from original colors to normalized colors
        
    Returns:
        The denormalized grid
    """
    # Create a copy of the grid to modify
    result = grid.copy()
    
    # Create inverse mapping (normalized -> original)
    inverse_mapping = {v: k for k, v in mapping.items()}
    
    # Apply the inverse mapping to each color
    for norm_color, orig_color in inverse_mapping.items():
        mask = (grid == norm_color)
        result[mask] = orig_color
        
    return result
