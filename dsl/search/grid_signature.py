"""
Grid signature utilities for early pruning in the search process.

This module provides functions to compute signatures of grids that can be
used to quickly determine if a partial program is incompatible with target outputs.
"""
from typing import Tuple, List, Set, Optional
import numpy as np
from collections import Counter

from ..dsl_utils.types import Grid


def compute_shape_signature(grid: Grid) -> Tuple[int, int]:
    """
    Compute the shape signature of a grid.
    
    Args:
        grid: The grid to compute the signature for
        
    Returns:
        The shape of the grid as (height, width)
    """
    return grid.shape


def compute_color_multiset(grid: Grid) -> Tuple[int, ...]:
    """
    Compute the color multiset signature of a grid.
    
    Args:
        grid: The grid to compute the signature for
        
    Returns:
        A sorted tuple of unique colors present in the grid
    """
    return tuple(sorted(np.unique(grid.data)))


def count_connected_components(grid: Grid) -> int:
    """
    Count the number of connected components in the grid.
    
    Args:
        grid: The grid to count connected components in
        
    Returns:
        The number of connected components
    """
    # Use a simple flood fill algorithm to count connected components
    data = grid.data.copy()
    height, width = data.shape
    visited = np.zeros_like(data, dtype=bool)
    count = 0
    
    # Define the 4-connected neighborhood
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in range(height):
        for j in range(width):
            if not visited[i, j]:
                color = data[i, j]
                count += 1
                
                # Perform flood fill from this cell
                stack = [(i, j)]
                visited[i, j] = True
                
                while stack:
                    r, c = stack.pop()
                    
                    for dr, dc in directions:
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width and 
                            not visited[nr, nc] and data[nr, nc] == color):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
    
    return count


def compute_dimension_parity(grid: Grid) -> Tuple[int, int]:
    """
    Compute the dimension parity signature of a grid.
    
    Args:
        grid: The grid to compute the signature for
        
    Returns:
        A tuple of (height % 2, width % 2)
    """
    height, width = grid.shape
    return (height % 2, width % 2)


def compute_grid_signature(grid: Grid) -> Tuple:
    """
    Compute a comprehensive signature for a grid.
    
    Args:
        grid: The grid to compute the signature for
        
    Returns:
        A tuple containing (shape, color_multiset, connected_component_count, dimension_parity)
    """
    shape = compute_shape_signature(grid)
    color_multiset = compute_color_multiset(grid)
    cc_count = count_connected_components(grid)
    dimension_parity = compute_dimension_parity(grid)
    
    return (shape, color_multiset, cc_count, dimension_parity)


def is_signature_compatible(current_sig: Tuple, target_sig: Tuple) -> bool:
    """
    Check if a current signature is compatible with a target signature.
    
    Args:
        current_sig: The current grid signature
        target_sig: The target grid signature
        
    Returns:
        True if the signatures are compatible, False otherwise
    """
    current_shape, current_colors, current_cc, current_parity = current_sig
    target_shape, target_colors, target_cc, target_parity = target_sig
    
    # Shape must match exactly
    if current_shape != target_shape:
        return False
    
    # Check color multiset compatibility
    # All colors in the current grid must be in the target grid
    if not all(color in target_colors for color in current_colors):
        return False
    
    # Connected component count can be used for pruning in some cases
    # If the current count is already greater than the target, it can't decrease
    # (most operations can only increase or maintain the count)
    if current_cc > target_cc:
        return False
    
    # Dimension parity check
    # Many operations preserve parity, so if it's already different, it's likely wrong
    if current_parity != target_parity:
        return False
    
    return True
