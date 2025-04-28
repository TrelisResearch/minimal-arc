"""
Debug script for the flood_fill function.

This script tests the flood_fill function with different inputs and prints the results.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from dsl.dsl_utils.primitives import flood_fill_fn
from dsl.dsl_utils.types import Grid


def main():
    """Test the flood_fill function with different inputs."""
    # Create a test grid
    grid = Grid(np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0]
    ]))
    
    print("Original grid:")
    print(grid.data)
    
    # Test filling from the top-left
    result1 = flood_fill_fn(grid, 0, 0, 2)
    print("\nAfter filling from (0, 0) with color 2:")
    print(result1.data)
    
    # Test filling from the bottom-right
    result2 = flood_fill_fn(grid, 2, 2, 3)
    print("\nAfter filling from (2, 2) with color 3:")
    print(result2.data)
    
    # The expected result for the second test
    expected2 = Grid(np.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 3]
    ]))
    print("\nExpected result for the second test:")
    print(expected2.data)
    
    # Check if the results match the expected values
    print("\nDo the results match the expected values?")
    print(f"First test: {np.array_equal(result1.data, np.array([[2, 2, 2], [2, 0, 0], [2, 0, 0]]))}")
    print(f"Second test: {np.array_equal(result2.data, expected2.data)}")
    
    # Print the differences for the second test
    print("\nDifferences for the second test:")
    print("result2.data:")
    print(result2.data)
    print("expected2.data:")
    print(expected2.data)
    print("Equality matrix:")
    print(result2.data == expected2.data)


if __name__ == "__main__":
    main()
