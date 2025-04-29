"""
Tests for color utilities.

This module contains tests for the color utility functions.
"""
import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from dsl.utils.color import normalise_palette, denormalise


class TestColorUtils(unittest.TestCase):
    """Test color utility functions."""
    
    def test_normalise_palette(self):
        """Test palette normalization."""
        # Test case from the example
        grid = np.array([[0, 4], [9, 4]])
        expected = np.array([[0, 1], [2, 1]])
        
        norm, mapping = normalise_palette(grid)
        self.assertTrue(np.array_equal(norm, expected))
        self.assertEqual(mapping, {4: 1, 9: 2})
        
        # Test with more complex grid
        grid = np.array([
            [0, 5, 8, 0],
            [5, 0, 8, 3],
            [0, 3, 5, 0]
        ])
        # The colors are sorted in ascending order: 3, 5, 8
        expected = np.array([
            [0, 2, 3, 0],
            [2, 0, 3, 1],
            [0, 1, 2, 0]
        ])
        
        norm, mapping = normalise_palette(grid)
        self.assertTrue(np.array_equal(norm, expected))
        self.assertEqual(mapping, {3: 1, 5: 2, 8: 3})
        
        # Test with already normalized grid
        grid = np.array([
            [0, 1, 2],
            [2, 1, 0]
        ])
        
        norm, mapping = normalise_palette(grid)
        self.assertTrue(np.array_equal(norm, grid))
        self.assertEqual(mapping, {1: 1, 2: 2})
    
    def test_denormalise(self):
        """Test palette denormalization."""
        # Test case from the example
        grid = np.array([[0, 1], [2, 1]])
        mapping = {4: 1, 9: 2}
        expected = np.array([[0, 4], [9, 4]])
        
        denorm = denormalise(grid, mapping)
        self.assertTrue(np.array_equal(denorm, expected))
        
        # Test with more complex grid
        grid = np.array([
            [0, 1, 2, 0],
            [1, 0, 2, 3],
            [0, 3, 1, 0]
        ])
        mapping = {3: 3, 5: 1, 8: 2}
        expected = np.array([
            [0, 5, 8, 0],
            [5, 0, 8, 3],
            [0, 3, 5, 0]
        ])
        
        denorm = denormalise(grid, mapping)
        self.assertTrue(np.array_equal(denorm, expected))


if __name__ == '__main__':
    unittest.main()
