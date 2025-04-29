"""
Tests for ARC DSL Program execution.

This module contains tests for the Program class in the ARC DSL.
"""
import sys
import os
from pathlib import Path
import unittest
import numpy as np

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from dsl.dsl_utils.program import Program
from dsl.dsl_utils.primitives import (
    ROT90, ROT180, FLIP_H, FLIP_V, TRANSPOSE,
    MASK_C_1, MASK_C_2, MASK_C_3, MASK_C_4, MASK_C_5,
    FILL_CENTER_1, FILL_TL_1, OBJECTS, BBOX, 
    TILE_3x3, TILE_PATTERN, CROP_CENTER_HALF,
    REPLACE_1_TO_2, REPLACE_2_TO_3, COUNT_C_1
)
from dsl.dsl_utils.types import Grid, ObjList, Object, Grid_T, ObjList_T, Int_T


class TestProgramExecution(unittest.TestCase):
    """Test Program execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2x2 test grid
        self.grid_2x2 = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        
        # Create a simple 3x3 test grid
        self.grid_3x3 = Grid(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
    
    def test_single_operation(self):
        """Test a program with a single operation."""
        # Test ROT90
        program = Program([ROT90])
        result = program.run(self.grid_2x2)
        expected = Grid(np.array([
            [3, 1],
            [4, 2]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        # Test FLIP_H
        program = Program([FLIP_H])
        result = program.run(self.grid_2x2)
        expected = Grid(np.array([
            [2, 1],
            [4, 3]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_multiple_operations(self):
        """Test a program with multiple operations."""
        # Test ROT90 followed by FLIP_H
        program = Program([ROT90, FLIP_H])
        result = program.run(self.grid_2x2)
        expected = Grid(np.array([
            [1, 3],
            [2, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        # Test FLIP_H followed by FLIP_V
        program = Program([FLIP_H, FLIP_V])
        result = program.run(self.grid_2x2)
        expected = Grid(np.array([
            [4, 3],
            [2, 1]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_tile_operation(self):
        """Test the tile operation."""
        program = Program([TILE_3x3])
        result = program.run(self.grid_2x2)
        
        # The tiling should be 3x3
        expected = Grid(np.array([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_tile_pattern_operation(self):
        """Test the tile_pattern operation for task 00576224."""
        program = Program([TILE_PATTERN])
        result = program.run(self.grid_2x2)
        
        expected = Grid(np.array([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [2, 1, 2, 1, 2, 1],
            [4, 3, 4, 3, 4, 3],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_type_compatibility(self):
        """Test type compatibility checking."""
        # Valid program: Grid -> Grid -> Grid
        program = Program([ROT90, FLIP_H])
        self.assertTrue(program.is_compatible(Grid_T, Grid_T))
        
        # Invalid program: Grid -> ObjList -> Grid
        program = Program([OBJECTS, ROT90])
        self.assertFalse(program.is_compatible(Grid_T, Grid_T))
        
        # Valid program: Grid -> ObjList -> Grid
        program = Program([OBJECTS, BBOX])
        self.assertTrue(program.is_compatible(Grid_T, Grid_T))
    
    def test_error_handling(self):
        """Test error handling during program execution."""
        # Create a program that will fail due to type mismatch
        program = Program([COUNT_C_1, ROT90])
        result = program.run(self.grid_2x2)
        
        # The program should return None due to the error
        self.assertIsNone(result)


class TestComplexPrograms(unittest.TestCase):
    """Test more complex program execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a grid with objects
        self.grid = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ]))
    
    def test_object_detection_and_bbox(self):
        """Test object detection followed by bounding box generation."""
        program = Program([OBJECTS, BBOX])
        result = program.run(self.grid)
        
        # The result should be a grid with the bounding boxes
        expected = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_color_operations(self):
        """Test a sequence of color operations."""
        # Create a program that applies replace_color operations
        # REPLACE_1_TO_2 replaces color 1 with 2
        # REPLACE_2_TO_3 replaces color 2 with 3
        program = Program([REPLACE_1_TO_2, REPLACE_2_TO_3])
        result = program.run(self.grid)
        
        # Expected result after applying both operations:
        # Color 1 becomes 2, then 2 becomes 3
        # Color 2 becomes 3
        expected = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0],  # 1 -> 2 -> 3
            [0, 3, 3, 0, 0],  # 1 -> 2 -> 3
            [0, 0, 0, 3, 3],  # 2 -> 3
            [0, 0, 0, 3, 3]   # 2 -> 3
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))


if __name__ == '__main__':
    unittest.main()
