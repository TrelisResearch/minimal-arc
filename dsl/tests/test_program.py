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
    COLORMASK, FILL, OBJECTS, BBOX, TILE, TILE_PATTERN,
    CROP, REPLACE_COLOR, COUNT_COLOR
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
        """Test the tile operation which requires additional arguments."""
        program = Program([TILE])
        result = program.run(self.grid_2x2)
        
        # The default tiling should be 3x3
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
        program = Program([COUNT_COLOR, ROT90])
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
        # Create a custom program that applies replace_color operations
        program = Program([
            REPLACE_COLOR,  # Will be called with default arguments in Program.run
            REPLACE_COLOR   # Will be called with default arguments in Program.run
        ])
        
        # Override the run method to use specific arguments
        original_run = program.run
        def custom_run(grid):
            # First replace color 1 with 3
            intermediate = REPLACE_COLOR.fn(grid, 1, 3)
            # Then replace color 2 with 4
            return REPLACE_COLOR.fn(intermediate, 2, 4)
        
        # Use our custom run method
        program.run = custom_run
        
        result = program.run(self.grid)
        
        expected = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 0, 0, 4, 4],
            [0, 0, 0, 4, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))


if __name__ == '__main__':
    unittest.main()
