"""
Tests for ARC DSL primitives.

This module contains tests for the primitive operations in the ARC DSL.
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

from dsl.dsl_utils.primitives import (
    rot90_fn, rot180_fn, rot270_fn, flip_h_fn, flip_v_fn, transpose_fn,
    color_mask_fn, find_objects_fn, get_bbox_fn,
    tile_fn, tile_pattern_fn, crop_fn, replace_color_fn, count_color_fn,
    shift_up_pad, shift_down_pad, hole_mask
)
from dsl.dsl_utils.types import Grid, ObjList, Object


class TestGridTransformations(unittest.TestCase):
    """Test basic grid transformation primitives."""
    
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
    
    def test_rot90(self):
        """Test 90-degree rotation."""
        result = rot90_fn(self.grid_2x2)
        expected = Grid(np.array([
            [2, 4],
            [1, 3]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = rot90_fn(self.grid_3x3)
        expected = Grid(np.array([
            [3, 6, 9],
            [2, 5, 8],
            [1, 4, 7]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_rot180(self):
        """Test 180-degree rotation."""
        result = rot180_fn(self.grid_2x2)
        expected = Grid(np.array([
            [4, 3],
            [2, 1]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = rot180_fn(self.grid_3x3)
        expected = Grid(np.array([
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_rot270(self):
        """Test 270-degree rotation."""
        result = rot270_fn(self.grid_2x2)
        expected = Grid(np.array([
            [3, 1],
            [4, 2]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = rot270_fn(self.grid_3x3)
        expected = Grid(np.array([
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_flip_h(self):
        """Test horizontal flip."""
        result = flip_h_fn(self.grid_2x2)
        expected = Grid(np.array([
            [2, 1],
            [4, 3]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = flip_h_fn(self.grid_3x3)
        expected = Grid(np.array([
            [3, 2, 1],
            [6, 5, 4],
            [9, 8, 7]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_flip_v(self):
        """Test vertical flip."""
        result = flip_v_fn(self.grid_2x2)
        expected = Grid(np.array([
            [3, 4],
            [1, 2]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = flip_v_fn(self.grid_3x3)
        expected = Grid(np.array([
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_transpose(self):
        """Test transpose."""
        result = transpose_fn(self.grid_2x2)
        expected = Grid(np.array([
            [1, 3],
            [2, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = transpose_fn(self.grid_3x3)
        expected = Grid(np.array([
            [1, 4, 7],
            [2, 5, 8],
            [3, 6, 9]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))


class TestColorOperations(unittest.TestCase):
    """Test color-related primitives."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test grid with multiple colors
        self.grid = Grid(np.array([
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ]))
    
    def test_color_mask(self):
        """Test color masking."""
        result = color_mask_fn(self.grid, 1)
        expected = Grid(np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = color_mask_fn(self.grid, 2)
        expected = Grid(np.array([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_replace_color(self):
        """Test color replacement."""
        result = replace_color_fn(self.grid, 1, 5)
        expected = Grid(np.array([
            [0, 5, 2],
            [5, 2, 0],
            [2, 0, 5]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_count_color(self):
        """Test color counting."""
        result = count_color_fn(self.grid, 1)
        self.assertEqual(result, 3)
        
        result = count_color_fn(self.grid, 2)
        self.assertEqual(result, 3)
        
        result = count_color_fn(self.grid, 0)
        self.assertEqual(result, 3)
        
        result = count_color_fn(self.grid, 3)
        self.assertEqual(result, 0)


class TestObjectOperations(unittest.TestCase):
    """Test object-related primitives."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a grid with two objects
        self.grid = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ]))
    
    def test_find_objects(self):
        """Test object detection."""
        objects = find_objects_fn(self.grid)
        
        # Check that we found two objects
        self.assertEqual(len(objects), 2)
        
        # Check the first object (color 1)
        obj1 = objects[0]
        self.assertEqual(obj1.color, 1)
        self.assertEqual(obj1.position, (1, 1))
        self.assertEqual(obj1.grid.shape, (2, 2))
        self.assertTrue(np.array_equal(obj1.grid.data, np.array([[1, 1], [1, 1]])))
        
        # Check the second object (color 2)
        obj2 = objects[1]
        self.assertEqual(obj2.color, 2)
        self.assertEqual(obj2.position, (3, 3))
        self.assertEqual(obj2.grid.shape, (2, 2))
        self.assertTrue(np.array_equal(obj2.grid.data, np.array([[2, 2], [2, 2]])))
    
    def test_get_bbox(self):
        """Test bounding box generation."""
        objects = find_objects_fn(self.grid)
        bbox = get_bbox_fn(objects)
        
        # Expected bounding boxes: outlines of the objects
        expected = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 2]
        ]))
        
        # The bounding box should match the original objects
        # (since they're already rectangular)
        self.assertTrue(np.array_equal(bbox.data, expected.data))


class TestGridManipulations(unittest.TestCase):
    """Test grid manipulation primitives."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2x2 test grid
        self.grid_2x2 = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
    
    def test_tile(self):
        """Test tiling."""
        result = tile_fn(self.grid_2x2, 2, 3)
        expected = Grid(np.array([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_tile_pattern(self):
        """Test the special tile pattern for task 00576224."""
        # Create a 2x2 grid
        grid = Grid(np.array([
            [1, 2],
            [3, 4]
        ]))
        
        result = tile_pattern_fn(grid)
        expected = Grid(np.array([
            # Rows 0-1: Original pattern repeated 3 times horizontally
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            # Rows 2-3: Flipped pattern (both horizontally and vertically)
            [4, 3, 4, 3, 4, 3],
            [2, 1, 2, 1, 2, 1],
            # Rows 4-5: Original pattern again
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        # Test with a non-2x2 grid (should return the original grid)
        grid = Grid(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]))
        
        result = tile_pattern_fn(grid)
        self.assertTrue(np.array_equal(result.data, grid.data))
    
    def test_crop(self):
        """Test cropping."""
        # Create a 4x4 grid
        grid = Grid(np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16]
        ]))
        
        # Crop a 2x2 section from the center
        result = crop_fn(grid, 1, 1, 2, 2)
        expected = Grid(np.array([
            [6, 7],
            [10, 11]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        # Test cropping with out-of-bounds coordinates
        result = crop_fn(grid, -1, -1, 3, 3)
        expected = Grid(np.array([
            [1, 2, 3],
            [5, 6, 7],
            [9, 10, 11]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))


class TestShiftOperations(unittest.TestCase):
    """Test shift operations with zero-padding."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple 2x2 test grid
        self.grid_2x2 = Grid(np.array([
            [1, 1],
            [0, 1]
        ]))
        
        # Create a simple 3x3 test grid
        self.grid_3x3 = Grid(np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]))
    
    def test_shift_up_pad(self):
        """Test shift up with zero-padding."""
        result = shift_up_pad(self.grid_2x2)
        expected = Grid(np.array([
            [0, 1],
            [0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = shift_up_pad(self.grid_3x3)
        expected = Grid(np.array([
            [4, 5, 6],
            [7, 8, 9],
            [0, 0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_shift_down_pad(self):
        """Test shift down with zero-padding."""
        result = shift_down_pad(self.grid_2x2)
        expected = Grid(np.array([
            [1, 1],
            [0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
        
        result = shift_down_pad(self.grid_3x3)
        expected = Grid(np.array([
            [4, 5, 6],
            [1, 2, 3],
            [0, 0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_rot_chain_identity(self):
        """Test that ROT90 followed by ROT270 returns the original grid."""
        # Create a random grid
        data = np.random.randint(0, 10, (3, 3))
        grid = Grid(data)
        
        # Apply ROT90 followed by ROT270
        result = rot270_fn(rot90_fn(grid))
        
        # Check that the result is the same as the original
        self.assertTrue(np.array_equal(result.data, grid.data))


class TestHoleMask(unittest.TestCase):
    """Test hole mask operation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a grid with a hole in the middle
        self.grid_with_hole = Grid(np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]))
        
        # Create a grid with no holes
        self.grid_no_holes = Grid(np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]))
    
    def test_hole_mask_with_hole(self):
        """Test hole_mask with a grid containing a hole."""
        result = hole_mask(self.grid_with_hole)
        expected = Grid(np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]))
        self.assertTrue(np.array_equal(result.data, expected.data))
    
    def test_hole_mask_no_holes(self):
        """Test hole_mask with a grid containing no holes."""
        result = hole_mask(self.grid_no_holes)
        expected = Grid(np.zeros((3, 3), dtype=int))
        self.assertTrue(np.array_equal(result.data, expected.data))


if __name__ == '__main__':
    unittest.main()
