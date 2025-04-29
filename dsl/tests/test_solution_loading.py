"""
Test script to examine solution loading and dimensionality issues.
"""
import os
import sys
import numpy as np
from pathlib import Path

# Add the parent directory to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from dsl.io.loader import load_task, load_solution
from dsl.utils.color import normalise_palette, denormalise

def inspect_solution(task_id, data_path="../arc-data-cleaned"):
    """Inspect the solution for a task to check dimensionality."""
    print(f"Inspecting solution for task {task_id}...")
    
    # Load the solution
    try:
        solution = load_solution(task_id, data_path)
        print(f"Solution type: {type(solution)}")
        print(f"Solution shape: {np.array(solution).shape}")
        
        # Check if solution has 3 dimensions
        solution_array = np.array(solution)
        if len(solution_array.shape) == 3:
            print(f"WARNING: Solution has 3 dimensions: {solution_array.shape}")
            print(f"First dimension values: {[solution_array[i, 0, 0] for i in range(solution_array.shape[0])]}")
            
            # Extract the 2D grid (assuming the first dimension is irrelevant)
            solution_2d = solution_array[0]
            print(f"Extracted 2D solution shape: {solution_2d.shape}")
        else:
            print("Solution is already 2D.")
            solution_2d = solution_array
            
        # Test color normalization
        normalized, mapping = normalise_palette(solution_2d)
        print(f"Color mapping: {mapping}")
        
        # Test denormalization
        denormalized = denormalise(normalized, mapping)
        
        # Check if denormalized matches original
        if np.array_equal(denormalized, solution_2d):
            print("✅ Denormalized solution matches original")
        else:
            print("❌ Denormalized solution does NOT match original")
            print(f"Original unique colors: {np.unique(solution_2d)}")
            print(f"Denormalized unique colors: {np.unique(denormalized)}")
            
    except Exception as e:
        print(f"Error loading solution: {e}")
        
    # Load the task to check test output
    try:
        task = load_task(task_id, data_path)
        test_input = np.array(task['test'][0]['input'])
        print(f"Test input shape: {test_input.shape}")
        
        # If we have a solution, check its shape against test input
        if 'solution' in locals():
            if solution_2d.shape != test_input.shape:
                print(f"WARNING: Solution shape {solution_2d.shape} doesn't match test input shape {test_input.shape}")
    except Exception as e:
        print(f"Error loading task: {e}")

if __name__ == "__main__":
    # Use command line argument for task_id if provided
    task_id = sys.argv[1] if len(sys.argv) > 1 else "692cd3b6"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "../arc-data-cleaned"
    
    inspect_solution(task_id, data_path)
