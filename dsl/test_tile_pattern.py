"""
Test script for the tile_pattern function on task 00576224.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Import the necessary modules
from dsl.dsl_utils.primitives import tile_pattern_fn
from dsl.dsl_utils.types import Grid
from dsl.io.loader import load_task, load_train_pairs, load_test_input

def main():
    """Test the tile_pattern function on task 00576224."""
    task_id = "00576224"
    
    # Load the task
    print(f"Loading task {task_id}...")
    task = load_task(task_id)
    
    # Extract training pairs
    train_pairs = load_train_pairs(task)
    test_input = load_test_input(task)
    
    # Create a figure for visualization
    fig, axs = plt.subplots(len(train_pairs) + 1, 3, figsize=(12, 4 * (len(train_pairs) + 1)))
    
    # Process each training example
    for i, (inp, expected) in enumerate(train_pairs):
        # Apply tile_pattern to the input
        result = tile_pattern_fn(inp)
        
        # Check if the result matches the expected output
        matches = np.array_equal(result.data, expected.data)
        status = "✓" if matches else "✗"
        
        # Display the input, expected output, and result
        axs[i, 0].imshow(inp.data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
        axs[i, 0].set_title(f"Train {i+1} Input")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(expected.data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
        axs[i, 1].set_title(f"Train {i+1} Expected")
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(result.data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
        axs[i, 2].set_title(f"Train {i+1} Result {status}")
        axs[i, 2].axis('off')
        
        # Add grid lines
        for ax in axs[i]:
            ax.set_xticks(np.arange(-0.5, max(inp.data.shape[1], expected.data.shape[1], result.data.shape[1]), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, max(inp.data.shape[0], expected.data.shape[0], result.data.shape[0]), 1), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
        
        # Print the results
        print(f"Training example {i+1}:")
        print(f"  Input: {inp.data.tolist()}")
        print(f"  Expected: {expected.data.tolist()}")
        print(f"  Result: {result.data.tolist()}")
        print(f"  Matches: {matches}")
    
    # Process the test input
    test_result = tile_pattern_fn(test_input)
    
    # Display the test input and result
    axs[-1, 0].imshow(test_input.data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
    axs[-1, 0].set_title("Test Input")
    axs[-1, 0].axis('off')
    
    # Leave the middle column empty for the test (no expected output)
    axs[-1, 1].axis('off')
    
    axs[-1, 2].imshow(test_result.data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
    axs[-1, 2].set_title("Test Result")
    axs[-1, 2].axis('off')
    
    # Add grid lines for test
    for ax in [axs[-1, 0], axs[-1, 2]]:
        ax.set_xticks(np.arange(-0.5, max(test_input.data.shape[1], test_result.data.shape[1]), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, max(test_input.data.shape[0], test_result.data.shape[0]), 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Print the test results
    print(f"Test input: {test_input.data.tolist()}")
    print(f"Test result: {test_result.data.tolist()}")
    
    # Show the figure
    plt.tight_layout()
    plt.show()
    
    print("Test completed!")

if __name__ == "__main__":
    main()
