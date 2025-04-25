"""show_grids.py – quick matplotlib visualizer

Usage: python -m greenblatt.viz.show_grids path/to/task.json

Draws: input(s), expected output(s), candidate LLM output
"""
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ARC colors (10 colors for indices 0-9)
ARC_COLORS = [
    "#000000",  # 0: Black
    "#0074D9",  # 1: Blue
    "#FF4136",  # 2: Red
    "#2ECC40",  # 3: Green
    "#FFDC00",  # 4: Yellow
    "#AAAAAA",  # 5: Gray
    "#F012BE",  # 6: Magenta
    "#FF851B",  # 7: Orange
    "#7FDBFF",  # 8: Light Blue
    "#870C25",  # 9: Brown
]

def create_arc_colormap():
    """Create a colormap for ARC grids."""
    return ListedColormap(ARC_COLORS)

def pad_grid(grid: List[List[int]], max_rows: int, max_cols: int) -> np.ndarray:
    """Pad a grid to the specified dimensions."""
    grid_array = np.array(grid, dtype=np.int8)
    rows, cols = grid_array.shape
    
    if rows < max_rows or cols < max_cols:
        padded = np.zeros((max_rows, max_cols), dtype=np.int8)
        padded[:rows, :cols] = grid_array
        return padded
    
    return grid_array

def plot_grid(ax, grid: List[List[int]], title: str = None):
    """Plot a single grid on the given axis."""
    grid_array = np.array(grid, dtype=np.int8)
    
    # Plot the grid
    ax.imshow(grid_array, cmap=create_arc_colormap(), vmin=0, vmax=9)
    
    # Add grid lines
    ax.grid(color='black', linestyle='-', linewidth=0.5)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(-0.5, grid_array.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_array.shape[0], 1), minor=True)
    ax.set_xticks(np.arange(0, grid_array.shape[1], 1))
    ax.set_yticks(np.arange(0, grid_array.shape[0], 1))
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    # Add title
    if title:
        ax.set_title(title)
    
    # Turn off axis
    ax.axis('off')

def grid_equals(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Check if two grids are equal."""
    # Convert to numpy arrays for comparison
    array1 = np.array(grid1)
    array2 = np.array(grid2)
    
    # Check if shapes match
    if array1.shape != array2.shape:
        return False
    
    # Compare all elements
    return np.array_equal(array1, array2)

def visualize_task(
    task_data: Dict[str, Any], 
    solutions_data: Dict[str, Any],
    task_id: str, 
    candidate_output: Optional[List[List[int]]] = None,
    valid_programs: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize a task with input, expected output, and candidate output.
    
    Args:
        task_data: Dictionary of task data
        solutions_data: Dictionary of solutions data
        task_id: Task ID
        candidate_output: Optional candidate output for test example
        valid_programs: Optional list of valid programs to visualize training predictions
        save_path: Optional path to save the visualization
    """
    train_examples = task_data[task_id]["train"]
    test_examples = task_data[task_id]["test"]
    
    # Get ground truth for test examples from solutions data if available
    test_ground_truth = {}
    if task_id in solutions_data:
        # The solutions data is just an array of outputs
        if isinstance(solutions_data[task_id], list) and len(solutions_data[task_id]) > 0:
            for i, solution in enumerate(solutions_data[task_id]):
                if i < len(test_examples):
                    test_ground_truth[i] = solution
    
    # If we have valid programs, get their predictions for training examples
    training_predictions = []
    if valid_programs and len(valid_programs) > 0:
        # Use the first valid program to predict training outputs
        program = valid_programs[0]
        train_inputs = [example["input"] for example in train_examples]
        
        # Get training predictions without using asyncio directly
        # We'll use a helper function that handles the async execution
        training_predictions = get_training_predictions(program, train_inputs)
    
    # Determine the number of rows in the figure
    n_rows = len(train_examples) + len(test_examples)
    
    # Determine the number of columns (3 for standard, 4 if showing training predictions)
    n_cols = 4 if training_predictions else 3
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    
    # If there's only one row, wrap axes in a list
    if n_rows == 1:
        axes = [axes]
    
    # Add a title to the figure - position it to avoid overlap
    fig.suptitle(f"Task: {task_id}", fontsize=16, y=0.98)
    
    # Plot training examples
    for i, example in enumerate(train_examples):
        plot_grid(axes[i][0], example["input"], f"Train {i+1} Input")
        plot_grid(axes[i][1], example["output"], f"Train {i+1} Expected Output")
        
        # If we have training predictions, show them
        if training_predictions and i < len(training_predictions) and training_predictions[i] is not None:
            # Check if prediction matches expected output
            is_correct = grid_equals(training_predictions[i], example["output"])
            
            # Add a title that indicates correctness
            title = f"Train {i+1} Prediction"
            title += f" ({'✓' if is_correct else '✗'})"
            
            # Plot with a green or red border based on correctness
            plot_grid(axes[i][2], training_predictions[i], title)
            
            # Add a colored border
            border_color = 'green' if is_correct else 'red'
            for spine in axes[i][2].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            axes[i][2].set_frame_on(True)
        else:
            axes[i][2].axis('off')
        
        # Turn off the last column if we're using 4 columns for training examples
        if n_cols == 4:
            axes[i][3].axis('off')
    
    # Plot test examples
    for i, example in enumerate(test_examples):
        row_idx = len(train_examples) + i
        plot_grid(axes[row_idx][0], example["input"], f"Test {i+1} Input")
        
        # Check if we have ground truth from solutions data
        ground_truth_available = i in test_ground_truth
        
        # Always show the expected output column for test examples
        if ground_truth_available:
            plot_grid(axes[row_idx][1], test_ground_truth[i], f"Test {i+1} Ground Truth")
        elif "output" in example:
            plot_grid(axes[row_idx][1], example["output"], f"Test {i+1} Expected Output")
        else:
            axes[row_idx][1].text(0.5, 0.5, "Ground Truth Not Available", 
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=axes[row_idx][1].transAxes)
            axes[row_idx][1].axis('on')
        
        # If we have a candidate output
        if candidate_output is not None:
            # Determine if the candidate output is correct (if ground truth is available)
            is_correct = False
            if ground_truth_available:
                is_correct = grid_equals(candidate_output, test_ground_truth[i])
            elif "output" in example:
                is_correct = grid_equals(candidate_output, example["output"])
            
            # Add a title that indicates correctness
            title = "Candidate Output"
            if ground_truth_available or "output" in example:
                title += f" ({'✓' if is_correct else '✗'})"
            
            # Plot with a green or red border based on correctness
            plot_grid(axes[row_idx][2], candidate_output, title)
            
            # Add a colored border if ground truth is available
            if ground_truth_available or "output" in example:
                border_color = 'green' if is_correct else 'red'
                for spine in axes[row_idx][2].spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(3)
                axes[row_idx][2].set_frame_on(True)
        else:
            axes[row_idx][2].axis('off')
        
        # Turn off the last column if we're using 4 columns
        if n_cols == 4:
            axes[row_idx][3].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the suptitle
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def get_training_predictions(program: str, inputs: List[List[List[int]]]) -> List[Optional[List[List[int]]]]:
    """
    Helper function to get training predictions without directly using asyncio.
    This avoids the "event loop is already running" error.
    
    Args:
        program: The Python code string containing a solve function
        inputs: A list of input grids to test
        
    Returns:
        A list of output grids or None for each input if execution failed
    """
    from sandbox.runner import run_in_sandbox
    import asyncio
    import nest_asyncio
    
    # Apply nest_asyncio to allow nested event loops
    nest_asyncio.apply()
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Run the sandbox in this loop
        return loop.run_until_complete(run_in_sandbox(program, inputs))
    finally:
        # Clean up
        loop.close()

def load_task_data(task_file: str) -> Dict[str, Any]:
    """Load task data from a JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)

def load_solutions_data(solutions_file: str) -> Dict[str, Any]:
    """Load solutions data from a JSON file."""
    with open(solutions_file, 'r') as f:
        return json.load(f)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize ARC tasks")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("solutions_file", help="Path to solutions JSON file")
    parser.add_argument("--task-id", help="Specific task ID to visualize")
    parser.add_argument("--candidate", help="Path to JSON file with candidate outputs")
    parser.add_argument("--save", help="Path to save visualization")
    args = parser.parse_args()
    
    # Load task data
    task_data = load_task_data(args.task_file)
    
    # Load solutions data
    solutions_data = load_solutions_data(args.solutions_file)
    
    # Load candidate outputs if provided
    candidate_outputs = None
    if args.candidate:
        with open(args.candidate, 'r') as f:
            candidate_outputs = json.load(f)
    
    # If task ID is provided, visualize only that task
    if args.task_id:
        if args.task_id not in task_data:
            print(f"Task ID {args.task_id} not found in {args.task_file}")
            return
        
        candidate_output = None
        if candidate_outputs and args.task_id in candidate_outputs:
            candidate_output = candidate_outputs[args.task_id]
        
        visualize_task(
            task_data, 
            solutions_data,
            args.task_id, 
            candidate_output,
            args.save
        )
    else:
        # Visualize all tasks
        for task_id in task_data:
            candidate_output = None
            if candidate_outputs and task_id in candidate_outputs:
                candidate_output = candidate_outputs[task_id]
            
            save_path = None
            if args.save:
                save_dir = Path(args.save)
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / f"{task_id}.png"
            
            visualize_task(
                task_data, 
                solutions_data,
                task_id, 
                candidate_output,
                save_path
            )

if __name__ == "__main__":
    main()
