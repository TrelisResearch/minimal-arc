"""show_grids.py – quick matplotlib visualizer

Usage: python -m llmgs.viz.show_grids path/to/task.json

Draws: input(s), expected output(s), candidate LLM output
"""
import json
import sys
import argparse
import time
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

# Create the colormap once
ARC_COLORMAP = ListedColormap(ARC_COLORS)

def pad_grid(grid: List[List[int]], max_rows: int, max_cols: int) -> np.ndarray:
    """Pad a grid to the specified dimensions."""
    grid_array = np.array(grid, dtype=np.int8)
    rows, cols = grid_array.shape
    
    if rows < max_rows or cols < max_cols:
        padded = np.zeros((max_rows, max_cols), dtype=np.int8)
        padded[:rows, :cols] = grid_array
        return padded
    
    return grid_array

def plot_grid(ax, grid: Union[List[List[int]], np.ndarray], title: str = None, border_color: Optional[str] = None):
    """Plot a single grid on the given axis."""
    # Convert to numpy array if not already
    if not isinstance(grid, np.ndarray):
        grid_array = np.array(grid, dtype=np.int8)
    else:
        grid_array = grid
    
    # Plot the grid using the global colormap
    ax.imshow(grid_array, cmap=ARC_COLORMAP, vmin=0, vmax=9)
    
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
    
    # Add colored border if specified
    if border_color:
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_frame_on(True)
    else:
        # Turn off axis
        ax.axis('off')

def grid_equals(grid1: Union[List[List[int]], np.ndarray], grid2: Union[List[List[int]], np.ndarray]) -> bool:
    """Check if two grids are equal."""
    # Convert to numpy arrays for comparison if not already
    if not isinstance(grid1, np.ndarray):
        array1 = np.array(grid1)
    else:
        array1 = grid1
        
    if not isinstance(grid2, np.ndarray):
        array2 = np.array(grid2)
    else:
        array2 = grid2
    
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
    save_path: Optional[str] = None,
    training_predictions: Optional[Dict[str, List[List[List[int]]]]] = None
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
        training_predictions: Optional dictionary mapping programs to their training predictions
    """
    viz_start_time = time.time()
    
    # Convert all grids to numpy arrays upfront to avoid repeated conversions
    train_examples = task_data[task_id]["train"]
    test_examples = task_data[task_id]["test"]
    
    # Pre-convert all grids to numpy arrays
    for example in train_examples:
        example["input_array"] = np.array(example["input"], dtype=np.int8)
        example["output_array"] = np.array(example["output"], dtype=np.int8)
    
    for example in test_examples:
        example["input_array"] = np.array(example["input"], dtype=np.int8)
        if "output" in example:
            example["output_array"] = np.array(example["output"], dtype=np.int8)
    
    # Convert candidate output if provided
    candidate_output_array = None
    if candidate_output is not None:
        candidate_output_array = np.array(candidate_output, dtype=np.int8)
    
    # Get ground truth for test examples from solutions data if available
    test_ground_truth = {}
    test_ground_truth_arrays = {}
    if task_id in solutions_data:
        # The solutions data is just an array of outputs
        if isinstance(solutions_data[task_id], list) and len(solutions_data[task_id]) > 0:
            for i, solution in enumerate(solutions_data[task_id]):
                if i < len(test_examples):
                    test_ground_truth[i] = solution
                    test_ground_truth_arrays[i] = np.array(solution, dtype=np.int8)
    
    # Check if we have valid programs and training predictions
    has_training_predictions = False
    program_training_outputs = []
    program_training_arrays = []
    
    if valid_programs and len(valid_programs) > 0 and training_predictions:
        # Use the first valid program's training predictions
        first_valid_program = valid_programs[0]
        if first_valid_program in training_predictions:
            program_training_outputs = training_predictions[first_valid_program]
            # Pre-convert training predictions to numpy arrays
            for output in program_training_outputs:
                if output is not None:
                    program_training_arrays.append(np.array(output, dtype=np.int8))
                else:
                    program_training_arrays.append(None)
            has_training_predictions = len(program_training_outputs) > 0
    
    # Determine the number of rows in the figure
    n_rows = len(train_examples) + len(test_examples)
    
    # Determine the number of columns (3 for standard, 4 if showing training predictions)
    n_cols = 4 if has_training_predictions else 3
    
    # Use a smaller figure size to reduce memory usage
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), dpi=80)
    
    # If there's only one row, wrap axes in a list
    if n_rows == 1:
        axes = [axes]
    
    # Add a title to the figure - position it to avoid overlap
    fig.suptitle(f"Task: {task_id}", fontsize=14, y=0.98)
    
    # Plot training examples
    for i, example in enumerate(train_examples):
        plot_grid(axes[i][0], example["input_array"], f"Train {i+1} Input")
        plot_grid(axes[i][1], example["output_array"], f"Train {i+1} Expected Output")
        
        # If we have training predictions, show them
        if has_training_predictions and i < len(program_training_arrays) and program_training_arrays[i] is not None:
            # Check if prediction matches expected output
            is_correct = np.array_equal(program_training_arrays[i], example["output_array"])
            
            # Add a title that indicates correctness
            title = f"Train {i+1} Prediction ({'✓' if is_correct else '✗'})"
            
            # Plot with a green or red border based on correctness
            border_color = 'green' if is_correct else 'red'
            plot_grid(axes[i][2], program_training_arrays[i], title, border_color)
        else:
            axes[i][2].axis('off')
        
        # Turn off the last column if we're using 4 columns for training examples
        if n_cols == 4:
            axes[i][3].axis('off')
    
    # Plot test examples
    for i, example in enumerate(test_examples):
        row_idx = len(train_examples) + i
        plot_grid(axes[row_idx][0], example["input_array"], f"Test {i+1} Input")
        
        # Check if we have ground truth from solutions data
        ground_truth_available = i in test_ground_truth
        
        # Always show the expected output column for test examples
        if ground_truth_available:
            plot_grid(axes[row_idx][1], test_ground_truth_arrays[i], f"Test {i+1} Ground Truth")
        elif "output_array" in example:
            plot_grid(axes[row_idx][1], example["output_array"], f"Test {i+1} Expected Output")
        else:
            axes[row_idx][1].text(0.5, 0.5, "Ground Truth Not Available", 
                                 horizontalalignment='center', verticalalignment='center',
                                 transform=axes[row_idx][1].transAxes)
            axes[row_idx][1].axis('on')
        
        # If we have a candidate output
        if candidate_output_array is not None:
            # Determine if the candidate output is correct (if ground truth is available)
            is_correct = False
            if ground_truth_available:
                is_correct = np.array_equal(candidate_output_array, test_ground_truth_arrays[i])
            elif "output_array" in example:
                is_correct = np.array_equal(candidate_output_array, example["output_array"])
            
            # Add a title that indicates correctness if we know
            title = f"Test {i+1} Candidate Output"
            if ground_truth_available or "output_array" in example:
                title += f" ({'✓' if is_correct else '✗'})"
            
            # Plot with a green or red border based on correctness
            border_color = None
            if ground_truth_available or "output_array" in example:
                border_color = 'green' if is_correct else 'red'
            
            plot_grid(axes[row_idx][2], candidate_output_array, title, border_color)
        else:
            axes[row_idx][2].axis('off')
        
        # Turn off the last column if we're using 4 columns
        if n_cols == 4:
            axes[row_idx][3].axis('off')
    
    # Use a simpler layout adjustment instead of tight_layout
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Save or show the figure
    if save_path:
        plt.savefig(save_path, dpi=80, bbox_inches=None)
    else:
        plt.show()
    
    # Close the figure to free memory
    plt.close(fig)
    
    viz_time = time.time() - viz_start_time
    print(f"Visualization completed in {viz_time:.2f}s")

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
    parser.add_argument("task_file", help="Path to the task JSON file")
    parser.add_argument("--solutions-file", help="Path to the solutions JSON file")
    parser.add_argument("--task-id", help="Specific task ID to visualize")
    parser.add_argument("--output-dir", help="Directory to save visualizations")
    args = parser.parse_args()
    
    # Load task data
    task_data = load_task_data(args.task_file)
    
    # Load solutions data if provided
    solutions_data = {}
    if args.solutions_file:
        solutions_data = load_solutions_data(args.solutions_file)
    
    # Create output directory if needed
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # If task ID is provided, visualize only that task
    if args.task_id:
        if args.task_id not in task_data:
            print(f"Task ID {args.task_id} not found in task data")
            return
        
        save_path = None
        if args.output_dir:
            save_path = str(Path(args.output_dir) / f"{args.task_id}.png")
        
        visualize_task(
            task_data=task_data,
            solutions_data=solutions_data,
            task_id=args.task_id,
            save_path=save_path
        )
    else:
        # Visualize all tasks
        for task_id in task_data:
            print(f"Visualizing task {task_id}")
            
            save_path = None
            if args.output_dir:
                save_path = str(Path(args.output_dir) / f"{task_id}.png")
            
            visualize_task(
                task_data=task_data,
                solutions_data=solutions_data,
                task_id=task_id,
                save_path=save_path
            )

if __name__ == "__main__":
    main()
