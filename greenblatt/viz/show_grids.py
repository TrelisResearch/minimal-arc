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

def visualize_task(
    task_data: Dict[str, Any], 
    task_id: str, 
    candidate_output: Optional[List[List[int]]] = None,
    save_path: Optional[str] = None
):
    """Visualize a task with input, expected output, and candidate output."""
    train_examples = task_data[task_id]["train"]
    test_examples = task_data[task_id]["test"]
    
    # Determine the number of rows in the figure
    n_rows = len(train_examples) + len(test_examples)
    
    # Create figure
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows))
    
    # If there's only one row, wrap axes in a list
    if n_rows == 1:
        axes = [axes]
    
    # Plot training examples
    for i, example in enumerate(train_examples):
        plot_grid(axes[i][0], example["input"], f"Train {i+1} Input")
        plot_grid(axes[i][1], example["output"], f"Train {i+1} Expected Output")
        axes[i][2].axis('off')  # No candidate output for training examples
    
    # Plot test examples
    for i, example in enumerate(test_examples):
        row_idx = len(train_examples) + i
        plot_grid(axes[row_idx][0], example["input"], f"Test {i+1} Input")
        
        # If we have ground truth for the test example
        if "output" in example:
            plot_grid(axes[row_idx][1], example["output"], f"Test {i+1} Expected Output")
        else:
            axes[row_idx][1].axis('off')
        
        # If we have a candidate output
        if candidate_output is not None:
            plot_grid(axes[row_idx][2], candidate_output, "Candidate Output")
        else:
            axes[row_idx][2].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

def load_task_data(task_file: str) -> Dict[str, Any]:
    """Load task data from a JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Visualize ARC tasks")
    parser.add_argument("task_file", help="Path to task JSON file")
    parser.add_argument("--task-id", help="Specific task ID to visualize")
    parser.add_argument("--candidate", help="Path to JSON file with candidate outputs")
    parser.add_argument("--save", help="Path to save visualization")
    args = parser.parse_args()
    
    # Load task data
    task_data = load_task_data(args.task_file)
    
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
                task_id, 
                candidate_output,
                save_path
            )

if __name__ == "__main__":
    main()
