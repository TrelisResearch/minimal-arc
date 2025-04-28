"""
ARC Grid Visualizer.

This module provides utilities for visualizing ARC grids.
"""
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def show_grid(grid: Union[np.ndarray, List[List[int]]], ax: Optional[Axes] = None, 
             title: str = "", show_grid_lines: bool = True) -> Axes:
    """
    Display a grid using matplotlib.
    
    Args:
        grid: The grid to display (numpy array or list of lists)
        ax: The matplotlib axis to use (optional)
        title: The title for the plot
        show_grid_lines: Whether to show grid lines
        
    Returns:
        The matplotlib axis
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    
    # Convert to numpy array if needed
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    
    # Use a discrete colormap with 10 colors (0-9)
    cmap = plt.cm.get_cmap('tab10', 10)
    
    # Display the grid
    ax.imshow(grid, interpolation='nearest', vmin=0, vmax=9, cmap=cmap)
    
    # Add grid lines
    if show_grid_lines:
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    ax.set_title(title)
    
    return ax


def compare_grids(input_grid: Union[np.ndarray, List[List[int]]],
                 prediction: Union[np.ndarray, List[List[int]]],
                 target: Optional[Union[np.ndarray, List[List[int]]]] = None,
                 label: str = "") -> Figure:
    """
    Compare input, prediction, and optionally target grids.
    
    Args:
        input_grid: The input grid
        prediction: The predicted output grid
        target: The target output grid (optional)
        label: The label for the figure
        
    Returns:
        The matplotlib figure
    """
    n_cols = 3 if target is not None else 2
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    # Display the input grid
    show_grid(input_grid, axs[0], "Input")
    
    # Display the prediction
    show_grid(prediction, axs[1], "Prediction")
    
    # Display the target if provided
    if target is not None:
        show_grid(target, axs[2], "Target")
        
        # Add a visual indicator if prediction matches target
        if isinstance(prediction, np.ndarray) and isinstance(target, np.ndarray):
            matches = np.array_equal(prediction, target)
        else:
            matches = prediction == target
            
        if matches:
            axs[1].set_title("Prediction (Correct)")
            # Add a green border
            for spine in axs[1].spines.values():
                spine.set_edgecolor('green')
                spine.set_linewidth(3)
        else:
            axs[1].set_title("Prediction (Incorrect)")
            # Add a red border
            for spine in axs[1].spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    # Add a super title
    if label:
        fig.suptitle(label, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    else:
        fig.tight_layout()
    
    return fig


def visualize_task(task: dict, prediction: Optional[Union[np.ndarray, List[List[int]]]] = None,
                  solution: Optional[Union[np.ndarray, List[List[int]]]] = None) -> Figure:
    """
    Visualize a complete ARC task with training examples and test.
    
    Args:
        task: The task dictionary
        prediction: The predicted output for the test input (optional)
        solution: The ground truth solution (optional)
        
    Returns:
        The matplotlib figure
    """
    # Count the number of training examples
    n_train = len(task['train'])
    
    # Create a grid of subplots
    n_rows = n_train + 1  # Training examples + test
    n_cols = 3 if prediction is not None else 2  # Input, Output, (Prediction)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # If there's only one row, wrap the axes in a list
    if n_rows == 1:
        axs = [axs]
    
    # Visualize training examples
    for i, example in enumerate(task['train']):
        show_grid(example['input'], axs[i][0], f"Train {i+1} Input")
        show_grid(example['output'], axs[i][1], f"Train {i+1} Output")
        
        # If we have a prediction column, keep it empty for training examples
        if n_cols > 2:
            axs[i][2].axis('off')
    
    # Visualize test example
    test_input = task['test'][0]['input']
    show_grid(test_input, axs[-1][0], "Test Input")
    
    # If we have the solution, show it
    if solution is not None:
        show_grid(solution, axs[-1][1], "Test Solution")
    else:
        axs[-1][1].axis('off')
    
    # If we have a prediction, show it
    if prediction is not None:
        show_grid(prediction, axs[-1][2], "Test Prediction")
        
        # Add a visual indicator if prediction matches solution
        if solution is not None:
            if isinstance(prediction, np.ndarray) and isinstance(solution, np.ndarray):
                matches = np.array_equal(prediction, solution)
            else:
                matches = prediction == solution
                
            if matches:
                axs[-1][2].set_title("Test Prediction (Correct)")
                # Add a green border
                for spine in axs[-1][2].spines.values():
                    spine.set_edgecolor('green')
                    spine.set_linewidth(3)
            else:
                axs[-1][2].set_title("Test Prediction (Incorrect)")
                # Add a red border
                for spine in axs[-1][2].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
    
    fig.tight_layout()
    return fig


def save_visualization(fig: Figure, filename: str) -> None:
    """
    Save a visualization to a file.
    
    Args:
        fig: The matplotlib figure
        filename: The output filename
    """
    fig.savefig(filename, bbox_inches='tight', dpi=100)
