"""
ARC Task Loader.

This module handles loading ARC tasks from JSON files.
"""
from typing import Dict, List, Any, Optional, Tuple
import json
import os
import pathlib
import numpy as np

from ..dsl_utils.types import Grid
from ..utils.color import normalise_palette, denormalise


def load_task(task_id: str, data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a single task by ID.
    
    Args:
        task_id: The task ID
        data_path: Path to the data directory (default: ../arc-data-cleaned)
        
    Returns:
        A dictionary with 'train' and 'test' keys
    """
    if data_path is None:
        # Default to the arc-data-cleaned directory
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'arc-data-cleaned'
        )
    
    # Try to load from the evaluation challenges file
    challenges_file = os.path.join(data_path, 'arc-agi_evaluation_challenges.json')
    if os.path.exists(challenges_file):
        with open(challenges_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                return data[task_id]
    
    # Try to load from the training challenges file
    challenges_file = os.path.join(data_path, 'arc-agi_training_challenges.json')
    if os.path.exists(challenges_file):
        with open(challenges_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                return data[task_id]
    
    # Try to load from the test challenges file
    challenges_file = os.path.join(data_path, 'arc-agi_test_challenges.json')
    if os.path.exists(challenges_file):
        with open(challenges_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                return data[task_id]
    
    raise ValueError(f"Task {task_id} not found in any challenges file")


def load_solution(task_id: str, data_path: Optional[str] = None) -> np.ndarray:
    """
    Load the solution for a task.
    
    Args:
        task_id: The task ID
        data_path: Path to the data directory (default: ../arc-data-cleaned)
        
    Returns:
        The solution grid as a 2D numpy array
    """
    if data_path is None:
        # Default to the arc-data-cleaned directory
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'arc-data-cleaned'
        )
    
    # Try to load from the evaluation solutions file
    solutions_file = os.path.join(data_path, 'arc-agi_evaluation_solutions.json')
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                solution = np.array(data[task_id])
                # Handle 3D solutions by extracting the first 2D grid
                if len(solution.shape) == 3:
                    solution = solution[0]  # Extract the first 2D grid
                return solution
    
    # Try to load from the training solutions file
    solutions_file = os.path.join(data_path, 'arc-agi_training_solutions.json')
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                solution = np.array(data[task_id])
                # Handle 3D solutions by extracting the first 2D grid
                if len(solution.shape) == 3:
                    solution = solution[0]  # Extract the first 2D grid
                return solution
    
    raise ValueError(f"Solution for task {task_id} not found")


def load_id_list(json_file: str) -> List[str]:
    """
    Load a list of task IDs from a JSON file.
    
    Args:
        json_file: Path to the JSON file
        
    Returns:
        A list of task IDs
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def load_train_pairs(task: Dict[str, Any], normalise: bool = True) -> List[Tuple[Grid, Grid]]:
    """
    Extract training pairs from a task.
    
    Args:
        task: The task dictionary
        normalise: Whether to normalize the color palette
        
    Returns:
        A list of (input_grid, output_grid) pairs
    """
    pairs = []
    for example in task['train']:
        g_in = np.array(example['input'])
        g_out = np.array(example['output'])
        
        if normalise:
            g_in, _ = normalise_palette(g_in)
            g_out, _ = normalise_palette(g_out)  # same rule because colors match pair-wise
            
        pairs.append((Grid(g_in), Grid(g_out)))
    return pairs


def load_test_pair(task: Dict[str, Any], normalise: bool = True) -> Tuple[Grid, Dict[int, int]]:
    """
    Extract the test input from a task and normalize its palette.
    
    Args:
        task: The task dictionary
        normalise: Whether to normalize the color palette
        
    Returns:
        A tuple of (test_input_grid, color_mapping)
    """
    g_in = np.array(task['test'][0]['input'])
    
    if normalise:
        g_in, mapping = normalise_palette(g_in)
        return Grid(g_in), mapping
    else:
        return Grid(g_in), {}


def load_test_input(task: Dict[str, Any], normalise: bool = False) -> Grid:
    """
    Extract the test input from a task.
    
    Args:
        task: The task dictionary
        normalise: Whether to normalize the color palette
        
    Returns:
        The test input grid
    """
    if normalise:
        grid, _ = load_test_pair(task, normalise=True)
        return grid
    return Grid(task['test'][0]['input'])
