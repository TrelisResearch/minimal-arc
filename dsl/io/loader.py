"""
ARC Task Loader.

This module handles loading ARC tasks from JSON files.
"""
from typing import Dict, List, Any, Optional
import json
import os
import pathlib

from ..dsl_utils.types import Grid


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


def load_solution(task_id: str, data_path: Optional[str] = None) -> List[List[int]]:
    """
    Load the solution for a task.
    
    Args:
        task_id: The task ID
        data_path: Path to the data directory (default: ../arc-data-cleaned)
        
    Returns:
        The solution grid
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
                return data[task_id]
    
    # Try to load from the training solutions file
    solutions_file = os.path.join(data_path, 'arc-agi_training_solutions.json')
    if os.path.exists(solutions_file):
        with open(solutions_file, 'r') as f:
            data = json.load(f)
            if task_id in data:
                return data[task_id]
    
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


def load_train_pairs(task: Dict[str, Any]) -> List[tuple]:
    """
    Extract training pairs from a task.
    
    Args:
        task: The task dictionary
        
    Returns:
        A list of (input_grid, output_grid) pairs
    """
    pairs = []
    for example in task['train']:
        inp = Grid(example['input'])
        out = Grid(example['output'])
        pairs.append((inp, out))
    return pairs


def load_test_input(task: Dict[str, Any]) -> Grid:
    """
    Extract the test input from a task.
    
    Args:
        task: The task dictionary
        
    Returns:
        The test input grid
    """
    return Grid(task['test'][0]['input'])
