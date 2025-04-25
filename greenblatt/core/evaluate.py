"""evaluate.py – unit-tests candidate programs against train pairs

Filters out programs that:
• raise exceptions
• produce mismatched outputs or wrong grid sizes
"""
import asyncio
import json
import hashlib
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path

from sandbox.runner import run_in_sandbox

def grid_equals(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Check if two grids are equal."""
    if len(grid1) != len(grid2):
        return False
    
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False
    
    return True

def hash_code(code: str) -> str:
    """Create a hash of the code for deduplication."""
    # Normalize whitespace to avoid trivial differences
    normalized = "\n".join(line.strip() for line in code.strip().split("\n") if line.strip())
    return hashlib.md5(normalized.encode()).hexdigest()

async def evaluate_program(
    code: str, 
    train_examples: List[Dict[str, List[List[int]]]], 
    code_hashes: Set[str] = None
) -> Tuple[bool, str]:
    """
    Evaluate a program against training examples.
    
    Args:
        code: The Python code string containing a solve function
        train_examples: List of training examples with 'input' and 'output' keys
        code_hashes: Set of code hashes for deduplication
        
    Returns:
        (is_valid, code): Tuple of validation result and the code
    """
    # Check for duplicate code
    if code_hashes is not None:
        code_hash = hash_code(code)
        if code_hash in code_hashes:
            return False, code
        code_hashes.add(code_hash)
    
    # Prepare inputs and expected outputs
    inputs = [example["input"] for example in train_examples]
    expected_outputs = [example["output"] for example in train_examples]
    
    # Run the program in the sandbox
    outputs = await run_in_sandbox(code, inputs)
    
    # Check if all outputs match expected outputs
    if outputs is None:
        return False, code
    
    for output, expected in zip(outputs, expected_outputs):
        if output is None or not grid_equals(output, expected):
            return False, code
    
    return True, code

async def filter_valid_programs(
    programs: List[str], 
    train_examples: List[Dict[str, List[List[int]]]]
) -> List[str]:
    """
    Filter out invalid programs.
    
    Args:
        programs: List of Python code strings
        train_examples: List of training examples with 'input' and 'output' keys
        
    Returns:
        List of valid programs
    """
    valid_programs = []
    code_hashes = set()
    
    for program in programs:
        is_valid, code = await evaluate_program(program, train_examples, code_hashes)
        if is_valid:
            valid_programs.append(code)
    
    return valid_programs

async def majority_vote(
    valid_programs: List[str], 
    test_input: List[List[int]]
) -> Optional[List[List[int]]]:
    """
    Run all valid programs on the test input and take the majority vote.
    
    Args:
        valid_programs: List of valid Python code strings
        test_input: Test input grid
        
    Returns:
        The majority output grid or None if no valid output
    """
    if not valid_programs:
        return None
    
    # Run all programs on the test input
    all_outputs = []
    for program in valid_programs:
        outputs = await run_in_sandbox(program, [test_input])
        if outputs and outputs[0] is not None:
            all_outputs.append(json.dumps(outputs[0]))
    
    if not all_outputs:
        return None
    
    # Count occurrences of each output
    output_counts = {}
    for output in all_outputs:
        output_counts[output] = output_counts.get(output, 0) + 1
    
    # Find the majority output
    majority_output = max(output_counts.items(), key=lambda x: x[1])[0]
    
    return json.loads(majority_output)

async def evaluate_task(
    task_data: Dict[str, Any], 
    task_id: str, 
    programs: List[str]
) -> Dict[str, Any]:
    """
    Evaluate programs for a specific task.
    
    Args:
        task_data: Dictionary of task data
        task_id: Task ID
        programs: List of Python code strings
        
    Returns:
        Dictionary with evaluation results
    """
    train_examples = task_data[task_id]["train"]
    test_input = task_data[task_id]["test"][0]["input"]
    
    # Filter valid programs
    valid_programs = await filter_valid_programs(programs, train_examples)
    
    # Get majority vote for test input
    majority_output = await majority_vote(valid_programs, test_input)
    
    return {
        "task_id": task_id,
        "total_programs": len(programs),
        "valid_programs": len(valid_programs),
        "valid_ratio": len(valid_programs) / len(programs) if programs else 0,
        "majority_output": majority_output,
        "valid_program_examples": valid_programs[:3] if valid_programs else []
    }
