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
    # Check if dimensions match
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

async def evaluate_programs_batch(
    programs: List[str],
    train_examples: List[Dict[str, List[List[int]]]],
    test_input: List[List[int]],
    code_hashes: Set[str] = None
) -> Dict[str, Any]:
    """
    Evaluate all programs against training examples and test input in a single batch.
    
    Args:
        programs: List of Python code strings
        train_examples: List of training examples with 'input' and 'output' keys
        test_input: Test input grid
        code_hashes: Set of code hashes for deduplication
        
    Returns:
        Dictionary with evaluation results including valid programs and their outputs
    """
    if not programs:
        return {
            "valid_programs": [],
            "training_predictions": {},
            "test_predictions": {},
            "first_program_test_output": None
        }
    
    # Deduplicate programs
    unique_programs = []
    program_indices = {}  # Maps program index in unique_programs to original index
    
    if code_hashes is None:
        code_hashes = set()
        
    for i, program in enumerate(programs):
        code_hash = hash_code(program)
        if code_hash not in code_hashes:
            code_hashes.add(code_hash)
            program_indices[len(unique_programs)] = i
            unique_programs.append(program)
    
    # Prepare all inputs (training + test)
    train_inputs = [example["input"] for example in train_examples]
    expected_outputs = [example["output"] for example in train_examples]
    all_inputs = train_inputs + [test_input]
    
    # Run all programs on all inputs in a single batch
    all_results = {}
    for i, program in enumerate(unique_programs):
        # Run the program on all inputs
        outputs = await run_in_sandbox(program, all_inputs)
        
        if outputs is None:
            continue
            
        # Split results into training and test outputs
        train_outputs = outputs[:len(train_inputs)]
        test_output = outputs[len(train_inputs)] if len(outputs) > len(train_inputs) else None
        
        # Check if all training outputs match expected outputs
        is_valid = True
        for output, expected in zip(train_outputs, expected_outputs):
            if output is None or not grid_equals(output, expected):
                is_valid = False
                break
        
        # Store results for this program
        orig_index = program_indices[i]
        all_results[orig_index] = {
            "program": program,
            "is_valid": is_valid,
            "training_outputs": train_outputs,
            "test_output": test_output
        }
    
    # Collect valid programs and their outputs
    valid_programs = []
    training_predictions = {}
    test_predictions = {}
    first_program_test_output = None
    
    # Get the first program's test output (valid or not)
    if programs and 0 in all_results and all_results[0]["test_output"] is not None:
        first_program_test_output = all_results[0]["test_output"]
    
    # Process results in original program order
    for i in range(len(programs)):
        if i in all_results and all_results[i]["is_valid"]:
            program = all_results[i]["program"]
            valid_programs.append(program)
            
            # Store training predictions for this valid program
            training_predictions[program] = all_results[i]["training_outputs"]
            
            # Store test prediction for this valid program
            if all_results[i]["test_output"] is not None:
                test_predictions[program] = all_results[i]["test_output"]
    
    return {
        "valid_programs": valid_programs,
        "training_predictions": training_predictions,
        "test_predictions": test_predictions,
        "first_program_test_output": first_program_test_output
    }

async def majority_vote(
    valid_programs: List[str],
    test_predictions: Dict[str, List[List[int]]]
) -> Optional[List[List[int]]]:
    """
    Take the majority vote of all valid programs' test predictions.
    
    Args:
        valid_programs: List of valid Python code strings
        test_predictions: Dictionary mapping programs to their test predictions
        
    Returns:
        The majority output grid or None if no valid output
    """
    if not valid_programs:
        return None
    
    # Collect all outputs for the test input
    all_outputs = []
    for program in valid_programs:
        if program in test_predictions and test_predictions[program] is not None:
            all_outputs.append(json.dumps(test_predictions[program]))
    
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
    solutions_data: Dict[str, Any],
    task_id: str, 
    programs: List[str]
) -> Dict[str, Any]:
    """
    Evaluate programs for a specific task.
    
    Args:
        task_data: Dictionary of task data
        solutions_data: Dictionary of solutions data
        task_id: Task ID
        programs: List of Python code strings
        
    Returns:
        Dictionary with evaluation results
    """
    train_examples = task_data[task_id]["train"]
    test_input = task_data[task_id]["test"][0]["input"]
    
    # Get ground truth for test example from solutions data if available
    test_output = None
    if task_id in solutions_data:
        # The solutions data is just an array of outputs
        if isinstance(solutions_data[task_id], list) and len(solutions_data[task_id]) > 0:
            test_output = solutions_data[task_id][0]
    
    # Evaluate all programs in a single batch
    batch_results = await evaluate_programs_batch(programs, train_examples, test_input)
    
    valid_programs = batch_results["valid_programs"]
    training_predictions = batch_results["training_predictions"]
    test_predictions = batch_results["test_predictions"]
    first_program_output = batch_results["first_program_test_output"]
    
    # Get majority vote for test input if there are valid programs
    majority_output = await majority_vote(valid_programs, test_predictions) if valid_programs else None
    
    # Check if the majority output matches the ground truth
    test_correct = False
    if majority_output and test_output:
        test_correct = grid_equals(majority_output, test_output)
    
    return {
        "task_id": task_id,
        "total_programs": len(programs),
        "valid_programs": len(valid_programs),
        "valid_ratio": len(valid_programs) / len(programs) if programs else 0,
        "majority_output": majority_output,
        "first_program_output": first_program_output,
        "test_output": test_output,  # Add the ground truth
        "test_correct": test_correct,  # Add whether the test output is correct
        "valid_program_examples": valid_programs[:3] if valid_programs else [],
        "training_predictions": training_predictions  # Add training predictions for visualization
    }
