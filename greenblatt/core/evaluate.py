"""evaluate.py – unit-tests candidate programs against train pairs

Filters out programs that:
• raise exceptions
• produce mismatched outputs or wrong grid sizes
"""
import asyncio
import json
import hashlib
import time
from typing import List, Dict, Any, Set, Tuple, Optional
from pathlib import Path

from sandbox.runner import run_batch_programs

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
    code_hashes: Set[str] = None,
    timeout: float = 30.0  # Add timeout parameter with default of 30 seconds
) -> Dict[str, Any]:
    """
    Evaluate all programs against training examples and test input in a single batch.
    
    Args:
        programs: List of Python code strings
        train_examples: List of training examples with 'input' and 'output' keys
        test_input: Test input grid
        code_hashes: Set of code hashes for deduplication
        timeout: Maximum time to wait for evaluation (in seconds)
        
    Returns:
        Dictionary with evaluation results including valid programs and their outputs
    """
    print(f"Starting evaluation of {len(programs)} programs with {len(train_examples)} training examples...")
    start_time = time.time()
    
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
    
    print("Deduplicating programs...")
    for i, program in enumerate(programs):
        code_hash = hash_code(program)
        if code_hash not in code_hashes:
            code_hashes.add(code_hash)
            program_indices[len(unique_programs)] = i
            unique_programs.append(program)
    
    print(f"After deduplication: {len(unique_programs)} unique programs")
    
    # Prepare all inputs (training + test)
    train_inputs = [example["input"] for example in train_examples]
    expected_outputs = [example["output"] for example in train_examples]
    all_inputs = train_inputs + [test_input]
    
    # Run all programs on all inputs in a single batch with timeout
    print(f"Running batch execution at {time.strftime('%H:%M:%S')}...")
    batch_exec_start = time.time()
    
    try:
        print(f"Running batch execution with {timeout}s timeout...")
        batch_results = await asyncio.wait_for(
            run_batch_programs(unique_programs, all_inputs, timeout=timeout),
            timeout=timeout
        )
        batch_exec_time = time.time() - batch_exec_start
        print(f"Batch execution completed in {batch_exec_time:.2f}s")
    except asyncio.TimeoutError:
        batch_exec_time = time.time() - batch_exec_start
        print(f"Batch execution timed out after {batch_exec_time:.2f}s")
        # Return empty results if timeout occurs
        return {
            "valid_programs": [],
            "training_predictions": {},
            "test_predictions": {},
            "first_program_test_output": None,
            "timed_out": True
        }
    
    # Process the batch results
    print(f"Processing results for {len(programs)} programs...")
    process_start = time.time()
    
    all_results = {}
    for i, program_idx in enumerate(program_indices.keys()):
        if program_idx in batch_results:
            outputs = batch_results[program_idx]
            
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
            orig_index = program_indices[program_idx]
            all_results[orig_index] = {
                "program": unique_programs[i],
                "is_valid": is_valid,
                "training_outputs": train_outputs,
                "test_output": test_output
            }
    
    process_time = time.time() - process_start
    print(f"Result processing completed in {process_time:.2f}s")
    
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
    
    elapsed_time = time.time() - start_time
    print(f"Evaluation completed in {elapsed_time:.2f}s with {len(valid_programs)} valid programs")
    
    return {
        "valid_programs": valid_programs,
        "training_predictions": training_predictions,
        "test_predictions": test_predictions,
        "first_program_test_output": first_program_test_output
    }

async def majority_vote(
    valid_programs: List[str],
    test_predictions: Dict[str, List[List[int]]],
    timeout: float = 5.0  # Add timeout parameter
) -> Optional[List[List[int]]]:
    """
    Take the majority vote of all valid programs' test predictions.
    
    Args:
        valid_programs: List of valid Python code strings
        test_predictions: Dictionary mapping programs to their test predictions
        timeout: Maximum time to wait for voting (in seconds)
        
    Returns:
        The majority output grid or None if no valid output
    """
    if not valid_programs:
        return None
    
    # Use a timeout to prevent hanging
    try:
        # This operation is mostly CPU-bound, but we'll add a timeout just in case
        return await asyncio.wait_for(_majority_vote_impl(valid_programs, test_predictions), timeout)
    except asyncio.TimeoutError:
        print(f"Majority voting timed out after {timeout} seconds")
        return None

async def _majority_vote_impl(valid_programs: List[str], test_predictions: Dict[str, List[List[int]]]) -> Optional[List[List[int]]]:
    """Implementation of majority voting without timeout handling."""
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
    programs: List[str],
    timeout: float = 60.0  # Default timeout of 60 seconds
) -> Dict[str, Any]:
    """
    Evaluate programs for a task.
    
    Args:
        task_data: Dictionary of task data
        solutions_data: Dictionary of solutions data
        task_id: Task ID
        programs: List of Python code strings
        timeout: Maximum time to wait for the entire evaluation (in seconds)
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Starting evaluation for task {task_id} with {len(programs)} programs (timeout: {timeout}s)...")
    overall_start_time = time.time()
    
    try:
        # Use a timeout for the entire evaluation process
        return await asyncio.wait_for(
            _evaluate_task_impl(task_data, solutions_data, task_id, programs),
            timeout
        )
    except asyncio.TimeoutError:
        elapsed_time = time.time() - overall_start_time
        print(f"Task evaluation timed out after {elapsed_time:.2f} seconds")
        return {
            "task_id": task_id,
            "total_programs": len(programs),
            "valid_programs": 0,
            "valid_ratio": 0,
            "majority_output": None,
            "first_program_output": None,
            "test_output": None,
            "test_correct": False,
            "valid_program_examples": [],
            "training_predictions": {},
            "timed_out": True,
            "elapsed_time": elapsed_time,
            "error": f"Task evaluation timed out after {elapsed_time:.2f} seconds"
        }

async def _evaluate_task_impl(
    task_data: Dict[str, Any], 
    solutions_data: Dict[str, Any],
    task_id: str, 
    programs: List[str]
) -> Dict[str, Any]:
    """Implementation of task evaluation without timeout handling."""
    start_time = time.time()
    
    train_examples = task_data[task_id]["train"]
    test_input = task_data[task_id]["test"][0]["input"]
    
    # Get ground truth for test example from solutions data if available
    test_output = None
    if task_id in solutions_data:
        # The solutions data is just an array of outputs
        if isinstance(solutions_data[task_id], list) and len(solutions_data[task_id]) > 0:
            test_output = solutions_data[task_id][0]
    
    # Evaluate all programs in a single batch with timeout
    print(f"Evaluating {len(programs)} programs on {len(train_examples)} training examples...")
    eval_start_time = time.time()
    
    try:
        # Use a timeout for the batch evaluation
        batch_results = await asyncio.wait_for(
            evaluate_programs_batch(programs, train_examples, test_input),
            timeout=30.0  # 30 second timeout for batch evaluation
        )
        eval_time = time.time() - eval_start_time
        print(f"Batch evaluation completed in {eval_time:.2f}s")
    except asyncio.TimeoutError:
        eval_time = time.time() - eval_start_time
        print(f"Batch evaluation timed out after {eval_time:.2f}s")
        return {
            "task_id": task_id,
            "total_programs": len(programs),
            "valid_programs": 0,
            "valid_ratio": 0,
            "majority_output": None,
            "first_program_output": None,
            "test_output": test_output,
            "test_correct": False,
            "valid_program_examples": [],
            "training_predictions": {},
            "timed_out": True,
            "elapsed_time": eval_time,
            "error": f"Batch evaluation timed out after {eval_time:.2f} seconds"
        }
    
    valid_programs = batch_results.get("valid_programs", [])
    valid_count = len(valid_programs)
    print(f"Found {valid_count} valid programs out of {len(programs)}")
    
    training_predictions = batch_results.get("training_predictions", {})
    test_predictions = batch_results.get("test_predictions", {})
    first_program_output = batch_results.get("first_program_test_output")
    
    # Get majority vote for test input if there are valid programs
    if valid_programs:
        try:
            print(f"Running majority voting with {valid_count} programs...")
            vote_start_time = time.time()
            majority_output = await asyncio.wait_for(
                majority_vote(valid_programs, test_predictions),
                timeout=5.0  # 5 second timeout for majority voting
            )
            vote_time = time.time() - vote_start_time
            print(f"Majority voting completed in {vote_time:.2f}s")
        except asyncio.TimeoutError:
            vote_time = time.time() - vote_start_time
            print(f"Majority voting timed out after {vote_time:.2f}s")
            majority_output = None
    else:
        majority_output = None
    
    # Check if the majority output matches the ground truth
    test_correct = False
    if majority_output and test_output:
        test_correct = grid_equals(majority_output, test_output)
    
    elapsed_time = time.time() - start_time
    print(f"Task {task_id} evaluation completed in {elapsed_time:.2f}s - Test correct: {test_correct}")
    
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
        "training_predictions": training_predictions,  # Add training predictions for visualization
        "timed_out": batch_results.get("timed_out", False),  # Indicate if evaluation timed out
        "elapsed_time": elapsed_time
    }
