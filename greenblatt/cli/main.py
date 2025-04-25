"""CLI entrypoint.

Options
--task-id <hash>
--task-file <json list>  # e.g. arc-data-cleaned/mit-easy.json
--k <int>  # samples per task
--concurrency <int>
--temperature <float>  # temperature for generation
--top-p <float>  # top-p sampling parameter
--top-k <int>  # top-k sampling parameter
--save-results <path>  # save results to JSON file
--debug  # debug mode

Samples programs, filters them, majority-votes test output, and prints / saves results.
"""
import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.generate_programs import generate_programs_for_task, load_task_data, load_task_ids
from core.evaluate import evaluate_task
from viz.show_grids import visualize_task
from sandbox.runner import cleanup

async def process_single_task(
    task_data: Dict[str, Any],
    solutions_data: Dict[str, Any],
    task_id: str,
    k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    concurrency: int,
    visualize: bool = False,
    save_dir: Optional[Path] = None,
    evaluation_timeout: float = 3.0
) -> Dict[str, Any]:
    """
    Process a single task.
    
    Args:
        task_data: Dictionary of task data
        solutions_data: Dictionary of solutions data
        task_id: Task ID
        k: Number of programs to generate
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        concurrency: Number of concurrent API calls
        visualize: Whether to visualize the results
        save_dir: Directory to save visualizations
        evaluation_timeout: Maximum time for evaluation phase
        
    Returns:
        Dictionary with results for the task
    """
    print(f"\n==================================================")
    print(f"Processing task: {task_id}")
    print(f"==================================================")
    
    # Generate programs
    print(f"Generating {k} programs...")
    start_time = time.time()
    programs = []
    
    async for program, token_usage in generate_programs_for_task(
        task_data=task_data,
        task_id=task_id,
        k=k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        concurrency=concurrency
    ):
        programs.append(program)
        if token_usage:
            print(f"Token usage: {token_usage}")
    
    # Evaluate programs with a separate timeout
    print(f"Evaluating {len(programs)} programs with timeout {evaluation_timeout}s...")
    try:
        evaluation_result = await asyncio.wait_for(
            evaluate_task(
                task_data=task_data,
                solutions_data=solutions_data,
                task_id=task_id,
                programs=programs
            ),
            timeout=evaluation_timeout
        )
        print(f"Evaluation completed successfully")
    except asyncio.TimeoutError:
        print(f"Evaluation timed out after {evaluation_timeout}s")
        evaluation_result = {
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
            "error": f"Evaluation timed out after {evaluation_timeout}s"
        }
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nResults for task {task_id}:")
    print(f"Total programs: {evaluation_result['total_programs']}")
    print(f"Valid programs: {evaluation_result['valid_programs']} ({evaluation_result['valid_ratio']:.2%})")
    if 'test_correct' in evaluation_result:
        print(f"Test correct: {'Yes' if evaluation_result['test_correct'] else 'No'}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Visualize if requested
    if visualize:
        print("Visualizing results...")
        # Use majority_output if available, otherwise use first_program_output
        candidate_output = evaluation_result.get('majority_output') or evaluation_result.get('first_program_output')
        
        # Get valid programs from the result
        valid_programs = evaluation_result.get('valid_program_examples', [])
        
        # Get training predictions from the result
        training_predictions = evaluation_result.get('training_predictions', {})
        
        # Check if we have valid programs but the test failed
        has_valid_programs = evaluation_result.get('valid_programs', 0) > 0
        test_failed = not evaluation_result.get('test_correct', False)
        
        if has_valid_programs and test_failed:
            print("Note: Valid programs exist but test failed. Showing training predictions for comparison.")
        
        visualize_task(
            task_data=task_data,
            solutions_data=solutions_data,
            task_id=task_id,
            candidate_output=candidate_output,
            valid_programs=valid_programs,
            save_path=str(save_dir / f"{task_id}.png") if save_dir else None,
            training_predictions=training_predictions
        )
    
    return evaluation_result

async def debug_task(
    task_id: str,
    data_file: str,
    solutions_file: Optional[str] = None,
    k: int = 8,
    temperature: float = 0.8,
    top_p: float = 1.0,
    top_k: int = 40,
    concurrency: int = 8,
    save_results: Optional[str] = None,
    visualize: bool = False,
    save_viz: Optional[str] = None,
    extended_timeout: bool = False  # Default to normal timeouts
) -> Dict[str, Any]:
    """
    Debug a single task with extended timeouts and detailed logging.
    
    Args:
        task_id: Task ID to debug
        data_file: Path to data file
        solutions_file: Path to solutions file
        k: Number of programs to generate
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        concurrency: Number of concurrent API calls
        save_results: Path to save results
        visualize: Whether to visualize the results
        save_viz: Directory to save visualizations
        extended_timeout: Whether to use extended timeouts
        
    Returns:
        Dictionary with results for the task
    """
    print(f"\n==================================================")
    print(f"DEBUGGING TASK: {task_id}")
    print(f"==================================================")
    
    # Load task data
    print(f"Loading task data from {data_file}...")
    task_data = load_task_data(data_file)
    
    # Load solutions data if provided
    solutions_data = {}
    if solutions_file:
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        print(f"Loaded solutions data from {solutions_file}")
    else:
        # Try to find solutions file based on data file
        if 'challenges' in data_file:
            potential_solutions_file = data_file.replace('challenges', 'solutions')
            if os.path.exists(potential_solutions_file):
                with open(potential_solutions_file, 'r') as f:
                    solutions_data = json.load(f)
                print(f"Automatically loaded solutions from {potential_solutions_file}")
    
    # Create save directory if needed
    save_dir = None
    if save_viz:
        save_dir = Path(save_viz)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set debug timeouts
    if extended_timeout:
        # Set longer timeouts for debugging
        sandbox_timeout = 5.0  # 5 seconds for sandbox calls
        evaluation_timeout = 10.0  # 10 seconds for evaluation
        task_timeout = 60.0  # 60 seconds for the entire task
        print(f"Using extended timeouts: sandbox={sandbox_timeout}s, evaluation={evaluation_timeout}s, task={task_timeout}s")
    else:
        # Use normal timeouts
        sandbox_timeout = 3.0  # 3 seconds for sandbox calls
        evaluation_timeout = 3.0  # 3 seconds for evaluation
        task_timeout = 30.0  # 30 seconds for the entire task
        print(f"Using normal timeouts: sandbox={sandbox_timeout}s, evaluation={evaluation_timeout}s, task={task_timeout}s")
    
    # Override the timeout in the runner module
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sandbox.runner import set_default_timeout
    set_default_timeout(sandbox_timeout)
    
    try:
        print(f"Generating {k} programs...")
        # Generate programs
        programs = []
        async for program, token_usage in generate_programs_for_task(
            task_data=task_data,
            task_id=task_id,
            k=k,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            concurrency=concurrency
        ):
            programs.append(program)
            print(f"Token usage: {token_usage}")
        
        print(f"Generated {len(programs)} programs")
        
        # Evaluate programs
        print(f"Evaluating programs with timeout {evaluation_timeout}s...")
        evaluation_result = await evaluate_task(
            task_data=task_data,
            solutions_data=solutions_data,
            task_id=task_id,
            programs=programs,
            timeout=evaluation_timeout
        )
        
        print(f"Evaluation result: {json.dumps(evaluation_result, indent=2)}")
        
        # Visualize if requested
        if visualize:
            print("Visualizing results...")
            await visualize_task(
                task_data=task_data,
                solutions_data=solutions_data,
                task_id=task_id,
                result=evaluation_result,
                save_dir=save_dir
            )
        
        # Save results if requested
        if save_results:
            with open(save_results, 'w') as f:
                json.dump({task_id: evaluation_result}, f, indent=2)
            print(f"Saved results to {save_results}")
        
        return evaluation_result
    
    except Exception as e:
        print(f"Error debugging task {task_id}: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def process_task_file(
    task_file: str,
    task_ids_file: str,
    k: int,
    temperature: float,
    top_p: float,
    top_k: int,
    concurrency: int,
    data_file: str,
    solutions_file: Optional[str] = None,
    save_results: Optional[str] = None,
    visualize: bool = False,
    save_viz: Optional[str] = None,
    task_timeout: float = 30.0,
    max_tasks: Optional[int] = None,
    evaluation_timeout: float = 3.0
) -> Dict[str, Any]:
    """
    Process tasks from a file.
    
    Args:
        task_file: Path to task list file
        task_ids_file: Path to task IDs file
        k: Number of programs to generate per task
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        concurrency: Number of concurrent API calls
        data_file: Path to data file
        solutions_file: Path to solutions file
        save_results: Path to save results
        visualize: Whether to visualize the results
        save_viz: Directory to save visualizations
        task_timeout: Maximum time to spend on a single task (in seconds)
        max_tasks: Maximum number of tasks to process (None for all)
        evaluation_timeout: Maximum time for the evaluation phase (in seconds)
        
    Returns:
        Dictionary with results for all tasks
    """
    # Load task IDs
    task_ids = load_task_ids(task_ids_file)
    print(f"Loaded {len(task_ids)} task IDs from {task_ids_file}")
    
    # Apply max_tasks limit if specified
    if max_tasks is not None and max_tasks > 0:
        task_ids = task_ids[:max_tasks]
        print(f"Limited to processing {max_tasks} tasks")
    
    # Load task data
    task_data = load_task_data(data_file)
    print(f"Loaded task data from {data_file}")
    
    # Load solutions data if provided
    solutions_data = {}
    if solutions_file:
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
        print(f"Loaded solutions data from {solutions_file}")
    else:
        # Try to find solutions file based on data file
        if 'challenges' in data_file:
            potential_solutions_file = data_file.replace('challenges', 'solutions')
            if os.path.exists(potential_solutions_file):
                with open(potential_solutions_file, 'r') as f:
                    solutions_data = json.load(f)
                print(f"Automatically loaded solutions from {potential_solutions_file}")
    
    # Create save directory if needed
    save_dir = None
    if save_viz:
        save_dir = Path(save_viz)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task
    results = {}
    for task_id in task_ids:
        if task_id not in task_data:
            print(f"Task ID {task_id} not found in {data_file}")
            continue
        
        try:
            # Process the task with a timeout
            print(f"\n==================================================")
            print(f"Processing task: {task_id} (timeout: {task_timeout}s)")
            print(f"==================================================")
            
            task_start_time = time.time()
            
            try:
                # Use asyncio.wait_for to enforce a timeout for the entire task
                result = await asyncio.wait_for(
                    process_single_task(
                        task_data=task_data,
                        solutions_data=solutions_data,
                        task_id=task_id,
                        k=k,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        concurrency=concurrency,
                        visualize=visualize,
                        save_dir=save_dir,
                        evaluation_timeout=evaluation_timeout
                    ),
                    timeout=task_timeout
                )
                
                task_elapsed_time = time.time() - task_start_time
                print(f"Task {task_id} completed in {task_elapsed_time:.2f}s")
                
            except asyncio.TimeoutError:
                task_elapsed_time = time.time() - task_start_time
                print(f"Task {task_id} timed out after {task_elapsed_time:.2f}s")
                result = {
                    "task_id": task_id,
                    "error": f"Task timed out after {task_timeout} seconds",
                    "timed_out": True
                }
            
            results[task_id] = result
            
            # Save incremental results after each task
            if save_results:
                with open(save_results, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved incremental results to {save_results}")
                
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            results[task_id] = {"error": str(e)}
            
            # Save results even if there was an error
            if save_results:
                with open(save_results, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved results after error to {save_results}")
    
    # Calculate overall statistics
    total_tasks = len(results)
    valid_tasks = sum(1 for r in results.values() if isinstance(r, dict) and r.get('valid_programs', 0) > 0)
    correct_tasks = sum(1 for r in results.values() if isinstance(r, dict) and r.get('test_correct', False))
    timed_out_tasks = sum(1 for r in results.values() if isinstance(r, dict) and r.get('timed_out', False))
    
    # Print overall statistics
    print(f"\n==================================================")
    print(f"Overall Results")
    print(f"==================================================")
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks with valid programs: {valid_tasks} ({valid_tasks/total_tasks:.2%})")
    print(f"Tasks with correct test outputs: {correct_tasks} ({correct_tasks/total_tasks:.2%})")
    print(f"Tasks that timed out: {timed_out_tasks} ({timed_out_tasks/total_tasks:.2%})")
    
    # Save final results
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved final results to {save_results}")
    
    return results

async def main_async() -> int:
    """Main async function."""
    parser = argparse.ArgumentParser(description="Greenblatt ARC Demo")
    
    # Task selection
    parser.add_argument("--task-id", type=str,
                        help="Task ID to process")
    parser.add_argument("--task-file", type=str,
                        help="Path to task list file")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode")
    
    # Data files
    parser.add_argument("--data-file", type=str, default="../arc-data-cleaned/arc-agi_evaluation_challenges.json",
                        help="Path to data file")
    parser.add_argument("--solutions-file", type=str,
                        help="Path to solutions file")
    
    # Generation parameters
    parser.add_argument("--k", type=int, default=8,
                        help="Number of programs to generate per task")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation (0.0-2.0)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Top-p sampling parameter (0.0-1.0)")
    parser.add_argument("--top-k", type=int, default=40,
                        help="Top-k sampling parameter (1-100)")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Number of concurrent API calls")
    
    # Output options
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results")
    parser.add_argument("--save-results", type=str,
                        help="Path to save results JSON")
    parser.add_argument("--save-viz", type=str,
                        help="Directory to save visualizations")
    parser.add_argument("--task-timeout", type=float, default=30.0,
                        help="Maximum time to spend on a single task (in seconds)")
    parser.add_argument("--max-tasks", type=int,
                        help="Maximum number of tasks to process")
    parser.add_argument("--evaluation-timeout", type=float, default=3.0,
                        help="Maximum time for the evaluation phase (in seconds)")
    
    args = parser.parse_args()
    
    try:
        if args.debug:
            # Debug mode
            await debug_task(
                task_id=args.task_id,
                data_file=args.data_file,
                solutions_file=args.solutions_file,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                concurrency=args.concurrency,
                save_results=args.save_results,
                visualize=args.visualize,
                save_viz=args.save_viz
            )
        elif args.task_id:
            # Process a single task
            with open(args.data_file, 'r') as f:
                task_data = json.load(f)
            
            # Load solutions data if provided
            solutions_data = {}
            if args.solutions_file:
                with open(args.solutions_file, 'r') as f:
                    solutions_data = json.load(f)
            else:
                # Try to find solutions file based on data file
                if 'challenges' in args.data_file:
                    potential_solutions_file = args.data_file.replace('challenges', 'solutions')
                    if os.path.exists(potential_solutions_file):
                        with open(potential_solutions_file, 'r') as f:
                            solutions_data = json.load(f)
                        print(f"Automatically loaded solutions from {potential_solutions_file}")
            
            # Create save directory if needed
            save_dir = None
            if args.save_viz:
                save_dir = Path(args.save_viz)
                save_dir.mkdir(parents=True, exist_ok=True)
            
            await process_single_task(
                task_data=task_data,
                solutions_data=solutions_data,
                task_id=args.task_id,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                concurrency=args.concurrency,
                visualize=args.visualize,
                save_dir=save_dir
            )
        elif args.task_file:
            # Process tasks from a file
            await process_task_file(
                task_file=args.task_file,
                task_ids_file=args.task_file,
                k=args.k,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                concurrency=args.concurrency,
                data_file=args.data_file,
                solutions_file=args.solutions_file,
                save_results=args.save_results,
                visualize=args.visualize,
                save_viz=args.save_viz,
                task_timeout=args.task_timeout,
                max_tasks=args.max_tasks,
                evaluation_timeout=args.evaluation_timeout
            )
        else:
            print("Error: Either --task-id, --task-file, or --debug must be specified")
            sys.exit(1)
    finally:
        # Clean up resources
        await cleanup()
    
    return 0

def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
