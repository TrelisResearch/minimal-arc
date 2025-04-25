"""CLI entrypoint.

Options
--task-id <hash>
--task-file <json list>  # e.g. arc-data-cleaned/mit-easy.json
--k <int>  # samples per task
--concurrency <int>
--temperature <float>  # temperature for generation
--save-results <path>  # save results to JSON file

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
    concurrency: int,
    visualize: bool,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Process a single task.
    
    Args:
        task_data: Dictionary of task data
        solutions_data: Dictionary of solutions data
        task_id: Task ID
        k: Number of programs to generate
        temperature: Temperature for generation
        concurrency: Number of concurrent API calls
        visualize: Whether to visualize the results
        save_dir: Directory to save visualizations
        
    Returns:
        Dictionary with task results
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
        concurrency=concurrency
    ):
        programs.append(program)
        if token_usage:
            print(f"Token usage: {token_usage}")
    
    # Evaluate programs
    print(f"Evaluating {len(programs)} programs...")
    result = await evaluate_task(
        task_data=task_data,
        solutions_data=solutions_data,
        task_id=task_id,
        programs=programs
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"\nResults for task {task_id}:")
    print(f"Total programs: {result['total_programs']}")
    print(f"Valid programs: {result['valid_programs']} ({result['valid_ratio']:.2%})")
    if 'test_correct' in result:
        print(f"Test correct: {'Yes' if result['test_correct'] else 'No'}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Visualize if requested
    if visualize:
        print("Visualizing results...")
        # Use majority_output if available, otherwise use first_program_output
        candidate_output = result.get('majority_output') or result.get('first_program_output')
        
        # Get valid programs from the result
        valid_programs = result.get('valid_program_examples', [])
        
        # Check if we have valid programs but the test failed
        has_valid_programs = result.get('valid_programs', 0) > 0
        test_failed = not result.get('test_correct', False)
        
        if has_valid_programs and test_failed:
            print("Note: Valid programs exist but test failed. Showing training predictions for comparison.")
        
        visualize_task(
            task_data=task_data,
            solutions_data=solutions_data,
            task_id=task_id,
            candidate_output=candidate_output,
            valid_programs=valid_programs,
            save_path=str(save_dir / f"{task_id}.png") if save_dir else None
        )
    
    return result

async def process_task_file(
    task_file: str,
    task_ids_file: Optional[str],
    k: int,
    temperature: float,
    concurrency: int,
    data_file: str,
    solutions_file: Optional[str] = None,
    save_results: Optional[str] = None,
    visualize: bool = False,
    save_viz: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process tasks from a file.
    
    Args:
        task_file: Path to JSON file with task data
        task_ids_file: Path to JSON file with task IDs
        k: Number of programs to generate
        temperature: Temperature for generation
        concurrency: Number of concurrent API calls
        data_file: Path to JSON file with task data
        solutions_file: Path to JSON file with solutions data
        save_results: Path to save results JSON
        visualize: Whether to visualize the results
        save_viz: Directory to save visualizations
        
    Returns:
        Dictionary with results for all tasks
    """
    # Load task data
    with open(data_file, 'r') as f:
        task_data = json.load(f)
    
    # Load solutions data if provided
    solutions_data = {}
    if solutions_file:
        with open(solutions_file, 'r') as f:
            solutions_data = json.load(f)
    else:
        # Try to find solutions file based on data file
        if 'challenges' in data_file:
            potential_solutions_file = data_file.replace('challenges', 'solutions')
            if os.path.exists(potential_solutions_file):
                with open(potential_solutions_file, 'r') as f:
                    solutions_data = json.load(f)
                print(f"Automatically loaded solutions from {potential_solutions_file}")
    
    # Load task IDs
    if task_ids_file:
        with open(task_ids_file, 'r') as f:
            task_ids = json.load(f)
    else:
        # Use all task IDs from the task file
        task_ids = list(task_data.keys())
    
    print(f"Processing {len(task_ids)} tasks from {data_file}")
    
    # Create save directory if needed
    save_dir = None
    if save_viz:
        save_dir = Path(save_viz)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each task
    results = {}
    for task_id in task_ids:
        result = await process_single_task(
            task_data=task_data,
            solutions_data=solutions_data,
            task_id=task_id,
            k=k,
            temperature=temperature,
            concurrency=concurrency,
            visualize=visualize,
            save_dir=save_dir
        )
        results[task_id] = result
    
    # Calculate overall statistics
    total_tasks = len(results)
    valid_tasks = sum(1 for r in results.values() if r['valid_programs'] > 0)
    correct_tasks = sum(1 for r in results.values() if r.get('test_correct', False))
    
    # Print overall statistics
    print(f"\n==================================================")
    print(f"Overall Results")
    print(f"==================================================")
    print(f"Total tasks: {total_tasks}")
    print(f"Tasks with valid programs: {valid_tasks} ({valid_tasks/total_tasks:.2%})")
    print(f"Tasks with correct test outputs: {correct_tasks} ({correct_tasks/total_tasks:.2%})")
    
    # Save results if requested
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {save_results}")
    
    return results

async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="ARC Demo")
    
    # Task selection
    parser.add_argument("--task-id", type=str, help="Task ID to process")
    parser.add_argument("--task-file", type=str, help="Path to JSON file with task IDs")
    
    # Data files
    parser.add_argument("--data-file", type=str, 
                        default="../arc-data-cleaned/arc-agi_evaluation_challenges.json",
                        help="Path to JSON file with task data")
    parser.add_argument("--solutions-file", type=str,
                        help="Path to JSON file with solutions data")
    
    # Generation parameters
    parser.add_argument("--k", type=int, default=8,
                        help="Number of programs to generate per task")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Number of concurrent API calls")
    
    # Output options
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize results")
    parser.add_argument("--save-results", type=str,
                        help="Path to save results JSON")
    parser.add_argument("--save-viz", type=str,
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    try:
        if args.task_id:
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
                concurrency=args.concurrency,
                data_file=args.data_file,
                solutions_file=args.solutions_file,
                save_results=args.save_results,
                visualize=args.visualize,
                save_viz=args.save_viz
            )
        else:
            print("Error: Either --task-id or --task-file must be specified")
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
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
