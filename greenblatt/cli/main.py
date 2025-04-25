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
    task_id: str,
    k: int,
    temperature: float,
    concurrency: int,
    visualize: bool,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Process a single task."""
    print(f"\n{'='*50}")
    print(f"Processing task: {task_id}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # Generate programs
    print(f"Generating {k} programs...")
    programs = await generate_programs_for_task(
        task_data=task_data,
        task_id=task_id,
        k=k,
        temperature=temperature,
        concurrency=concurrency
    )
    
    # Evaluate programs
    print(f"Evaluating {len(programs)} programs...")
    result = await evaluate_task(
        task_data=task_data,
        task_id=task_id,
        programs=programs
    )
    
    # Add timing information
    elapsed_time = time.time() - start_time
    result["elapsed_time"] = elapsed_time
    
    # Print results
    print(f"\nResults for task {task_id}:")
    print(f"Total programs: {result['total_programs']}")
    print(f"Valid programs: {result['valid_programs']} ({result['valid_ratio']:.2%})")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    # Visualize if requested
    if visualize:
        print("Visualizing results...")
        # Use majority_output if available, otherwise use first_program_output
        candidate_output = result.get('majority_output') or result.get('first_program_output')
        visualize_task(
            task_data=task_data,
            task_id=task_id,
            candidate_output=candidate_output,
            save_path=str(save_dir / f"{task_id}.png") if save_dir else None
        )
    
    return result

async def process_task_file(
    task_file: str,
    task_ids_file: Optional[str],
    k: int,
    temperature: float,
    concurrency: int,
    visualize: bool,
    save_results: Optional[str],
    save_viz: Optional[str]
) -> Dict[str, Any]:
    """Process all tasks in a file."""
    # Load task data
    task_data = load_task_data(task_file)
    
    # Load task IDs if provided, otherwise use all tasks in the file
    if task_ids_file:
        task_ids = load_task_ids(task_ids_file)
        # Filter to only include tasks that exist in the task data
        task_ids = [task_id for task_id in task_ids if task_id in task_data]
    else:
        task_ids = list(task_data.keys())
    
    print(f"Processing {len(task_ids)} tasks from {task_file}")
    
    # Create save directories if needed
    save_viz_dir = None
    if save_viz:
        save_viz_dir = Path(save_viz)
        save_viz_dir.mkdir(exist_ok=True, parents=True)
    
    # Process each task
    results = {}
    for task_id in task_ids:
        try:
            result = await process_single_task(
                task_data=task_data,
                task_id=task_id,
                k=k,
                temperature=temperature,
                concurrency=concurrency,
                visualize=visualize,
                save_dir=save_viz_dir
            )
            results[task_id] = result
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            results[task_id] = {"error": str(e)}
    
    # Calculate overall statistics
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results.values() if "majority_output" in r and r["majority_output"] is not None)
    
    overall_results = {
        "tasks": results,
        "summary": {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_valid_ratio": sum(r.get("valid_ratio", 0) for r in results.values()) / total_tasks if total_tasks > 0 else 0,
            "total_elapsed_time": sum(r.get("elapsed_time", 0) for r in results.values())
        }
    }
    
    # Save results if requested
    if save_results:
        with open(save_results, 'w') as f:
            json.dump(overall_results, f, indent=2)
        print(f"Saved results to {save_results}")
    
    # Print summary
    print("\nOverall Summary:")
    print(f"Total tasks: {total_tasks}")
    print(f"Successful tasks: {successful_tasks} ({overall_results['summary']['success_rate']:.2%})")
    print(f"Average valid ratio: {overall_results['summary']['average_valid_ratio']:.2%}")
    print(f"Total elapsed time: {overall_results['summary']['total_elapsed_time']:.2f} seconds")
    
    return overall_results

async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description="ARC Greenblatt-style demo")
    
    # Task selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", help="Specific task ID to process")
    group.add_argument("--task-file", help="Path to JSON file with task IDs")
    
    # Task data
    parser.add_argument("--data-file", default="../arc-data-cleaned/arc-agi_evaluation_challenges.json",
                        help="Path to JSON file with task data")
    
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
    parser.add_argument("--save-results", 
                        help="Path to save results JSON")
    parser.add_argument("--save-viz", 
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Check for OpenRouter API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY='your-key'")
        return 1
    
    try:
        if args.task_id:
            # Process a single task
            task_data = load_task_data(args.data_file)
            if args.task_id not in task_data:
                print(f"Error: Task ID {args.task_id} not found in {args.data_file}")
                return 1
            
            save_viz_dir = None
            if args.save_viz:
                save_viz_dir = Path(args.save_viz)
                save_viz_dir.mkdir(exist_ok=True, parents=True)
            
            result = await process_single_task(
                task_data=task_data,
                task_id=args.task_id,
                k=args.k,
                temperature=args.temperature,
                concurrency=args.concurrency,
                visualize=args.visualize,
                save_dir=save_viz_dir
            )
            
            if args.save_results:
                with open(args.save_results, 'w') as f:
                    json.dump({args.task_id: result}, f, indent=2)
                print(f"Saved results to {args.save_results}")
        else:
            # Process multiple tasks
            await process_task_file(
                task_file=args.data_file,
                task_ids_file=args.task_file,
                k=args.k,
                temperature=args.temperature,
                concurrency=args.concurrency,
                visualize=args.visualize,
                save_results=args.save_results,
                save_viz=args.save_viz
            )
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
