#!/usr/bin/env python3
"""
Run K Analysis

This script runs a dataset multiple times with different values of k and plots
the number of correctly solved tasks versus k using the majority voting approach.
"""
import os
import json
import asyncio
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv()
    print("Loaded environment variables from .env file")
    if os.environ.get("OPENROUTER_API_KEY"):
        print(f"Using OpenRouter API key: {os.environ.get('OPENROUTER_API_KEY')[:2]}{'*' * 69}{os.environ.get('OPENROUTER_API_KEY')[-4:]}")
    else:
        print("WARNING: OPENROUTER_API_KEY is not set")
else:
    print("WARNING: .env file not found")

# Import from the existing codebase
from cli.main import process_task_file


async def run_with_different_k(
    task_file: str,
    data_file: str,
    solutions_file: str,
    k_values: List[int],
    concurrency: int = 32,
    temperature: float = 1.0,
    output_dir: str = "results"
) -> Dict[int, Dict[str, Any]]:
    """
    Run the dataset with different values of k.
    
    Args:
        task_file: Path to the task file
        data_file: Path to the data file
        solutions_file: Path to the solutions file
        k_values: List of k values to try
        concurrency: Number of concurrent API calls
        temperature: Temperature for generation
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping k values to results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run for each k value
    results = {}
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Running with k = {k}")
        print(f"{'='*50}")
        
        # Create results directory for this k value
        k_dir = output_path / f"k_{k}"
        k_dir.mkdir(exist_ok=True)
        
        # Run the task file
        result = await process_task_file(
            task_file=task_file,
            task_ids_file=task_file,
            k=k,
            temperature=temperature,
            concurrency=concurrency,
            data_file=data_file,
            solutions_file=solutions_file,
            save_results=str(k_dir / "results.json"),
            visualize=False
        )
        
        # Store the results
        results[k] = result
        
        # Save the combined results so far
        with open(output_path / "combined_results.json", "w") as f:
            json.dump({str(k): results[k] for k in results}, f, indent=2)
    
    return results


def analyze_results(results: Dict[int, Dict[str, Any]]) -> Tuple[List[int], List[int], List[float]]:
    """
    Analyze the results to extract metrics for plotting.
    
    Args:
        results: Dictionary mapping k values to results
        
    Returns:
        Tuple of (k_values, correct_tasks, correct_ratio)
    """
    k_values = sorted(results.keys())
    correct_tasks = []
    correct_ratio = []
    
    for k in k_values:
        # Count correct tasks
        correct = sum(1 for task_id, task_result in results[k].items() 
                      if task_result.get("test_correct", False))
        total = len(results[k])
        
        correct_tasks.append(correct)
        correct_ratio.append(correct / total if total > 0 else 0)
    
    return k_values, correct_tasks, correct_ratio


def plot_results(
    k_values: List[int],
    correct_tasks: List[int],
    correct_ratio: List[float],
    output_path: str = "results/k_analysis.png"
):
    """
    Plot the results.
    
    Args:
        k_values: List of k values
        correct_tasks: List of correct task counts
        correct_ratio: List of correct task ratios
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot correct tasks vs k
    ax1.plot(k_values, correct_tasks, 'o-', linewidth=2)
    ax1.set_xlabel('k (Number of Programs Generated)')
    ax1.set_ylabel('Number of Correct Tasks')
    ax1.set_title('Correct Tasks vs k')
    ax1.grid(True)
    
    # Plot correct ratio vs k
    ax2.plot(k_values, [r * 100 for r in correct_ratio], 'o-', linewidth=2)
    ax2.set_xlabel('k (Number of Programs Generated)')
    ax2.set_ylabel('Correct Tasks (%)')
    ax2.set_title('Correct Task Percentage vs k')
    ax2.grid(True)
    
    # Set y-axis to start from 0
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0, top=100)
    
    # Add k values as x-ticks
    ax1.set_xticks(k_values)
    ax2.set_xticks(k_values)
    
    # Add a note about majority voting
    fig.text(0.5, 0.01, 
             'Note: Results use majority voting among valid programs, which is more strict than standard pass@k',
             ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run K Analysis")
    
    # Input files
    parser.add_argument("--task-file", type=str, required=True,
                        help="Path to JSON file with task IDs")
    parser.add_argument("--data-file", type=str, 
                        default="../arc-data-cleaned/arc-agi_evaluation_challenges.json",
                        help="Path to JSON file with task data")
    parser.add_argument("--solutions-file", type=str,
                        default="../arc-data-cleaned/arc-agi_evaluation_solutions.json",
                        help="Path to JSON file with solutions data")
    
    # K values
    parser.add_argument("--k-values", type=str, default="2,8,32",
                        help="Comma-separated list of k values to try")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for generation")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Number of concurrent API calls")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="results/k_analysis",
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Run with different k values
        results = await run_with_different_k(
            task_file=args.task_file,
            data_file=args.data_file,
            solutions_file=args.solutions_file,
            k_values=k_values,
            concurrency=args.concurrency,
            temperature=args.temperature,
            output_dir=args.output_dir
        )
        
        # Analyze and plot results
        k_values, correct_tasks, correct_ratio = analyze_results(results)
        plot_results(
            k_values=k_values,
            correct_tasks=correct_tasks,
            correct_ratio=correct_ratio,
            output_path=os.path.join(args.output_dir, "k_analysis.png")
        )
        
        # Print summary
        print("\nSummary:")
        for k, correct, ratio in zip(k_values, correct_tasks, correct_ratio):
            print(f"k = {k}: {correct} correct tasks ({ratio:.2%})")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Clean up resources
        from sandbox.runner import cleanup
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
