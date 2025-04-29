"""
ARC DSL Task Runner.

This script runs the DSL solver on a single ARC task.
"""
import argparse
import time
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Fix the import path issue
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Use correct imports for the new directory structure
from dsl.dsl_utils.primitives import ALL_PRIMITIVES, TILE_PATTERN, print_primitives_summary
from dsl.dsl_utils.types import Grid
from dsl.dsl_utils.program import Program, TimeoutException
from dsl.solver.task_solver import solve_task, evaluate_program
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input


def main():
    """Main entry point for the task runner."""
    parser = argparse.ArgumentParser(description='Run the ARC DSL solver on a single task')
    parser.add_argument('task_id', type=str, help='The task ID to solve')
    parser.add_argument('--depth', type=int, default=4, help='Maximum search depth (default: 4)')
    parser.add_argument('--timeout', type=float, default=15.0, help='Search timeout in seconds (default: 15.0)')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--save', type=str, help='Save visualization to file')
    parser.add_argument('--data-path', type=str, help='Path to the data directory')
    parser.add_argument('--direct-test', action='store_true', help='Directly test the tile_pattern function')
    parser.add_argument('--parallel', action='store_true', default=True, help='Use parallel search (default: True)')
    parser.add_argument('--num-processes', type=int, help='Number of processes to use for parallel search (default: CPU count - 1)')
    parser.add_argument('--op-timeout', type=float, default=0.25, help='Timeout for individual operations in seconds (default: 0.25)')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    
    args = parser.parse_args()
    
    # Display welcome message with primitives summary
    print_primitives_summary()
    print()
    
    # Load the task
    print(f"Loading task {args.task_id}...")
    task = load_task(args.task_id, args.data_path)
    
    # Extract training pairs
    train_pairs = load_train_pairs(task)
    test_input = load_test_input(task)
    
    # Try to load the solution
    solution = None
    try:
        solution_grid = load_solution(args.task_id, args.data_path)
        solution = Grid(solution_grid)
    except Exception as e:
        print(f"Warning: Could not load solution: {e}")
    
    # Set up parallel processing
    if args.num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    else:
        num_processes = args.num_processes
    
    # Direct test for the tile_pattern function
    if args.direct_test and args.task_id == "00576224":
        print("Directly testing tile_pattern function...")
        
        # Create a program with just the tile_pattern operation
        program = Program([TILE_PATTERN])
        
        # Check if it works for all training examples
        all_correct = True
        for inp, expected in train_pairs:
            try:
                result = program.run(inp, op_timeout=args.op_timeout)
                if result != expected:
                    all_correct = False
                    print(f"Failed on training example: {inp.data.tolist()}")
                    print(f"Expected: {expected.data.tolist()}")
                    print(f"Got: {result.data.tolist()}")
            except TimeoutException:
                all_correct = False
                print(f"Operation timed out on training example: {inp.data.tolist()}")
            except Exception as e:
                all_correct = False
                print(f"Error on training example: {e}")
        
        if all_correct:
            print("tile_pattern function works for all training examples!")
            
            # Generate prediction for the test input
            try:
                prediction = program.run(test_input, op_timeout=args.op_timeout)
            except TimeoutException:
                print("Operation timed out on test input")
                return
            except Exception as e:
                print(f"Error on test input: {e}")
                return
            
            # Visualize the results
            if args.show or args.save:
                # Skip visualization if there's no prediction
                if prediction is None:
                    print("No prediction to visualize")
                    return
                    
                # Make sure we have the correct data format for visualization
                if isinstance(prediction, Grid):
                    prediction_data = prediction.data
                else:
                    prediction_data = prediction
                
                # Skip solution visualization if it's not available or causing errors
                solution_data = None
                
                try:
                    # Create a simplified visualization without the solution
                    n_train = len(task['train'])
                    n_rows = n_train + 1  # Training examples + test
                    n_cols = 2  # Input, Prediction
                    
                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
                    
                    # If there's only one row, wrap the axes in a list
                    if n_rows == 1:
                        axs = [axs]
                    
                    # Visualize training examples
                    for i, example in enumerate(task['train']):
                        # Display the input grid
                        input_grid = np.array(example['input'])
                        axs[i][0].imshow(input_grid, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                        axs[i][0].set_title(f"Train {i+1} Input")
                        axs[i][0].axis('off')
                        
                        # Display the output grid
                        output_grid = np.array(example['output'])
                        axs[i][1].imshow(output_grid, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                        axs[i][1].set_title(f"Train {i+1} Output")
                        axs[i][1].axis('off')
                        
                        # Add grid lines
                        for ax in axs[i]:
                            ax.set_xticks(np.arange(-0.5, max(input_grid.shape[1], output_grid.shape[1]), 1), minor=True)
                            ax.set_yticks(np.arange(-0.5, max(input_grid.shape[0], output_grid.shape[0]), 1), minor=True)
                            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    
                    # Visualize test example
                    test_input = np.array(task['test'][0]['input'])
                    axs[-1][0].imshow(test_input, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                    axs[-1][0].set_title("Test Input")
                    axs[-1][0].axis('off')
                    
                    # Display the prediction
                    axs[-1][1].imshow(prediction_data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                    axs[-1][1].set_title("Test Prediction")
                    axs[-1][1].axis('off')
                    
                    # Add grid lines for test
                    for ax in axs[-1]:
                        ax.set_xticks(np.arange(-0.5, max(test_input.shape[1], prediction_data.shape[1]), 1), minor=True)
                        ax.set_yticks(np.arange(-0.5, max(test_input.shape[0], prediction_data.shape[0]), 1), minor=True)
                        ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                    
                    fig.tight_layout()
                    
                    if args.save:
                        fig.savefig(args.save, bbox_inches='tight')
                        print(f"Visualization saved to {args.save}")
                    
                    if args.show:
                        plt.show()
                        
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    
            print("Solution found for task 00576224!")
            print(f"Program: {program}")
            return
    
    # Solve the task using the unified solver
    result = solve_task(
        task_id=args.task_id,
        train_pairs=train_pairs,
        test_input=test_input,
        depth=args.depth,
        timeout=args.timeout,
        op_timeout=args.op_timeout,
        parallel=args.parallel,
        num_processes=num_processes,
        debug=args.debug
    )
    
    found_solution = result['solved']
    valid_program = result['program']
    prediction = result['prediction']
    elapsed_time = result['elapsed_time']
    search_exhausted = result.get('search_exhausted', False)
    
    if not found_solution:
        if search_exhausted:
            print("No solution found: Search space exhausted (all programs up to depth tried)")
        else:
            print(f"No solution found: Search timed out after {args.timeout} seconds")
        return
    
    # Visualize the results
    if args.show or args.save:
        # Skip visualization if there's no prediction
        if prediction is None:
            print("No prediction to visualize")
            return
            
        # Make sure we have the correct data format for visualization
        if isinstance(prediction, Grid):
            prediction_data = prediction.data
        else:
            prediction_data = prediction
        
        # Skip solution visualization if it's not available or causing errors
        solution_data = None
        if solution is not None:
            solution_data = solution.data
        
        try:
            # Create a visualization
            n_train = len(task['train'])
            n_rows = n_train + 1  # Training examples + test
            n_cols = 3 if solution_data is not None else 2  # Input, Prediction, (Solution)
            
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            
            # If there's only one row, wrap the axes in a list
            if n_rows == 1:
                axs = [axs]
            
            # Visualize training examples
            for i, example in enumerate(task['train']):
                # Display the input grid
                input_grid = np.array(example['input'])
                axs[i][0].imshow(input_grid, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                axs[i][0].set_title(f"Train {i+1} Input")
                axs[i][0].axis('off')
                
                # Display the output grid
                output_grid = np.array(example['output'])
                axs[i][1].imshow(output_grid, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                axs[i][1].set_title(f"Train {i+1} Output")
                axs[i][1].axis('off')
                
                # Add grid lines
                for ax in axs[i][:2]:
                    ax.set_xticks(np.arange(-0.5, max(input_grid.shape[1], output_grid.shape[1]), 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, max(input_grid.shape[0], output_grid.shape[0]), 1), minor=True)
                    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
                
                # Add empty plot for solution column if needed
                if solution_data is not None:
                    axs[i][2].axis('off')
            
            # Visualize test example
            test_input = np.array(task['test'][0]['input'])
            axs[-1][0].imshow(test_input, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
            axs[-1][0].set_title("Test Input")
            axs[-1][0].axis('off')
            
            # Display the prediction
            axs[-1][1].imshow(prediction_data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
            axs[-1][1].set_title("Test Prediction")
            axs[-1][1].axis('off')
            
            # Display the solution if available
            if solution_data is not None:
                # If solution has an extra dimension (e.g., (1, 4, 4)), remove it
                if solution_data.ndim > 2 and solution_data.shape[0] == 1:
                    solution_data = solution_data[0]
                
                axs[-1][2].imshow(solution_data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                axs[-1][2].set_title("Test Solution")
                axs[-1][2].axis('off')
                
                # Check if the prediction matches the solution
                if np.array_equal(prediction_data, solution_data):
                    print("Prediction matches solution!")
                else:
                    print("Prediction does not match solution")
                    print(f"Prediction shape: {prediction_data.shape}, Solution shape: {solution_data.shape}")
            
            # Add grid lines for test
            for ax in axs[-1][:2]:
                ax.set_xticks(np.arange(-0.5, max(test_input.shape[1], prediction_data.shape[1]), 1), minor=True)
                ax.set_yticks(np.arange(-0.5, max(test_input.shape[0], prediction_data.shape[0]), 1), minor=True)
                ax.grid(which='minor', color='w', linestyle='-', linewidth=1)
            
            if solution_data is not None:
                axs[-1][2].set_xticks(np.arange(-0.5, solution_data.shape[1], 1), minor=True)
                axs[-1][2].set_yticks(np.arange(-0.5, solution_data.shape[0], 1), minor=True)
                axs[-1][2].grid(which='minor', color='w', linestyle='-', linewidth=1)
            
            fig.tight_layout()
            
            if args.save:
                fig.savefig(args.save, bbox_inches='tight')
                print(f"Visualization saved to {args.save}")
            
            if args.show:
                plt.show()
                
        except Exception as e:
            print(f"Error during visualization: {e}")


if __name__ == '__main__':
    main()
