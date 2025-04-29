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

# Fix the import path issue
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Use correct imports for the new directory structure
from dsl.dsl_utils.primitives import ALL_PRIMITIVES, TILE_PATTERN, print_primitives_summary
from dsl.dsl_utils.types import Grid
from dsl.dsl_utils.program import Program, TimeoutException
from dsl.search.enumerator import iter_deepening
from dsl.search.verifier import verify, evaluate_program
from dsl.io.loader import load_task, load_solution, load_train_pairs, load_test_input


def evaluate_program(program, input_grid, op_timeout=0.25):
    """
    Evaluate a program on an input grid with timeout handling.
    
    Args:
        program: The program to evaluate
        input_grid: The input grid
        op_timeout: Timeout for individual operations in seconds
        
    Returns:
        The result of running the program, or None if an error occurs
    """
    try:
        return program.run(input_grid, op_timeout=op_timeout)
    except TimeoutException:
        print("Operation timed out during evaluation")
        return None
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None


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
    parser.add_argument('--num-processes', type=int, help='Number of processes to use for parallel search')
    parser.add_argument('--op-timeout', type=float, default=0.25, help='Timeout for individual operations in seconds (default: 0.25)')
    
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
    
    # Get shapes for heuristics
    input_shape = train_pairs[0][0].shape
    output_shape = train_pairs[0][1].shape
    
    print(f"Input shape: {input_shape}, Output shape: {output_shape}")
    print(f"Searching for programs with max depth {args.depth}...")
    
    # Start the search
    start_time = time.time()
    found_solution = False
    valid_program = None
    prediction = None
    
    # Generate and verify programs
    for program in iter_deepening(ALL_PRIMITIVES, args.depth, input_shape, output_shape, args.timeout, 
                                 args.parallel, args.num_processes):
        try:
            if verify(program, train_pairs, op_timeout=args.op_timeout):
                valid_program = program
                found_solution = True
                print(f"Found valid program: {program}")
                
                # Generate prediction for the test input
                prediction = evaluate_program(program, test_input, op_timeout=args.op_timeout)
                if prediction is not None:
                    print(f"Generated prediction for test input")
                else:
                    print(f"Failed to generate prediction for test input")
                
                break
        except TimeoutException:
            print(f"Program timed out during verification: {program}")
            continue
        except Exception as e:
            print(f"Error during verification: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Search completed in {elapsed_time:.2f} seconds")
    
    if not found_solution:
        print("No solution found")
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
                axs[-1][2].imshow(solution_data, interpolation='nearest', vmin=0, vmax=9, cmap='tab10')
                axs[-1][2].set_title("Test Solution")
                axs[-1][2].axis('off')
                
                # Check if the prediction matches the solution
                if np.array_equal(prediction_data, solution_data):
                    print("Prediction matches solution!")
                else:
                    print("Prediction does not match solution")
            
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
