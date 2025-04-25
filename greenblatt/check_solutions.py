#!/usr/bin/env python3
"""
Check the structure of the solutions file.
"""
import json
import sys
from pathlib import Path

# Load the solutions data
solutions_file = Path("../arc-data-cleaned/arc-agi_evaluation_solutions.json")
with open(solutions_file, "r") as f:
    solutions_data = json.load(f)

# Get the task ID from command line or use default
task_id = sys.argv[1] if len(sys.argv) > 1 else "00576224"

# Check if the task ID exists in the solutions data
print(f"Task ID: {task_id}")
print(f"Task exists in solutions: {task_id in solutions_data}")

if task_id in solutions_data:
    print(f"\nSolutions data structure for task {task_id}:")
    print(json.dumps(solutions_data[task_id], indent=2))
    
    # Check if test examples have output
    if "test" in solutions_data[task_id]:
        for i, example in enumerate(solutions_data[task_id]["test"]):
            print(f"\nTest example {i+1} has output: {'output' in example}")
            if "output" in example:
                print(f"Output: {example['output']}")
    else:
        print("\nNo test examples found in solutions data")
else:
    print("\nTask not found in solutions data")
