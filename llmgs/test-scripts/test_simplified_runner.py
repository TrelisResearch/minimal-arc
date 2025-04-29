#!/usr/bin/env python3
"""
Test Simplified Sandbox Runner

This script tests the simplified sandbox runner with a basic grid transformation.
"""
import asyncio
import json
import sys
import os

# Add the parent directory to the path so we can import the sandbox runner
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sandbox.runner import run_in_sandbox

# Test grid transformation function
TEST_CODE = """
def solve(grid):
    # This function repeats the input grid in a 2x2 pattern
    rows = len(grid)
    cols = len(grid[0])
    
    # Create a new grid with twice the dimensions
    result = []
    for i in range(rows * 2):
        row = []
        for j in range(cols * 2):
            # Get the corresponding cell from the original grid
            orig_i = i % rows
            orig_j = j % cols
            row.append(grid[orig_i][orig_j])
        result.append(row)
    
    return result
"""

# Test input grid
TEST_INPUT = [[1, 2], [3, 4]]

# Expected output grid
EXPECTED_OUTPUT = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 2, 1, 2],
    [3, 4, 3, 4]
]

async def main():
    print("Testing simplified sandbox runner...")
    print("Make sure the MCP server is running in a separate terminal with:")
    print("deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python stdio")
    
    # Run the test code in the sandbox
    print("\nRunning code in sandbox...")
    results = await run_in_sandbox(TEST_CODE, [TEST_INPUT], timeout=5.0)
    
    # Check the results
    if results and results[0]:
        print("\nInput grid:")
        print(json.dumps(TEST_INPUT))
        
        print("\nOutput grid:")
        print(json.dumps(results[0]))
        
        print("\nTest passed:", results[0] == EXPECTED_OUTPUT)
    else:
        print("\nTest failed: No valid results returned")

if __name__ == "__main__":
    asyncio.run(main())
