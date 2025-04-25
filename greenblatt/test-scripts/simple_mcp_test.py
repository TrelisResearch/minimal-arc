#!/usr/bin/env python3
"""
Simple MCP Sandbox Test

This script tests running code in the MCP sandbox using the stdio transport.
"""
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import json

# Define the server parameters
server_params = StdioServerParameters(
    command='deno',
    args=[
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'jsr:@pydantic/mcp-run-python',
        'stdio',
    ],
)

# Test grid transformation function
code = """
# /// script
# dependencies = []
# ///
import json

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

# Test input and expected output
test_input = [[1, 2], [3, 4]]
expected_output = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 2, 1, 2],
    [3, 4, 3, 4]
]

# Run the test
result = solve(test_input)

# Print the results
print("Input grid:")
print(json.dumps(test_input))
print("\\nOutput grid:")
print(json.dumps(result))
print("\\nTest passed:", result == expected_output)
"""

async def main():
    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available MCP tools: {[tool.name for tool in tools.tools]}")
            
            # Run the code in the sandbox
            print("\nRunning code in sandbox...")
            result = await session.call_tool('run_python_code', {'python_code': code})
            
            # Print the result
            print("\nSandbox output:")
            print(result.content[0].text)
            
            print("\nTest completed successfully!")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
