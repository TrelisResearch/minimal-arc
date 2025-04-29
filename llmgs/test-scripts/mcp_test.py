#!/usr/bin/env python3
"""
MCP Sandbox Test Script

This script tests running code in the MCP sandbox using the running MCP server.
It connects to the MCP server via stdio and executes a simple grid transformation.
"""
import asyncio
import json
import sys
import subprocess
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import MCP client libraries
from mcp import ClientSession
from mcp.client.stdio import stdio_client

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
TEST_INPUT = [
    [1, 2],
    [3, 4]
]

# Expected output
EXPECTED_OUTPUT = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 2, 1, 2],
    [3, 4, 3, 4]
]

async def run_in_mcp_sandbox():
    """Run code in the MCP sandbox using the stdio client."""
    print("Connecting to the running MCP server...")
    
    # Create a client using the stdio transport
    async with stdio_client() as (read, write):
        # Create a session with the client
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available MCP tools: {[tool.name for tool in tools.tools]}")
            
            # Prepare the full code with imports and wrapper
            full_code = f"""
# /// script
# dependencies = []
# ///

# Only use standard library modules
import json

# LLM-generated code
{TEST_CODE}

# Test function
def run_tests(inputs):
    results = []
    for inp in inputs:
        try:
            result = solve(inp)
            results.append(result)
        except Exception as e:
            print(f"Error: {{e}}")
            results.append(None)
    return results

# Run tests
inputs = {json.dumps([TEST_INPUT])}
results = run_tests(inputs)
print(json.dumps(results))
"""
            
            # Execute the code in the sandbox
            result = await session.call_tool('run_python_code', {'python_code': full_code})
            
            # Print the raw output for debugging
            output_text = result.content[0].text
            print(f"\nRaw output from MCP sandbox:\n{output_text}")
            
            # Parse the output to extract the JSON results
            try:
                # Look for JSON array in the output
                for line in output_text.splitlines():
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        results = json.loads(line)
                        
                        # Check if we got a result
                        if results is None or len(results) == 0 or results[0] is None:
                            print("Error: No result returned from MCP sandbox")
                            return False
                        
                        # Get the first result
                        result = results[0]
                        
                        # Print the result
                        print("\nInput grid:")
                        for row in TEST_INPUT:
                            print(row)
                        
                        print("\nOutput grid:")
                        for row in result:
                            print(row)
                        
                        # Verify the result
                        if result == EXPECTED_OUTPUT:
                            print("\n✅ Test passed! The MCP sandbox is working correctly.")
                            return True
                        else:
                            print("\n❌ Test failed! The output does not match the expected output.")
                            print(f"Expected: {EXPECTED_OUTPUT}")
                            print(f"Got: {result}")
                            return False
                
                print("Error: Could not find JSON output in the response")
                return False
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON output: {e}")
                return False
            except Exception as e:
                print(f"Error processing result: {e}")
                return False

async def main_async():
    """Main async function."""
    print("Testing MCP sandbox...")
    print("Make sure the MCP server is running in a separate terminal with:")
    print("deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python stdio")
    
    try:
        # Run the test
        success = await run_in_mcp_sandbox()
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

def main():
    """Main function."""
    try:
        # Run the async main function
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
