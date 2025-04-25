"""runner.py – simplified wrapper around MCP:run-python

Uses the MCP client with stdio transport to connect to a running Deno MCP server.
Provides run_in_sandbox(code: str, inputs: list[list]) -> list[list]
"""
import asyncio
import json
from typing import List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Server parameters for the MCP Run Python server
SERVER_PARAMS = StdioServerParameters(
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

async def run_in_sandbox(code: str, inputs: List[List[List[int]]], timeout: float = 2.0) -> List[Optional[List[List[int]]]]:
    """
    Run the provided code in a sandbox and return the results.
    
    Args:
        code: The Python code string containing a solve function
        inputs: A list of input grids to test
        timeout: Maximum execution time in seconds
        
    Returns:
        A list of output grids or None for each input if execution failed
    """
    # Prepare the full code with imports and wrapper
    full_code = f"""
# /// script
# dependencies = []
# ///

import json

# LLM-generated code
{code}

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
inputs = {json.dumps(inputs)}
results = run_tests(inputs)
print(json.dumps(results))
"""
    
    try:
        # Connect to the running MCP server
        async with stdio_client(SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                
                # Execute the code in the sandbox with timeout
                try:
                    result = await asyncio.wait_for(
                        session.call_tool('run_python_code', {'python_code': full_code}),
                        timeout=timeout
                    )
                    
                    # Get the output text
                    output_text = result.content[0].text
                    
                    # Parse the output to extract the JSON results
                    try:
                        # Look for JSON array in the output
                        for line in output_text.splitlines():
                            line = line.strip()
                            if line.startswith('[') and line.endswith(']'):
                                return json.loads(line)
                        
                        # If we couldn't find a JSON array, check for output tags
                        if "<o>" in output_text:
                            output_start = output_text.find("<o>") + len("<o>")
                            output_end = output_text.find("</o>")
                            if output_start >= 0 and output_end >= 0:
                                output_content = output_text[output_start:output_end].strip()
                                
                                # Look for JSON array in the output content
                                for line in output_content.splitlines():
                                    line = line.strip()
                                    if line.startswith('[') and line.endswith(']'):
                                        return json.loads(line)
                        
                        # Check for errors
                        if "<e>" in output_text:
                            error_start = output_text.find("<e>") + len("<e>")
                            error_end = output_text.find("</e>")
                            if error_start >= 0 and error_end >= 0:
                                error_content = output_text[error_start:error_end].strip()
                                print(f"Execution error: {error_content}")
                        
                        # If we couldn't extract results, return None for each input
                        print(f"Could not parse output: {output_text}")
                        return [None] * len(inputs)
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON output: {e}")
                        return [None] * len(inputs)
                        
                except asyncio.TimeoutError:
                    print(f"Execution timed out after {timeout} seconds")
                    return [None] * len(inputs)
    
    except Exception as e:
        print(f"MCP execution error: {e}")
        return [None] * len(inputs)

async def cleanup():
    """
    Dummy cleanup function to maintain compatibility with the main script.
    
    With the simplified implementation, there's no persistent state to clean up,
    but we keep this function for API compatibility.
    """
    pass
