"""runner.py – simplified wrapper around MCP:run-python

Uses the MCP client with stdio transport to connect to a running Deno MCP server.
Provides run_in_sandbox(code: str, inputs: list[list]) -> list[list]
"""
import asyncio
import json
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
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

# Global cache for storing results
RESULT_CACHE: Dict[str, List[Optional[List[List[int]]]]] = {}

# Global default timeout
DEFAULT_TIMEOUT = 3.0

def set_default_timeout(timeout: float):
    """Set the default timeout for sandbox execution."""
    global DEFAULT_TIMEOUT
    print(f"Setting default sandbox timeout to {timeout}s")
    DEFAULT_TIMEOUT = timeout

def hash_input(input_grid: List[List[int]]) -> str:
    """Create a hash of an input grid for caching."""
    return hashlib.md5(json.dumps(input_grid).encode()).hexdigest()

def hash_program(code: str) -> str:
    """Create a hash of a program for caching."""
    # Normalize whitespace to avoid trivial differences
    normalized = "\n".join(line.strip() for line in code.strip().split("\n") if line.strip())
    return hashlib.md5(normalized.encode()).hexdigest()

def get_cache_key(code: str, input_grid: List[List[int]]) -> str:
    """Create a cache key from program and input hashes."""
    program_hash = hash_program(code)
    input_hash = hash_input(input_grid)
    return f"{program_hash}_{input_hash}"

async def run_in_sandbox(code: str, inputs: List[List[List[int]]], timeout: float = 3.0) -> List[Optional[List[List[int]]]]:
    """
    Run the provided code in a sandbox and return the results.
    
    Args:
        code: The Python code string containing a solve function
        inputs: A list of input grids to test
        timeout: Maximum execution time in seconds
        
    Returns:
        A list of output grids or None for each input if execution failed
    """
    # Check cache for all inputs
    all_cached = True
    results = [None] * len(inputs)
    cache_keys = []
    
    for i, input_grid in enumerate(inputs):
        cache_key = get_cache_key(code, input_grid)
        cache_keys.append(cache_key)
        
        if cache_key in RESULT_CACHE:
            results[i] = RESULT_CACHE[cache_key]
        else:
            all_cached = False
    
    # If all results are cached, return them
    if all_cached:
        return results
    
    # Prepare the full code with minimal wrapper
    # The code should already contain a solve function and any necessary imports
    full_code = f"""
# /// script
# dependencies = []
# ///

{code}

# Run tests on the provided inputs
import json
inputs = {json.dumps(inputs)}
results = []
for inp in inputs:
    try:
        result = solve(inp)
        results.append(result)
    except Exception as e:
        print(f"Error: {{e}}")
        results.append(None)

# Output results as JSON
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
                                parsed_results = json.loads(line)
                                
                                # Cache individual results
                                for i, (result, input_grid) in enumerate(zip(parsed_results, inputs)):
                                    if result is not None:
                                        RESULT_CACHE[cache_keys[i]] = result
                                
                                return parsed_results
                        
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
                                        parsed_results = json.loads(line)
                                        
                                        # Cache individual results
                                        for i, (result, input_grid) in enumerate(zip(parsed_results, inputs)):
                                            if result is not None:
                                                RESULT_CACHE[cache_keys[i]] = result
                                        
                                        return parsed_results
                        
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

async def run_batch_programs(programs: List[str], inputs: List[List[List[int]]], timeout: float = None) -> Dict[str, List[Optional[List[List[int]]]]]:
    """
    Run multiple programs in a single sandbox call.
    
    Args:
        programs: List of Python code strings containing solve functions
        inputs: A list of input grids to test
        timeout: Maximum execution time in seconds (default is None, which uses DEFAULT_TIMEOUT)
        
    Returns:
        Dictionary mapping program indices to their results
    """
    # Use default timeout if none provided
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    
    # Calculate dynamic timeout based on number of programs
    # Use min(timeout, 1.5 * number_of_programs) to prevent excessive timeouts
    dynamic_timeout = min(timeout, 1.5 * len(programs)) if len(programs) > 0 else timeout
    if dynamic_timeout != timeout:
        print(f"Adjusting timeout from {timeout:.1f}s to {dynamic_timeout:.1f}s based on {len(programs)} programs")
        timeout = dynamic_timeout
    
    print(f"Starting batch execution of {len(programs)} programs on {len(inputs)} inputs at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    if not programs:
        return {}
    
    # Check cache for all program-input combinations
    all_cached = True
    results = {i: [None] * len(inputs) for i in range(len(programs))}
    cache_keys = {}
    
    print("Checking cache for program-input combinations...")
    cache_start_time = time.time()
    for i, program in enumerate(programs):
        cache_keys[i] = []
        for j, input_grid in enumerate(inputs):
            cache_key = get_cache_key(program, input_grid)
            cache_keys[i].append(cache_key)
            
            if cache_key in RESULT_CACHE:
                results[i][j] = RESULT_CACHE[cache_key]
            else:
                all_cached = False
    
    cache_time = time.time() - cache_start_time
    print(f"Cache check completed in {cache_time:.2f}s")
    
    # If all results are cached, return them
    if all_cached:
        print("All results found in cache, skipping sandbox execution")
        return results
    
    print("Preparing batch code...")
    code_start_time = time.time()
    # Prepare the full code with all programs combined
    full_code = """
# /// script
# dependencies = []
# ///

import json
import traceback
import time

"""
    
    # Add each program with a unique solve function name
    for i, program in enumerate(programs):
        # Rename the solve function to avoid conflicts
        renamed_program = program.replace("def solve(", f"def solve_{i}(")
        full_code += f"\n# Program {i}\n{renamed_program}\n"
    
    # Add code to run all programs on all inputs
    full_code += f"""
# Run all programs on all inputs
inputs = {json.dumps(inputs)}
results = {{}}

for prog_idx in range({len(programs)}):
    solve_func = globals().get(f"solve_{{prog_idx}}")
    if solve_func:
        prog_results = []
        for inp_idx, inp in enumerate(inputs):
            try:
                start_time = time.time()
                result = solve_func(inp)
                elapsed = time.time() - start_time
                if elapsed > 0.1:  # Log slow executions
                    print(f"Program {{prog_idx}} on input {{inp_idx}} took {{elapsed:.2f}}s")
                prog_results.append(result)
            except Exception as e:
                print(f"Error in program {{prog_idx}} on input {{inp_idx}}: {{e}}")
                traceback.print_exc()
                prog_results.append(None)
        results[prog_idx] = prog_results

# Output results as JSON
print(json.dumps(results))
"""
    
    code_time = time.time() - code_start_time
    print(f"Code preparation completed in {code_time:.2f}s")
    
    print(f"Connecting to MCP server at {time.strftime('%H:%M:%S')}...")
    connect_start_time = time.time()
    try:
        # Connect to the running MCP server
        async with stdio_client(SERVER_PARAMS) as (read, write):
            connect_time = time.time() - connect_start_time
            print(f"Connected to MCP server in {connect_time:.2f}s")
            
            async with ClientSession(read, write) as session:
                # Initialize the session
                print(f"Initializing MCP session at {time.strftime('%H:%M:%S')}...")
                init_start_time = time.time()
                await session.initialize()
                init_time = time.time() - init_start_time
                print(f"Session initialized in {init_time:.2f}s")
                
                # Execute the code in the sandbox with timeout
                try:
                    print(f"Executing batch code in sandbox with {timeout}s timeout at {time.strftime('%H:%M:%S')}...")
                    execution_start = time.time()
                    result = await asyncio.wait_for(
                        session.call_tool('run_python_code', {'python_code': full_code}),
                        timeout=timeout
                    )
                    execution_time = time.time() - execution_start
                    print(f"Sandbox execution completed in {execution_time:.2f}s")
                    
                    # Get the output text
                    output_text = result.content[0].text
                    
                    # Parse the output to extract the JSON results
                    try:
                        print(f"Parsing sandbox output at {time.strftime('%H:%M:%S')}...")
                        parse_start_time = time.time()
                        # Look for JSON dictionary in the output
                        for line in output_text.splitlines():
                            line = line.strip()
                            if line.startswith('{') and line.endswith('}'):
                                parsed_results = json.loads(line)
                                
                                # Convert string keys to integers
                                parsed_results = {int(k): v for k, v in parsed_results.items()}
                                
                                # Cache individual results
                                print(f"Caching results for {len(parsed_results)} programs...")
                                for prog_idx, prog_results in parsed_results.items():
                                    for input_idx, result in enumerate(prog_results):
                                        if result is not None:
                                            RESULT_CACHE[cache_keys[prog_idx][input_idx]] = result
                                
                                parse_time = time.time() - parse_start_time
                                print(f"Output parsing completed in {parse_time:.2f}s")
                                
                                total_time = time.time() - start_time
                                print(f"Batch execution completed in {total_time:.2f}s")
                                return parsed_results
                        
                        # If we couldn't find a JSON dictionary, check for output tags
                        if "<o>" in output_text:
                            output_start = output_text.find("<o>") + len("<o>")
                            output_end = output_text.find("</o>")
                            if output_start >= 0 and output_end >= 0:
                                output_content = output_text[output_start:output_end].strip()
                                
                                # Look for JSON dictionary in the output content
                                for line in output_content.splitlines():
                                    line = line.strip()
                                    if line.startswith('{') and line.endswith('}'):
                                        parsed_results = json.loads(line)
                                        
                                        # Convert string keys to integers
                                        parsed_results = {int(k): v for k, v in parsed_results.items()}
                                        
                                        # Cache individual results
                                        print(f"Caching results for {len(parsed_results)} programs...")
                                        for prog_idx, prog_results in parsed_results.items():
                                            for input_idx, result in enumerate(prog_results):
                                                if result is not None:
                                                    RESULT_CACHE[cache_keys[prog_idx][input_idx]] = result
                                        
                                        parse_time = time.time() - parse_start_time
                                        print(f"Output parsing completed in {parse_time:.2f}s")
                                        
                                        total_time = time.time() - start_time
                                        print(f"Batch execution completed in {total_time:.2f}s")
                                        return parsed_results
                        
                        # Check for errors
                        if "<e>" in output_text:
                            error_start = output_text.find("<e>") + len("<e>")
                            error_end = output_text.find("</e>")
                            if error_start >= 0 and error_end >= 0:
                                error_content = output_text[error_start:error_end].strip()
                                print(f"Execution error: {error_content}")
                        
                        # If we couldn't extract results, return empty results
                        print(f"Could not parse output: {output_text[:200]}...")
                        return {i: [None] * len(inputs) for i in range(len(programs))}
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON output: {e}")
                        return {i: [None] * len(inputs) for i in range(len(programs))}
                        
                except asyncio.TimeoutError:
                    execution_time = time.time() - execution_start
                    print(f"Execution timed out after {execution_time:.2f}s (timeout was {timeout}s)")
                    return {i: [None] * len(inputs) for i in range(len(programs))}
    
    except Exception as e:
        connect_time = time.time() - connect_start_time
        print(f"MCP execution error after {connect_time:.2f}s: {e}")
        return {i: [None] * len(inputs) for i in range(len(programs))}

async def cleanup():
    """
    Dummy cleanup function to maintain compatibility with the main script.
    
    With the simplified implementation, there's no persistent state to clean up,
    but we keep this function for API compatibility.
    """
    pass
