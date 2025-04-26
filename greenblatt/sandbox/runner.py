"""runner.py – wrapper around MCP:run-python with local executor option

Uses either:
1. MCP client with stdio transport to connect to a running Deno MCP server (default)
2. Local Python executor with AST-based security for faster execution (with --no-sandbox flag)
"""
import asyncio
import json
import hashlib
import time
from typing import List, Optional, Dict, Any, Tuple
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import local executor
from sandbox.local_executor import run_batch_programs_local, run_in_local_sandbox

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

# Global flag to control whether to use the local executor
USE_LOCAL_EXECUTOR = False

def set_use_local_executor(use_local: bool):
    """Set whether to use the local executor instead of MCP sandbox."""
    global USE_LOCAL_EXECUTOR
    USE_LOCAL_EXECUTOR = use_local
    if use_local:
        print("Using local Python executor with AST-based security")
    else:
        print("Using MCP sandbox for code execution")

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
    # Check if we should use the local executor
    if USE_LOCAL_EXECUTOR:
        # Use the local executor
        return await run_in_local_sandbox(code, inputs)
    
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

async def run_batch_programs(programs: List[str], inputs: List[List[List[int]]]) -> Dict[str, List[Optional[List[List[int]]]]]:
    """
    Run multiple programs in a single sandbox session, but execute each program individually.
    
    Args:
        programs: List of Python code strings containing solve functions
        inputs: A list of input grids to test
        
    Returns:
        Dictionary mapping program indices to their results
    """
    # Check if we should use the local executor
    if USE_LOCAL_EXECUTOR:
        # Use the local executor for batch processing
        return await run_batch_programs_local(programs, inputs)
    
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
                
                # Execute each program individually using the same session
                for i, program in enumerate(programs):
                    # Skip if all results for this program are already cached
                    if all(results[i][j] is not None for j in range(len(inputs))):
                        print(f"Program {i} results all cached, skipping execution")
                        continue
                    
                    print(f"Preparing code for program {i}...")
                    # Prepare the code for this program
                    program_code = f"""
# /// script
# dependencies = []
# ///

{program}

import json
import traceback
import time

# Run tests on the provided inputs
inputs = {json.dumps(inputs)}
results = []

for inp_idx, inp in enumerate(inputs):
    try:
        start_time = time.time()
        result = solve(inp)
        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Log slow executions
            print(f"Input {{inp_idx}} took {{elapsed:.2f}}s")
        results.append(result)
    except Exception as e:
        print(f"Error on input {{inp_idx}}: {{e}}")
        traceback.print_exc()
        results.append(None)

# Output results as JSON
print(json.dumps(results))
"""
                    
                    # Execute the code in the sandbox with timeout
                    try:
                        print(f"Executing program {i} in sandbox with 1.5s timeout at {time.strftime('%H:%M:%S')}...")
                        execution_start = time.time()
                        result = await asyncio.wait_for(
                            session.call_tool('run_python_code', {'python_code': program_code}),
                            timeout=1.5
                        )
                        execution_time = time.time() - execution_start
                        print(f"Program {i} execution completed in {execution_time:.2f}s")
                        
                        # Get the output text
                        output_text = result.content[0].text
                        
                        # Parse the output to extract the JSON results
                        try:
                            # Look for JSON array in the output
                            program_results = None
                            for line in output_text.splitlines():
                                line = line.strip()
                                if line.startswith('[') and line.endswith(']'):
                                    program_results = json.loads(line)
                                    break
                            
                            # If we couldn't find a JSON array, check for output tags
                            if program_results is None and "<o>" in output_text:
                                output_start = output_text.find("<o>") + len("<o>")
                                output_end = output_text.find("</o>")
                                if output_start >= 0 and output_end >= 0:
                                    output_content = output_text[output_start:output_end].strip()
                                    
                                    # Look for JSON array in the output content
                                    for line in output_content.splitlines():
                                        line = line.strip()
                                        if line.startswith('[') and line.endswith(']'):
                                            program_results = json.loads(line)
                                            break
                            
                            # Check for errors
                            if program_results is None and "<e>" in output_text:
                                error_start = output_text.find("<e>") + len("<e>")
                                error_end = output_text.find("</e>")
                                if error_start >= 0 and error_end >= 0:
                                    error_content = output_text[error_start:error_end].strip()
                                    print(f"Execution error for program {i}: {error_content}")
                            
                            # Store and cache results if found
                            if program_results:
                                results[i] = program_results
                                # Cache individual results
                                for input_idx, result in enumerate(program_results):
                                    if result is not None:
                                        RESULT_CACHE[cache_keys[i][input_idx]] = result
                            else:
                                print(f"Could not parse output for program {i}: {output_text[:200]}...")
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON output for program {i}: {e}")
                            # Program results remain as None for this program
                            
                    except asyncio.TimeoutError:
                        execution_time = time.time() - execution_start
                        print(f"Program {i} execution timed out after {execution_time:.2f}s (timeout was 1.5s)")
                        # Results remain as None for this program
                        
                    except Exception as e:
                        print(f"Error executing program {i}: {e}")
                        # Results remain as None for this program
                
                total_time = time.time() - start_time
                print(f"Batch execution completed in {total_time:.2f}s")
                return results
    
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
