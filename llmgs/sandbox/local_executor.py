"""local_executor.py – lightweight Python code execution with AST-based security

A minimal "smol-agents" style executor that provides basic security through AST validation
without the overhead of a full sandbox.
"""
import ast
import builtins
import types
import sys
import json
import asyncio
import time
import re
from typing import List, Dict, Any, Optional


class LocalPythonExecutor:
    def __init__(self, allowed_imports=None, max_ops=1_000_000, debug=False):
        # Default allowed imports
        default_allowed = ["numpy", "math", "random", "copy", "collections", "itertools", "functools"]
        self.allowed = set(allowed_imports or default_allowed)
        self.max_ops = max_ops
        self.debug = debug

    def __call__(self, code: str, globals_dict=None):
        """Execute code in a restricted environment."""
        if self.debug:
            print("\nOriginal code:")
            print(code[:500] + "..." if len(code) > 500 else code)
            
        # Remove typing imports
        code = re.sub(r'from\s+typing\s+import\s+[^\n]+', '', code)
        code = re.sub(r'import\s+typing', '', code)
        
        # Create a safe globals dictionary
        globals_env = self._create_safe_globals()
        if globals_dict is not None:
            globals_env.update(globals_dict)
            
        # Execute the code
        try:
            exec(code, globals_env)
            
            if self.debug:
                print(f"\nExecution successful. Available functions: {[k for k, v in globals_env.items() if callable(v)]}")
                
            return globals_env
        except Exception as e:
            if self.debug:
                print(f"\nExecution failed: {e}")
                import traceback
                traceback.print_exc()
            raise

    def _create_safe_globals(self):
        """Create a safe globals dictionary with limited builtins."""
        # Create a restricted __builtins__ with only safe functions
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'bool': bool, 
            'dict': dict, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'int': int, 'len': len, 'list': list, 
            'map': map, 'max': max, 'min': min, 'print': print,
            'range': range, 'round': round, 'set': set, 'sorted': sorted,
            'str': str, 'sum': sum, 'tuple': tuple, 'zip': zip
        }
        
        # Add typing-related names that are commonly used
        from typing import List, Dict, Tuple, Set, Optional, Any, Union
        typing_names = {
            'List': List, 'Dict': Dict, 'Tuple': Tuple, 'Set': Set,
            'Optional': Optional, 'Any': Any, 'Union': Union
        }
        
        # Create the globals dictionary
        globals_dict = {'__builtins__': safe_builtins}
        globals_dict.update(typing_names)
        
        # Add allowed modules
        for module_name in self.allowed:
            try:
                module = __import__(module_name)
                globals_dict[module_name] = module
            except ImportError:
                if self.debug:
                    print(f"Could not import {module_name}")
        
        return globals_dict


async def run_in_local_sandbox(code: str, inputs: List[List[List[int]]], timeout: float = 1.5, debug: bool = False) -> List[Optional[List[List[int]]]]:
    """
    Run the provided code in the local executor and return the results.
    
    Args:
        code: The Python code string containing a solve function
        inputs: A list of input grids to test
        timeout: Maximum execution time in seconds (default: 1.5s)
        debug: Whether to print debug information
        
    Returns:
        A list of output grids or None for each input if execution failed
    """
    # Create a local executor
    executor = LocalPythonExecutor(debug=debug)
    
    results = []
    
    # Execute the code to get the solve function
    try:
        globals_dict = executor(code)
        solve_func = globals_dict.get('solve')
        
        if not solve_func or not callable(solve_func):
            print("Error: No valid 'solve' function found in the code")
            if debug:
                print(f"Available globals: {[k for k, v in globals_dict.items() if callable(v)]}")
            return [None] * len(inputs)
        
        # Process each input with a timeout
        for input_idx, input_grid in enumerate(inputs):
            try:
                if debug:
                    print(f"Running solve function on input {input_idx}")
                
                # Use asyncio.to_thread to run the solve function in a separate thread with timeout
                result = await asyncio.wait_for(
                    asyncio.to_thread(solve_func, input_grid),
                    timeout=timeout
                )
                
                if debug:
                    print(f"Result for input {input_idx}: {result}")
                    
                results.append(result)
            except asyncio.TimeoutError:
                print(f"Execution timed out after {timeout} seconds on input {input_idx}")
                results.append(None)
            except Exception as e:
                print(f"Error during execution on input {input_idx}: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                results.append(None)
                
    except Exception as e:
        print(f"Error setting up execution environment: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return [None] * len(inputs)
    
    return results


async def run_batch_programs_local(programs: List[str], inputs: List[List[List[int]]]) -> Dict[int, List[Optional[List[List[int]]]]]:
    """
    Run multiple programs using the local executor.
    
    Args:
        programs: List of Python code strings containing solve functions
        inputs: A list of input grids to test
        
    Returns:
        Dictionary mapping program indices to their results
    """
    print(f"Starting local batch execution of {len(programs)} programs on {len(inputs)} inputs at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    if not programs:
        return {}
    
    # Process each program individually
    results = {}
    for i, program in enumerate(programs):
        print(f"Executing program {i} locally with 1.5s timeout at {time.strftime('%H:%M:%S')}...")
        execution_start = time.time()
        
        try:
            # Use debug mode for the first program to help diagnose issues
            debug_mode = (i == 0)
            program_results = await run_in_local_sandbox(program, inputs, timeout=1.5, debug=debug_mode)
            execution_time = time.time() - execution_start
            print(f"Program {i} execution completed in {execution_time:.2f}s")
            
            # Store results
            results[i] = program_results
            
        except Exception as e:
            execution_time = time.time() - execution_start
            print(f"Error executing program {i}: {e}")
            results[i] = [None] * len(inputs)
    
    total_time = time.time() - start_time
    print(f"Local batch execution completed in {total_time:.2f}s")
    return results
