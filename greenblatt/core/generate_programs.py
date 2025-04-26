"""generate_programs.py

Async helper that:

Builds the chat prompt (from prompt_template.txt + task JSON)
Hits the OpenRouter 'chat/completions' endpoint with model="google/gemini-2.0-flash-001" # or gemini-flash-2.0-latest-001
Streams / batches k completions (uses 'n' per request for efficiency)
Yields raw code strings
"""
from __future__ import annotations
import os
import asyncio
import json
import time
import ast
from typing import AsyncIterator, List, Dict, Any, Optional, Tuple
from pathlib import Path
from openai import AsyncOpenAI

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.0-flash-001"  # or "gemini-flash-2.0-latest-001"

# Debug info for API key
if OPENROUTER_API_KEY:
    masked_key = OPENROUTER_API_KEY[:4] + '*' * (len(OPENROUTER_API_KEY) - 8) + OPENROUTER_API_KEY[-4:] if len(OPENROUTER_API_KEY) > 8 else '****'
    print(f"Using OpenRouter API key: {masked_key}")
else:
    print("WARNING: OPENROUTER_API_KEY is not set")

def load_prompt_template() -> str:
    """Load the prompt template from the file."""
    template_path = Path(__file__).parent / "prompt_template.txt"
    with open(template_path, "r") as f:
        return f.read()

def format_grid(grid: List[List[int]]) -> str:
    """Format a grid as a compact string representation."""
    return json.dumps(grid)

def build_prompt(task_data: Dict[str, Any], task_id: str) -> str:
    """Build the prompt from the template and task data."""
    template = load_prompt_template()
    
    # Format training examples
    train_examples = []
    for example in task_data[task_id]["train"]:
        train_examples.append(f"Input: {format_grid(example['input'])}")
        train_examples.append(f"Output: {format_grid(example['output'])}")
    train_block = "\n".join(train_examples)
    
    # Format test input
    test_input = task_data[task_id]["test"][0]["input"]
    test_block = format_grid(test_input)
    
    # Replace placeholders
    prompt = template.replace("{{TRAIN_BLOCK}}", train_block)
    prompt = prompt.replace("{{TEST_BLOCK}}", test_block)
    
    return prompt

def extract_code(response: str) -> Optional[str]:
    """Extract code from the response, handling markdown code blocks and thinking steps."""
    # Extract thinking steps if present
    thinking = ""
    if "<think>" in response and "</think>" in response:
        think_parts = response.split("<think>", 1)
        if len(think_parts) > 1:
            thinking_content = think_parts[1].split("</think>", 1)[0].strip()
            thinking = thinking_content
            # Remove the thinking part from the response for code extraction
            response = response.replace(f"<think>{thinking_content}</think>", "").strip()
    
    # Try to extract code from markdown code blocks
    code_part = None
    if "```python" in response:
        parts = response.split("```python", 1)
        if len(parts) > 1:
            code_part = parts[1].split("```", 1)[0].strip()
    elif "```" in response:
        parts = response.split("```", 1)
        if len(parts) > 1:
            code_part = parts[1].split("```", 1)[0].strip()
    else:
        # If no code blocks found, use the whole response
        code_part = response.strip()
    
    # If we have a code part and thinking, combine them
    if code_part:
        if thinking:
            # Format thinking as a multiline comment block at the top of the code
            formatted_thinking = "\n".join([f"# {line}" for line in thinking.split("\n")])
            return f"# THINKING START\n{formatted_thinking}\n# THINKING END\n\n{code_part}"
        return code_part
    
    return None

def is_valid_python(code: str) -> bool:
    """Check if the code is valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

async def sample_programs(
    prompt: str, 
    k: int, 
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 20, 
    concurrency: int = 32
) -> AsyncIterator[str]:
    """
    Streams ~k completions; respects OpenRouter rate-limits with a semaphore.
    
    Args:
        prompt: The prompt to send to the API
        k: Number of programs to generate
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter (1-100)
        concurrency: Number of concurrent API calls
    
    Yields:
        Generated program strings
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency)
    
    # Create a queue for completed programs
    queue = asyncio.Queue()
    
    # Create the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=API_BASE,
        api_key=OPENROUTER_API_KEY,
    )
    
    async def sample_program_with_backoff() -> None:
        """Sample a single program with backoff for rate limits."""
        # Implement exponential backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with semaphore:
                    # Make the request with extra_headers
                    response = await client.chat.completions.create(
                        model=MODEL_ID,
                        temperature=temperature,
                        top_p=top_p,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        extra_headers={
                            "HTTP-Referer": "https://github.com/TrelisResearch/arc-demo",
                            "X-Title": "ARC-Greenblatt-Demo"
                        }
                    )
                    
                    # Track token usage if available
                    if hasattr(response, 'usage'):
                        print(f"Token usage: {response.usage}")
                    
                    # Extract and validate code from responses
                    valid_code_found = False
                    for choice in response.choices:
                        code = extract_code(choice.message.content)
                        if code and is_valid_python(code):
                            valid_code_found = True
                            await queue.put(code)
                    
                    # If no valid code was found, signal completion anyway
                    if not valid_code_found:
                        print(f"DEBUG: No valid code found in response, putting None in queue")
                        await queue.put(None)
                    
                    # Signal completion
                    return
            
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 429:  # Rate limit exceeded
                    # Get retry-after header or use exponential backoff
                    retry_after = int(getattr(e, 'headers', {}).get("Retry-After", 2 ** attempt))
                    print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                else:
                    # For other errors, print more detailed error information
                    print(f"Error: {e}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"Retrying after {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Failed after {max_retries} attempts")
                        print(f"DEBUG: Task failed, putting None in queue")
                        await queue.put(None)  # Signal failure
                        return
        
        # If we get here, all retries failed
        print(f"DEBUG: All retries failed, putting None in queue")
        await queue.put(None)  # Signal failure
    
    # Create tasks for API calls
    tasks = []
    for i in range(k):
        task = asyncio.create_task(sample_program_with_backoff())
        tasks.append(task)
    
    # Add a background task to monitor the state
    async def monitor_state():
        while True:
            # Check task status
            done_tasks = sum(1 for t in tasks if t.done())
            pending_tasks = len(tasks) - done_tasks
            
            # Check queue size (approximate)
            queue_size = queue.qsize() if hasattr(queue, 'qsize') else "unknown"
            
            print(f"DEBUG: State - remaining: {remaining}, done_tasks: {done_tasks}/{len(tasks)}, queue_size: {queue_size}")
            
            await asyncio.sleep(5)
    
    monitor_task = asyncio.create_task(monitor_state())
    
    try:
        # Process completed programs as they arrive
        remaining = k
        while remaining > 0:
            try:
                print(f"DEBUG: Waiting for item from queue, remaining: {remaining}")
                program = await queue.get()
                print(f"DEBUG: Got item from queue: {'valid program' if program else 'None'}")
                
                if program is not None:
                    yield program
                    remaining -= 1
                    print(f"DEBUG: Yielded program, remaining: {remaining}")
                else:
                    remaining -= 1
                    print(f"DEBUG: Got None, remaining: {remaining}")
            except Exception as e:
                print(f"DEBUG: Error in queue.get(): {e}")
                # Decrement remaining to avoid infinite loop
                remaining -= 1
                print(f"DEBUG: Error handling, remaining: {remaining}")
        
        print(f"DEBUG: Exited collection loop, remaining: {remaining}")
    finally:
        # Cancel the monitor task
        monitor_task.cancel()
        
        # Cancel any remaining tasks
        for task in tasks:
            if not task.done():
                print(f"DEBUG: Cancelling unfinished task")
                task.cancel()
    
    # Wait for all tasks to complete
    print(f"DEBUG: Waiting for all tasks to complete")
    await asyncio.gather(*tasks, return_exceptions=True)
    print(f"DEBUG: All tasks completed")

async def sample_programs_with_usage(
    prompt: str, 
    k: int, 
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 20, 
    concurrency: int = 32
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Sample programs from the API with token usage information.
    
    Args:
        prompt: The prompt to send to the API
        k: Number of programs to generate
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter (1-100)
        concurrency: Number of concurrent API calls
        
    Yields:
        Tuples of (program, token_usage)
    """
    count = 0
    
    async for program in sample_programs(
        prompt=prompt,
        k=k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        concurrency=concurrency
    ):
        # Get the code from the completion
        code = extract_code(program)
        if code:
            count += 1
            # For now, we don't have per-completion token usage
            # so we'll just return the completion as token_usage
            yield code, None

async def generate_programs_for_task(
    task_data: Dict[str, Any], 
    task_id: str, 
    k: int, 
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 20, 
    concurrency: int = 32
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Generate programs for a specific task.
    
    Args:
        task_data: Dictionary of task data
        task_id: Task ID
        k: Number of programs to generate
        temperature: Temperature for generation (0.0-2.0)
        top_p: Top-p sampling parameter (0.0-1.0)
        top_k: Top-k sampling parameter (1-100)
        concurrency: Number of concurrent API calls
        
    Yields:
        Tuples of (program, token_usage)
    """
    prompt = build_prompt(task_data, task_id)
    
    async for program, token_usage in sample_programs_with_usage(
        prompt=prompt,
        k=k,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        concurrency=concurrency
    ):
        yield program, token_usage

def load_task_data(task_file: str) -> Dict[str, Any]:
    """Load task data from a JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)

def load_task_ids(task_list_file: str) -> List[str]:
    """
    Load a list of task IDs from a JSON file.
    
    Handles both dictionary format (returns keys) and list format (returns the list).
    """
    with open(task_list_file, 'r') as f:
        data = json.load(f)
        
        # Handle both dictionary and list formats
        if isinstance(data, dict):
            return list(data.keys())
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected format in {task_list_file}. Expected dict or list, got {type(data)}")
