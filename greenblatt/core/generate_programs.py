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
    """Extract code from the response, handling markdown code blocks."""
    # Try to extract code from markdown code blocks
    if "```python" in response:
        parts = response.split("```python", 1)
        if len(parts) > 1:
            code_part = parts[1].split("```", 1)[0].strip()
            return code_part
    elif "```" in response:
        parts = response.split("```", 1)
        if len(parts) > 1:
            code_part = parts[1].split("```", 1)[0].strip()
            return code_part
    
    # If no code blocks found, use the whole response
    return response.strip()

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
    concurrency: int = 32
) -> AsyncIterator[str]:
    """Streams ~k completions; respects OpenRouter rate-limits with a semaphore."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    sem = asyncio.Semaphore(concurrency)
    
    # Create the AsyncOpenAI client
    client = AsyncOpenAI(
        base_url=API_BASE,
        api_key=OPENROUTER_API_KEY,
    )
    
    async def _single_call() -> List[str]:
        # Implement exponential backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                async with sem:
                    # Make the request with extra_headers
                    response = await client.chat.completions.create(
                        model=MODEL_ID,
                        temperature=temperature,
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
                    codes = []
                    for choice in response.choices:
                        code = extract_code(choice.message.content)
                        if code and is_valid_python(code):
                            codes.append(code)
                    
                    return codes
            
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
                        return []
        
        # If we get here, all retries failed
        return []

    remaining = k
    while remaining > 0:
        got = await _single_call()
        for code in got:
            yield code
            remaining -= 1
            if remaining <= 0:
                break

async def sample_programs_with_usage(
    prompt: str, 
    k: int, 
    temperature: float = 1.0, 
    concurrency: int = 32
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Sample programs from the API with token usage information.
    
    Args:
        prompt: The prompt to send to the API
        k: Number of programs to generate
        temperature: Temperature for generation
        concurrency: Number of concurrent API calls
        
    Yields:
        Tuples of (program, token_usage)
    """
    count = 0
    
    async for program in sample_programs(
        prompt=prompt,
        k=k,
        temperature=temperature,
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
    concurrency: int = 32
) -> AsyncIterator[Tuple[str, Any]]:
    """
    Generate programs for a specific task.
    
    Args:
        task_data: Dictionary of task data
        task_id: Task ID
        k: Number of programs to generate
        temperature: Temperature for generation
        concurrency: Number of concurrent API calls
        
    Yields:
        Tuples of (program, token_usage)
    """
    prompt = build_prompt(task_data, task_id)
    
    async for program, token_usage in sample_programs_with_usage(
        prompt=prompt,
        k=k,
        temperature=temperature,
        concurrency=concurrency
    ):
        yield program, token_usage

def load_task_data(task_file: str) -> Dict[str, Any]:
    """Load task data from a JSON file."""
    with open(task_file, 'r') as f:
        return json.load(f)

def load_task_ids(task_list_file: str) -> List[str]:
    """Load a list of task IDs from a JSON file."""
    with open(task_list_file, 'r') as f:
        return json.load(f)
