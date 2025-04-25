#!/usr/bin/env python3
"""
OpenAI Client Test Script

This script tests the OpenRouter API using the OpenAI client with the correct configuration.
"""
import os
import sys
import json
from pathlib import Path

# Try to load from dotenv
try:
    from dotenv import load_dotenv
    # Load .env file from the parent directory (greenblatt)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    print(f"Loaded .env from: {env_path}")
except ImportError:
    print("Warning: python-dotenv not installed")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

# Get the API key
api_key = os.environ.get('OPENROUTER_API_KEY')
if not api_key:
    print("ERROR: OPENROUTER_API_KEY is not set")
    sys.exit(1)

# Print the API key info
print(f"API key length: {len(api_key)}")
print(f"API key prefix: {api_key[:10]}...")

# Import OpenAI client
from openai import OpenAI

# Create client with correct configuration
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

print("\nSending request to OpenRouter API...")

try:
    # Make the request with extra_headers
    completion = client.chat.completions.create(
        model="google/gemini-2.0-flash-001",
        messages=[
            {"role": "user", "content": "Say hello world"}
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/TrelisResearch/arc-demo",
            "X-Title": "ARC-Greenblatt-Demo"
        }
    )
    
    # Print the response
    print("\nAPI call successful!")
    print(f"Response: {completion.choices[0].message.content}")
    print(f"Model: {completion.model}")
    print(f"Token usage: {completion.usage}")
    
    # Print the full response as JSON
    response_dict = {
        "id": completion.id,
        "model": completion.model,
        "choices": [
            {
                "index": choice.index,
                "message": {
                    "role": choice.message.role,
                    "content": choice.message.content
                },
                "finish_reason": choice.finish_reason
            } for choice in completion.choices
        ],
        "usage": {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens
        }
    }
    print(f"\nFull response: {json.dumps(response_dict, indent=2)}")
    
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e)}")
    if hasattr(e, 'response'):
        print(f"Response status: {e.response.status_code}")
        print(f"Response body: {e.response.text}")
    if hasattr(e, '__dict__'):
        print(f"Error attributes: {e.__dict__}")
