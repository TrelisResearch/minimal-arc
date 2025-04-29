#!/usr/bin/env python3
"""
Direct OpenRouter Test Script

This script tests the OpenRouter API using the requests library for maximum transparency.
"""
import os
import sys
import json
import requests
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

# Print the full API key for debugging
print(f"Full API key: {api_key}")
print(f"API key length: {len(api_key)}")

# API configuration
api_base = "https://openrouter.ai/api/v1"
model_id = "google/gemini-2.0-flash-001"

# Prepare the request
headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://github.com/TrelisResearch/arc-demo",
    "X-Title": "ARC-Greenblatt-Demo",
    "Content-Type": "application/json"
}

data = {
    "model": model_id,
    "messages": [
        {"role": "user", "content": "Say hello world"}
    ]
}

# Print request details
print("\n=== REQUEST ===")
print(f"URL: {api_base}/chat/completions")
print("Headers:")
for name, value in headers.items():
    # Mask the Authorization header value
    if name.lower() == "authorization":
        value_parts = value.split()
        if len(value_parts) > 1:
            value = f"{value_parts[0]} {value_parts[1][:4]}...{value_parts[1][-4:]}"
    print(f"  {name}: {value}")
print(f"Body: {json.dumps(data, indent=2)}")

# Make the request
try:
    print("\nSending request to OpenRouter API...")
    response = requests.post(
        f"{api_base}/chat/completions",
        headers=headers,
        json=data
    )
    
    # Print response details
    print("\n=== RESPONSE ===")
    print(f"Status: {response.status_code} {response.reason}")
    print("Headers:")
    for name, value in response.headers.items():
        print(f"  {name}: {value}")
    
    # Try to parse the response as JSON
    try:
        response_data = response.json()
        print(f"Body: {json.dumps(response_data, indent=2)}")
    except:
        print(f"Body: {response.text}")
    
    # Check if the request was successful
    if response.status_code == 200:
        print("\nAPI call successful!")
        if "choices" in response_data and response_data["choices"]:
            print(f"Response: {response_data['choices'][0]['message']['content']}")
        if "usage" in response_data:
            print(f"Token usage: {response_data['usage']}")
    else:
        print(f"\nAPI call failed with status code {response.status_code}")
        
except Exception as e:
    print(f"\nError: {e}")
    print(f"Error type: {type(e)}")
