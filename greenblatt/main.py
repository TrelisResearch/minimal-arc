"""Main entry point for the Greenblatt-style ARC demo.

This file allows running the demo from the greenblatt directory using:
uv run main.py --task-id 00576224 --k 8 --visualize
"""
import sys
import os
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the current directory
    env_path = Path('.') / '.env'
    load_dotenv(dotenv_path=env_path)
    print("Loaded environment variables from .env file")
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables must be set manually.")
except Exception as e:
    print(f"Warning: Failed to load .env file: {e}")

import asyncio
from cli.main import main_async

def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
