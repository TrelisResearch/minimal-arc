# Greenblatt-style ARC Demo

This project implements a Ryan-Greenblatt-style approach to solving Abstraction and Reasoning Corpus (ARC) tasks. It uses LLMs to generate many program candidates, filters them based on training examples, and uses majority voting to determine the final output.

## Overview

The Greenblatt approach works as follows:

1. **Program Generation**: Generate many program candidates using an LLM (Gemini Flash 2.0)
2. **Filtering**: Test each program against training examples and filter out invalid ones
3. **Majority Voting**: Run all valid programs on the test input and take the majority vote as the final output

## Project Structure Explanation

- `core/`: Contains the core functionality for program generation and evaluation
- `sandbox/`: Handles safe execution of generated code
- `viz/`: Provides visualization tools for ARC grids
- `cli/`: Contains the command-line interface
- `agent/`: Empty placeholder for potential future extensions (not currently used)

## Setup

1. Install dependencies:
```bash
uv init
uv add httpx openai pydantic-ai-slim numpy matplotlib tqdm orjson ujson mcp
```

2. Create a `.env` file with your OpenRouter API key:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. Start the MCP server in a separate terminal:
```bash
deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python stdio
```

## Usage

Run the main script from the `greenblatt` directory:
```bash
uv run --with mcp main.py --task-id <task_id> --k <num_programs> [--visualize]
```

### Parameters

- `--task-id`: ID of the ARC task to solve (e.g., "00576224")
- `--k`: Number of programs to generate (default: 8)
- `--concurrency`: Number of concurrent API calls (default: 32)
- `--temperature`: Temperature for generation (default: 1.0)
- `--visualize`: Flag to visualize the task and solutions

## Testing the MCP Sandbox

To test the MCP sandbox functionality, run:
```bash
uv run --with mcp test-scripts/test_simplified_runner.py
```

This will execute a simple grid transformation in the MCP sandbox and verify the results.

## Environment Setup

### API Key

The demo requires an OpenRouter API key to generate programs. You have two options to set this up:

1. **Using a .env file (recommended)**:
   - Copy the `sample.env` file to `.env` in the same directory
   - Add your OpenRouter API key to the `.env` file
   - The application will automatically load variables from this file

2. **Setting environment variables directly**:
   ```bash
   export OPENROUTER_API_KEY='your-api-key'
   ```

You can get an API key by signing up at [OpenRouter](https://openrouter.ai/).

## MCP Server Setup

Before running the demo, you need to set up the MCP server for sandboxed code execution in a separate terminal:

1. Install Deno if you don't have it already:
```bash
curl -fsSL https://deno.land/install.sh | sh
```

2. Start the MCP server using stdio transport in a separate terminal:
```bash
deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python stdio
```

**Important**: Keep this server running in a separate terminal while using the demo. The sandbox runner will connect to this server to execute the generated code safely.

## Visualization

The `--visualize` flag will display the input grids, expected output grids (if available), and the candidate output grid. If `--save-viz` is specified, the visualizations will be saved to the specified directory.

## Performance Tuning

- **Generation Parameters**: Adjust `--k` and `--concurrency` based on your needs and rate limits
- **Temperature**: Higher values (e.g., 1.0) produce more diverse programs, lower values produce more focused ones
- **Concurrency**: Adjust based on your API rate limits and available resources

## Cost Management

The demo uses OpenRouter to access Gemini Flash 2.0, which has the following approximate pricing:
- Input tokens: ~$0.075/M
- Output tokens: ~$0.30/M

To manage costs:
- Start with smaller `--k` values (e.g., 50-100) for testing
- Use the `--task-id` option to run on a single task first
- Monitor token usage in the console output

## Troubleshooting

- **API Key Issues**: Ensure your OpenRouter API key is correctly set
- **Deno Installation**: If you encounter issues with Deno, install it manually:
  ```bash
  curl -fsSL https://deno.land/install.sh | sh
  ```
- **MCP Server**: If the MCP server fails to start, try running it manually:
  ```bash
  deno run -A jsr:@pydantic/mcp-run-python sse --port 4321
  ```
- **Memory Issues**: If you encounter memory issues, reduce the `--k` value

## Examples

### Example 1: Running on a Single Task

```bash
uv run main.py --task-id 00576224 --k 8 --visualize
```

### Example 2: Running on MIT-Easy Tasks

```bash
uv run main.py \
  --task-file ../arc-data-cleaned/mit-easy.json \
  --k 8 \
  --concurrency 32 \
  --save-results results/mit_easy_results.json \
  --visualize \
  --save-viz visualizations/mit_easy/
```

### Example 3: Custom Data File

```bash
uv run main.py \
  --task-id 00576224 \
  --data-file ../arc-data-cleaned/arc-agi_training_challenges.json \
  --k 8 \
  --visualize
