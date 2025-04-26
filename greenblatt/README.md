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

## MCP Server Setup

Before running the demo with the default sandbox mode, you need to set up the MCP server for sandboxed code execution in a separate terminal:

1. Install Deno if you don't have it already:
```bash
curl -fsSL https://deno.land/install.sh | sh
```

2. Start the MCP server using stdio transport in a separate terminal:
```bash
deno run -N -R=node_modules -W=node_modules --node-modules-dir=auto jsr:@pydantic/mcp-run-python stdio
```

**Important**: Keep this server running in a separate terminal while using the demo with the default sandbox mode. The sandbox runner will connect to this server to execute the generated code safely.

## Sandbox Options

The system supports two different execution environments for running generated programs:

### MCP Sandbox (Default)

By default, the system uses the MCP sandbox for code execution, which provides strong isolation but requires a separate MCP server to be running (see MCP Server Setup above).

### Local Executor (--no-sandbox)

For faster execution, you can use the `--no-sandbox` flag to run code in a lightweight local executor:

```bash
uv run main.py --task-id 00576224 --k 8 --visualize --no-sandbox
```

The local executor:
- Uses AST-based security to restrict imports and dangerous operations
- Runs code directly in the Python process without the overhead of the MCP server
- Is significantly faster than the MCP sandbox, especially for batch processing
- Automatically handles type annotations from the generated code

Note: The local executor is not a full sandbox and should only be used in trusted environments.

## Visualization

The `--visualize` flag will display the input grids, expected output grids (if available), and the candidate output grid. If `--save-viz` is specified, the visualizations will be saved to the specified directory.

The visualization includes:
- Training examples with input and expected output
- Test examples with input and expected output (if available)
- Candidate output with an indicator showing whether it's correct (✓) or incorrect (✗)

## Evaluation Methodology

### Valid Programs

A program is considered "valid" if:
1. It runs without errors
2. It produces the correct output for **all** training examples

### Task Solution

A task is considered "solved" if:
1. There are valid programs
2. The majority vote of those valid programs produces the correct output for the test example

### Pass@K Evaluation

This implementation uses a simplified approach that approximates pass@1 evaluation:
- We generate K programs for each task
- We check how many of those programs are valid
- We use a majority vote among valid programs to determine the final output

Note that this differs from the standard ARC pass@K methodology, which involves multiple independent runs and a more sophisticated statistical analysis. The standard ARC evaluation typically measures the probability of solving a task with K attempts.

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
- **Memory Issues**: If you encounter memory issues, reduce the `--k` value

## Examples

### Example 1: Running on a Single Task

```bash
uv run main.py --task-id 00576224 --k 8 --visualize --no-sandbox
```

### Example 2: Running on MIT-Easy Tasks

```bash
uv run main.py --task-file ../arc-data/mit-easy.json --data-file ../arc-data-cleaned/arc-agi_evaluation_challenges.json --k 32 --concurrency 32 --save-results results/mit_easy_results.json --visualize --save-viz visualizations/mit_easy/ --no-sandbox
```

### Example 3: Custom Data File

```bash
uv run main.py --task-id 00576224 --data-file ../arc-data-cleaned/arc-agi_training_challenges.json --k 8 --visualize --no-sandbox
```

### Example 3: Analyzing Performance with Different k Values

To analyze how performance varies with different values of k, you can use the `run_k_analysis.py` script:

```bash
uv run run_k_analysis.py --task-file ../arc-data/mit-easy.json --data-file ../arc-data-cleaned/arc-agi_evaluation_challenges.json --solutions-file ../arc-data-cleaned/arc-agi_evaluation_solutions.json --k-values 2,8,32 --concurrency 32 --output-dir results/k_analysis --no-sandbox
```

This will:
1. Run the specified tasks with k=2, k=8, and k=32
2. Save the results for each k value in the output directory
3. Generate a plot showing how the number of correct tasks varies with k
4. Print a summary of the results

The script uses our majority voting approach, which is more strict than the standard pass@k metric used in ARC benchmarks. In our approach, a task is only considered correct if the majority of valid programs produce the correct output.
