# ARC DSL Solver

A minimal DSL (Domain-Specific Language) implementation for solving ARC (Abstraction and Reasoning Corpus) tasks.

## Quick Start

```bash
# Install dependencies
uv init
uv add numpy matplotlib tqdm pydantic
uv sync # if cloning the repo

# Run on a single task
uv run cli/run_task.py 0a1d4ef5 --depth 4 --show

# Run on a dataset
uv run cli/run_dataset.py ../arc-data/mit-easy.json --depth 4 --parallel 32 --save-dir results
```

## How It Works

This implementation uses a simple DSL approach to solve ARC tasks:

1. **DSL Primitives**: A set of basic grid operations defined in `dsl/primitives.py` (rotate, flip, tile, etc.)
2. **Program Search**: Iterative deepening search over programs up to a specified depth (`search/enumerator.py`)
3. **Verification**: Testing candidate programs against training examples (`search/verifier.py`)
4. **Visualization**: Displaying inputs, outputs, and predictions (`io/visualizer.py`)

## Project Structure

- `dsl_utils/`: Core DSL implementation
  - `primitives.py`: Basic grid operations
  - `types.py`: Type system for grids and objects
  - `program.py`: Program representation and execution
- `search/`: Search and verification
  - `enumerator.py`: Program enumeration with iterative deepening
  - `heuristics.py`: Pruning strategies
  - `verifier.py`: Program verification against examples
- `io/`: Input/output utilities
  - `loader.py`: Load tasks from JSON files
  - `visualizer.py`: Visualize grids and results
- `cli/`: Command-line interfaces
  - `run_task.py`: Run solver on a single task
  - `run_dataset.py`: Run solver on multiple tasks
- `examples/`: Example notebooks
  - `01_demo.ipynb`: Demonstration of the solver

## Troubleshooting

- **Solver sits forever**: Lower `--depth` or set `--timeout` to a smaller value.
- **Colors look wrong**: Check `vmin`/`vmax` in visualization code.
- **Import errors**: Make sure you're running from the right directory.
- **Memory issues**: Reduce the search depth or add more aggressive pruning.

## Performance Tips

- Keep the maximum depth ≤ 4 to avoid combinatorial explosion
- Use the shape heuristic to prioritize promising operations
- Cache program results to avoid redundant computation
- Use parallel processing for dataset runs

## Example

For a detailed example, see the [demo notebook](examples/01_demo.ipynb).
