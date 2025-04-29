# ARC DSL Solver

A minimal DSL (Domain-Specific Language) implementation for solving ARC (Abstraction and Reasoning Corpus) tasks.

## Notes on Getting DSL to Work.

1. Primitives must be unary, i.e. you must define a set of operations on the input grid that ONLY require the input grid to be passed. What the operation does (rotate, flip, flood fill) should be entirely described by the oepration. Often this means creating multiple versions of one operation (e.g. a flood fill operation for each of ten colours).
2. You don't want too many primitive operations as that increases the search space. Ideally you want operations to be fairly specific - not just have a fill operation for each individual position, but some general fills, e.g. border fill.

## Quick Start

```bash
# Install dependencies
uv init
uv add numpy matplotlib tqdm pydantic
uv sync # if cloning the repo

# Run on a single task
uv run cli/run_task.py 692cd3b6 --depth 4 --show --parallel

# Run on a dataset
uv run cli/run_dataset.py ../arc-data/mit-easy.json --depth 6 --parallel 32 --save-dir results
```

## How It Works

This implementation uses a simple DSL approach to solve ARC tasks:

1. **DSL Primitives**: A set of basic grid operations defined in `dsl/primitives.py` (rotate, flip, tile, etc.)
2. **Program Search**: Iterative deepening search over programs up to a specified depth (`search/enumerator.py`)
3. **Verification**: Testing candidate programs against training examples (`search/verifier.py`)
4. **Visualization**: Displaying inputs, outputs, and predictions (`io/visualizer.py`)

## Search Capabilities

The DSL includes a search algorithm that can find programs to solve ARC tasks:

- **Iterative Deepening**: The search starts with simple programs (depth 1) and gradually increases complexity.
- **Parallelization**: The search can utilize multiple CPU cores to speed up the process.
- **Heuristics**: Shape-based heuristics guide the search toward promising programs.
- **Timeout Control**: A configurable timeout prevents excessive search time.

### Using Parallel Search

To use parallel search, set the `parallel` parameter to `True` when calling `iter_deepening`:

```python
from dsl.search.enumerator import iter_deepening, ALL_PRIMITIVES

# Sequential search (default)
for program in iter_deepening(ALL_PRIMITIVES, max_depth=4, input_shape=(3, 3), output_shape=(3, 3)):
    # Process program...

# Parallel search
for program in iter_deepening(ALL_PRIMITIVES, max_depth=4, input_shape=(3, 3), output_shape=(3, 3), 
                             parallel=True):
    # Process program...

# Parallel search with custom number of processes
for program in iter_deepening(ALL_PRIMITIVES, max_depth=4, input_shape=(3, 3), output_shape=(3, 3),
                             parallel=True, num_processes=4):
    # Process program...
```

By default, the parallel search will use `(number of CPU cores - 1)` processes to keep your system responsive.

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
