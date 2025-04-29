# ARC DSL Solver

A minimal DSL (Domain-Specific Language) implementation for solving ARC (Abstraction and Reasoning Corpus) tasks.

## Notes on Getting DSL to Work.

1. Primitives must be unary, i.e. you must define a set of operations on the input grid that ONLY require the input grid to be passed. What the operation does (rotate, flip, flood fill) should be entirely described by the oepration. Often this means creating multiple versions of one operation (e.g. a flood fill operation for each of ten colours).
2. You don't want too many primitive operations as that increases the search space. Ideally you want operations to be fairly specific - not just have a fill operation for each individual position, but some general fills, e.g. border fill. AT THIS POINT, YOU SHOULD BE ABLE TO SOLVE AT LEAST ONE MIT-EASY PROBLEM.
3. ADD HASHES FOR OUTPUTS/INTERMEDIATE STATES ALREADY VISITED, SO THAT THOSE ARE PRUNED. (Memoisation).
4. Stop search early if an intermediate state is reached that will not be able to reach the output within the remaining depth. (e.g. intermediate grid has more colours compared to output than can be removed with primative operations).

## Quick Start

```bash
# Install dependencies
uv init
uv add numpy matplotlib tqdm pydantic
uv sync # if cloning the repo

# Run on a single task
uv run cli/run_task.py 3194b014 --depth 4 --show --data-path ../arc-data-cleaned --timeout 5 --debug

# Run on a dataset
uv run cli/run_dataset.py ../arc-data/mit-easy.json --depth 4 --timeout 60 --save-dir results --data-path ../arc-data-cleaned
```

## Command Line Options

### Running a Single Task

```bash
uv run cli/run_task.py <task_id> [options]
```

Options:
- `--depth INT`: Maximum search depth (default: 4)
- `--timeout FLOAT`: Search timeout in seconds (default: 15.0)
- `--show`: Show visualization
- `--save PATH`: Save visualization to file
- `--data-path PATH`: Path to the data directory
- `--parallel`: Use parallel search (default: True)
- `--num-processes INT`: Number of processes to use (default: CPU count - 1) # that's cpu count minus one.
- `--op-timeout FLOAT`: Timeout for individual operations in seconds (default: 0.25)
- `--debug`: Print debug information

### Running a Dataset

```bash
uv run cli/run_dataset.py <json_file> [options]
```

Options:
- `--depth INT`: Maximum search depth (default: 4)
- `--timeout FLOAT`: Search timeout in seconds (default: 15.0)
- `--parallel INT`: Number of parallel processes (default: CPU count - 1)
- `--data-path PATH`: Path to the data directory
- `--save-dir PATH`: Directory to save results
- `--results-file PATH`: File to save results (default: results.json)
- `--op-timeout FLOAT`: Timeout for individual operations in seconds (default: 0.25)

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
- **Search Termination**: The search stops as soon as a valid program is found that correctly solves all training examples. Test outputs are only used for evaluation after a solution is found, not to guide the search.
- **Timeout Control**: A configurable timeout prevents excessive search time.

### Heuristics and Pruning

The search process uses several heuristics to guide and prune the search space:

- **Type Checking**: Ensures operations have compatible input/output types (actively used).
- **Symmetry Pruning**: Eliminates redundant operation sequences like consecutive involutions or full rotations (actively used).
- **Shape Heuristic**: Uses input/output shapes to prioritize operations that are likely to be useful (actively used).
- **Similarity Heuristic**: Implemented but not actively used in the main search process. Could be used with A* search.
- **A* Search**: Implemented but not currently used in the main task solver. The system primarily relies on iterative deepening.

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

## Performance Optimizations

### Optimized Primitives

The DSL has been optimized in several ways to improve search efficiency and runtime performance:

1. **Pre-grounded Operations**: Instead of parametric operations, we use concrete, argument-free versions to eliminate runtime parameter guessing. For example, instead of a generic `replace_color(grid, old_color, new_color)`, we have specific operations like `replace_0_to_1(grid)`.

2. **Reduced Primitive Set**: The primitive set has been carefully curated to minimize the branching factor while maintaining expressiveness:
   - Only operations that return Grid_T are included (not Int_T) for most search tasks
   - Only essential color replacement operations are included
   - Crop operations are limited to the most commonly used patterns

3. **Optimized Flood Fill**: Several flood fill operations have been optimized:
   - `fill_holes_fn`: Fills holes in the grid by flood filling from the border
   - `fill_background_X_fn`: Fills the background connected to the border with specific colors
   - `flood_object_fn`: Flood fills starting from the top-left non-zero pixel
   - All flood fill operations use `collections.deque` for O(n) performance

4. **Timeout Controls**: Individual operations have timeout limits to prevent excessive runtime on large grids.

### Runtime Efficiency

The system includes several runtime optimizations:

1. **Operation Timeouts**: Each operation has a configurable timeout (default: 0.25s) to prevent long-running operations from stalling the search.

2. **Parallel Processing**: Both individual task solving and dataset processing support parallel execution to utilize multiple CPU cores.

3. **Error Handling**: Robust error handling for timeouts and exceptions prevents the search from getting stuck on problematic programs.

4. **Dynamic Primitive Summary**: The system displays a summary of available primitives at startup, showing the number of operations by category.

### Search Space Reduction

By reducing the number of primitives from ~80 to ~40, we achieve approximately a 5x speedup for depth-4 searches, making it feasible to solve more complex tasks.

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
