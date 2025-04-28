Below is a cook-book–style build plan for a mini ARC DSL + search demo that sits in a new folder called dsl/, parallel to arc-data-cleaned/.
The emphasis is on clarity, visualization, and short runtimes (single tasks or mit-easy subsets).
All code snippets are Python-style pseudocode—fill in the details later.

1 Folder/Module Layout

/dsl
│
├── requirements.txt          # minimal deps
├── dsl-utils/
│   ├── primitives.py         # grid ops: rotate, flip, flood_fill, color_mask …
│   ├── types.py              # tiny type system (Grid, ObjList, Int, …)
│   └── program.py            # Program class: list[Op], run(), repr()
│
├── search/
│   ├── enumerator.py         # iterative-deepening / A* over Programs
│   ├── heuristics.py         # type-checking, symmetry pruning, timeout, …
│   └── verifier.py           # applies candidate to train pairs
│
├── io/
│   ├── loader.py             # read single-task JSON or dataset file
│   └── visualizer.py         # matplotlib imshow grid utilities
│
├── cli/
│   ├── run_task.py           # solve one task id
│   └── run_dataset.py        # loop over a file of ids (mit-easy.json …)
│
├── examples/                 # tiny notebooks / md walkthroughs
│   └── 01_demo.ipynb
└── README.md
Why this split?
dsl/ is 100 % pure functions—easy to unit-test and, if desired, sandbox.
search/ isolates the expensive logic; you can profile/replace heuristics without touching DSL code.
io/ keeps all file and plotting code in one place, so the solver stays headless for CI runs.
cli/ shows users exactly how to launch the demo—no digging.
2 Dependencies (requirements.txt)

numpy>=1.25      # fast array ops
matplotlib>=3.8  # visualization
tqdm             # progress bars
pydantic>=2      # config + type hints
# Optional:
# mcp-run-python  # sandbox runner if you need locked-down exec
Gotcha: Keep the list tiny—large packages slow down installs on a fresh laptop.
3 Core DSL Design (dsl/primitives.py)

Each primitive is a pure function plus a lightweight type signature:

@dataclass
class Op:
    name: str
    fn: Callable        # fn(grid | objlist | int, ...) -> grid | objlist | int
    in_type: Type       # from dsl.types
    out_type: Type
    commutes_with: set[str] = field(default_factory=set)  # for symmetry pruning

# examples  (implement bodies later)
ROT90     = Op("rot90", rot90_fn, Grid, Grid)
FLIP_H    = Op("flip_h", flip_h_fn, Grid, Grid)
COLORMASK = Op("mask_color", color_mask_fn, Grid, Grid)
FILL      = Op("flood_fill", flood_fill_fn, Grid, Grid)
OBJECTS   = Op("objects", find_objects_fn, Grid, ObjList)
BBOX      = Op("bbox", get_bbox_fn, ObjList, Grid)
Start with ≈ 10–15 primitives; depth-4 search is then ≤ 15⁴ ≈ 50 000 paths—tractable.
Add more only when you hit tasks that require them.

4 Program Container (dsl/program.py)

class Program:
    def __init__(self, ops: list[Op]):
        self.ops = ops

    def run(self, grid: Grid) -> Grid:
        val = grid
        for op in self.ops:
            val = op.fn(val)
        return val

    def types_ok(self) -> bool:
        # statically verify in/out chain
        ...
Tip: Cache Program.run() results by hash of (task_id, example_idx, program_repr) to avoid recomputation.

5 Enumeration & Search (search/enumerator.py)

High-level pseudocode (trim for depth-first, BFS, or A*):

def iter_deepening(primitives, max_depth):
    for depth in range(1, max_depth + 1):
        yield from enumerate_programs([], depth)

def enumerate_programs(prefix, remaining):
    if remaining == 0:
        yield Program(prefix)
    else:
        for op in primitives:
            # type-signature check: prefix_out_type == op.in_type
            if not type_flow_ok(prefix, op):
                continue
            # simple redundancy cut: rot180∘rot180, flip_h∘flip_h, etc.
            if breaks_symmetry(prefix, op):
                continue
            yield from enumerate_programs(prefix + [op], remaining - 1)
Early-exit heuristics (search/heuristics.py)
Type flow – reject chains whose intermediate types mismatch.
Symmetry pruning – if op is an involution (op∘op==identity), forbid doubling.
Shape heuristic – if train input & output shapes match (or differ by simple swap), push rotate/flip early.
Timeout – wrap the entire call in signal.alarm(t_sec); raise exception to abort long runs.
6 Verification Loop (search/verifier.py)

def verify(program: Program, train_pairs):
    for inp, out_expected in train_pairs:
        if program.run(inp) != out_expected:
            return False
    return True
Stop at first failure—cheap short-circuit.

7 CLI Entrypoints

cli/run_task.py
python cli/run_task.py 00576224 --depth 4
Skeleton:

def main(task_id, depth, timeout, show):
    task = io.loader.load_task(task_id)
    program = search.solve(task, depth, timeout)
    if program is None:
        print("No solution found")
        return
    if show:
        io.visualizer.compare_grids(
            input=task["test"][0]["input"],
            prediction=program.run(task["test"][0]["input"]),
            label="predicted vs ground truth?"
        )
cli/run_dataset.py
python cli/run_dataset.py arc-data-cleaned/mit-easy.json --depth 4 --parallel 4
Tiny loop:

ids = io.loader.load_id_list(json_file)
with Pool(args.parallel) as pool:
    pool.map(solve_id, ids)
8 Visualization (io/visualizer.py)

def show_grid(grid, ax=None, title=""):
    # Use matplotlib.imshow with discrete colormap
    ax.imshow(np.array(grid), interpolation="nearest", vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis("off")

def compare_grids(input, prediction, target=None, label=""):
    fig, axs = plt.subplots(1, 3 if target is not None else 2, figsize=(8, 3))
    show_grid(input, axs[0], "Input")
    show_grid(prediction, axs[1], "Prediction")
    if target is not None:
        show_grid(target, axs[2], "Target")
    fig.suptitle(label)
    plt.show()
Gotcha: vmin=0, vmax=9 works for standard ARC color palette (10 colors).
Use plt.cm.get_cmap('tab10', 10) for distinct color mapping.

9 Sandboxing with mcp-run-python (optional)

When enumerating DSL programs you execute your own trusted primitives—risk is low.
If you later let users inject code or want total isolation:

from mcp_run_python import sandbox

def safe_run(program, grid):
    @sandbox(max_exec_time_ms=200)
    def _inner():
        return program.run(grid)
    return _inner()
Hints:

Wrap only the run step, not the enumeration (that would spawn thousands of sandboxes).
Increase max_exec_time_ms if flood-fill on large grids times out.
10 Example Notebook (examples/01_demo.ipynb)

Load task 00576224.
Visualize train input/output.
Run solver with depth ≤ 4.
Print discovered program (rot90 ∘ tile ∘ …).
Visualize test prediction vs. expected solution (from arc-data-cleaned/…solutions.json).
Timing cell—show it finished in < 15 s.
This single notebook is the showpiece you reference in your README.

11 README Checklist

Quick-start: pip install -r requirements.txt, then python cli/run_task.py ....
How it works – short bullet list linking to dsl/ and search/.
Troubleshooting
“Solver sits forever” → lower --depth or set --timeout.
“Colors look wrong” → check vmin/vmax.
Credits & licence.
12 Caveats / Gotchas


Issue	Mitigation
Combinatorial blow-up on harder tasks	Keep max_depth ≤4, prune aggressively, early timeout.
Mutable grids corrupting state	Always grid.copy() before mutating in primitive implementations.
Symmetric grids causing duplicate states	Hash intermediate grids; skip if seen.
Some tasks need object lists rather than raw grids	Include objects, bbox, paint_objects primitives; add ObjList type in dsl/types.py.
JSON file can be large	Stream-load tasks by id instead of reading whole dict if memory tight.
Matplotlib slows batch runs	Gate visual code behind --show flag.
signal.alarm doesn’t work on Windows	Use multiprocessing.Process with join(timeout) as fallback.
mcp-run-python overhead	Use only for untrusted code; disable (safe=False) in default CLI for speed.
13 Minimum Working Example (MWE) Pseudocode

# core solver one-liner (glue code)
def solve(task, max_depth=4, timeout=30):
    for program in search.enumerator.iter_deepening(DSL_PRIMITIVES, max_depth):
        if time_exceeded(): break
        if search.verifier.verify(program, task["train"]):
            return program
    return None
On mit-easy tasks depth 3–4 this finishes in seconds–tens-of-seconds on a modern laptop CPU.

Next Steps
Expand primitives incrementally (resize, transpose, diagonal-mirror, etc.).
Plug in a lightweight neural prior (small MLP or rule-based heuristic) to rank primitives before enumeration.
Cache solved programs to dsl/solutions_cache.json for instant replay.
Follow the folder plan, implement each stub, and you’ll have a clean, educational ARC DSL solver that meets your runtime and visualization goals.