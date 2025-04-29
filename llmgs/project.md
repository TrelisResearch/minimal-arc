Below is a ** setup_demo.sh** skeleton plus a detailed build plan.
The script only scaffolds the project-tree and installs the right dependencies; every file it writes is a stub with TODO placeholders so you can fill in, test and iterate.
(You asked not to implement the full logic yet, only to give you the structure and guidance.)

#!/usr/bin/env bash
# =========================================================
# setup_demo.sh – scaffold a Ryan-Greenblatt–style ARC demo
# =========================================================
# 1. Creates a clean Python venv
# 2. Installs core deps (OpenRouter client, asyncio/httpx,
#    pydantic-ai + mcp-run-python, numpy/matplotlib, etc.)
# 3. Writes stub modules under ./greenblatt/
# 4. Pulls the official mcp-run-python server (needs deno)
# ---------------------------------------------------------
# USAGE
#   chmod +x setup_demo.sh
#   ./setup_demo.sh
#
# Afterwards open README_DEV.md for next steps.
# ---------------------------------------------------------

set -e
PROJECT_ROOT=$(pwd)
echo "Project root = ${PROJECT_ROOT}"

PYTHON_VERSION=${PYTHON_VERSION:-"3.11"}
VENV_DIR=".venv-arc-demo"

echo "🛠  Creating virtualenv (${VENV_DIR}) …"
python${PYTHON_VERSION} -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate
pip install --upgrade pip

echo "📦  Installing Python packages …"
pip install \
  "httpx[http2]" \
  openai \
  pydantic-ai-slim[mcp] \
  "uvicorn~=0.29" \
  numpy \
  matplotlib \
  tqdm \
  rich

# optional: faster JSON & TOML
pip install "orjson" "ujson"

echo "🌐  Installing & caching mcp-run-python assets …"
# Requires Deno ≥1.43 – user must have it in PATH.
deno install -qf -n deno_placeholder https://deno.land/std/examples/welcome.ts || true
deno cache jsr:@pydantic/mcp-run-python || true

echo "📁  Generating stub source tree …"
mkdir -p greenblatt/{agent,cli,core,sandbox,viz}

# ---------- stub files -------------------------------------------------
cat <<'PY' > greenblatt/core/prompt_template.txt
You are an expert Python programmer. Learn the grid-to-grid transformation
from the TRAINING examples and write a **pure function**:
```python
def solve(grid: List[List[int]]) -> List[List[int]]:
    ...
Constraints:

No I/O, network, or random numbers
Only builtin Python + itertools, math, copy TRAINING: {{TRAIN_BLOCK}} TEST_INPUT: {{TEST_BLOCK}} Return code only, no commentary. PY
cat <<'PY' > greenblatt/core/generate_programs.py """ generate_programs.py

Async helper that:

Builds the chat prompt (from prompt_template.txt + task JSON)
Hits the OpenRouter 'chat/completions' endpoint with model="google/gemini-2.0-flash-001" # or gemini-flash-2.0-latest-001
Streams / batches k completions (uses 'n' per request for efficiency)
Yields raw code strings NOTE: Fill the TODO sections before running. """ from future import annotations import os, asyncio, httpx, json, time from typing import AsyncIterator, List
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") API_BASE = "https://openrouter.ai/api/v1" MODEL_ID = "google/gemini-2.0-flash-001" # ⇐ confirm slug in dashboard

async def sample_programs(prompt: str, k: int, batch_size: int = 5, temperature: float = 1.0, concurrency: int = 4) -> AsyncIterator[str]: """ Streams ~k completions; respects OpenRouter rate-limits with a semaphore. """ sem = asyncio.Semaphore(concurrency) async with httpx.AsyncClient(base_url=API_BASE, timeout=30.0) as client: async def _single_call() -> List[str]: req = { "model": MODEL_ID, "temperature": temperature, "n": batch_size, "messages": [ {"role": "user", "content": prompt} ] } hdrs = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "https://github.com/your-handle/arc-demo", "X-Title": "ARC-Greenblatt-Demo"} async with sem: r = await client.post("/chat/completions", headers=hdrs, json=req) r.raise_for_status() data = r.json() return [c["message"]["content"] for c in data["choices"]]

    remaining = k
    while remaining > 0:
        got = await _single_call()
        for code in got:
            yield code
        remaining -= len(got)
PY

cat <<'PY' > greenblatt/sandbox/runner.py """ runner.py – thin wrapper around MCP:run-python Starts (or reuses) an MCP server to sandbox-execute LLM-generated code.

Uses pydantic_ai.mcp.MCPServerStdio to spin up 'deno run jsr:@pydantic/mcp-run-python stdio'
Provides run_in_sandbox(code: str, inputs: list[list]) -> list[list] """
TODO: wire together MCP client session, timeout guards, result parsing

PY

cat <<'PY' > greenblatt/core/evaluate.py """ evaluate.py – unit-tests candidate programs against train pairs Filters out programs that: • raise exceptions • produce mismatched outputs or wrong grid sizes """

TODO: implement strict equality + early bail-out on first failure

PY

cat <<'PY' > greenblatt/viz/show_grids.py """ show_grids.py – quick matplotlib visualizer Usage: python -m llmgs.viz.show_grids path/to/task.json Draws: input(s), expected output(s), candidate LLM output """

TODO: use plt.imshow & custom ListedColormap

PY

cat <<'PY' > greenblatt/cli/main.py """ CLI entrypoint.

Options

--task-id <hash> --task-file <json list> # e.g. arc-data-cleaned/mit-easy.json --k <int> # samples per task --concurrency <int> --batch-size <int> # completions per API call Samples programs, filters them, majority-votes test output, and prints / saves results. """

TODO: argparse boilerplate + orchestration

PY

-----------------------------------------------------------------------

echo "✅ Scaffolding complete." echo "🔑 Remember to: export OPENROUTER_API_KEY='<your-key>'" echo "📝 Next: open greenblatt/cli/main.py and start coding!"


---

## How the pieces fit together (implementation guide)

| Stage | Key file(s) | What to add | Gotchas & tips |
|-------|-------------|-------------|----------------|
| **Prompt assembly** | `core/prompt_template.txt` + helper inside `generate_programs.py` | Insert grid examples with minimal whitespace – every extra token costs money. Include *one* solved toy task to nudge the model toward 2-D reasoning. | Gemini Flash tokens are cheaper than GPT-4o but still add up. Trim prompt aggressively (no base-64 grids, no color legends). |
| **Program sampling** | `core/generate_programs.py` | Use `n` completions per request (e.g. `n=5`) to amortize prompt tokens. Fire multiple HTTP requests concurrently with a semaphore. | OpenRouter rate-limits by token/min; obey `Retry-After` header and exponential-back-off on 429. |
| **Sandbox execution** | `sandbox/runner.py` | Start **mcp-run-python** once and reuse. For each candidate: ① inject the code into the sandbox, ② run `solve(grid)` for each training input, ③ capture stdout/stderr. | *mcp-run-python* spins Pyodide inside Deno – import time on first run is ~1 s; warm-up to avoid latency. Limit per-call wall-time (e.g. 2 s) and memory. |
| **Unit-test filter** | `core/evaluate.py` | Reject programs on first failure to save cycles. Keep a hash-set to deduplicate identical code. | Watch for silent failures: some code returns `None` instead of grid; enforce type/shape checks. |
| **Voting / selection** | inside `cli/main.py` | After filtering, run all valid programs on the test input(s) and take the *mode* (majority) grid. | Ties are rare; break by `(first_seen)` or by grid hash order. |
| **Visualisation** | `viz/show_grids.py` | Build a 3×N subplot: original input, ground-truth output (if available), best candidate output. Use `matplotlib.colors.ListedColormap` with the 10 ARC colors. | Keep colour indices consistent (0-9); set `plt.axis('off')` for clarity. |
| **Evaluation harness** | extend `evaluate.py` | When you pass a *file* (e.g. `mit-easy.json` list), iterate over IDs, call the pipeline, compare result to solutions in `arc-data-cleaned`. | Some public ARC subsets contain tasks with *multiple* test inputs; handle each and compute per-task “all correct” metric. |
| **Concurrency tuning** | controlled in `cli/main.py` | A good default: `k=200`, `batch_size=5`, `concurrency=6` ⇒ ~200/5 = 40 HTTP calls. | Gemini Flash returns in 0.2-0.8 s; make sure you don’t saturate the 60k TPM quota. |
| **Cost tracking** | add a small utility | Collect `usage` from each OpenRouter response and accumulate. Print `$ spent` at the end. | Gemini Flash 2.0 pricing (Mar 2025): ~$0.075/M input, $0.30/M output tokens – cheaper than GPT-4o. |

### mcp-run-python quick-start

```bash
# install deno if not present
curl -fsSL https://deno.land/install.sh | sh
# warm-up & launch server on port 4321 (SSE mode)
deno run \
  -A jsr:@pydantic/mcp-run-python sse --port 4321
Inside Python:

from pydantic_ai.mcp import MCPClientSSE
mcp = await MCPClientSSE.connect("http://127.0.0.1:4321")
sandbox_id = await mcp.create_sandbox(image="python:3.11-slim")
out = await mcp.execute_python_code(
        sandbox_id=sandbox_id,
        code="from solution import solve\nprint(solve(inp))",
        timeout=2.0)
Command-line usage examples
# Single task demo
python -m llmgs.cli.main \
   --task-id 00576224 \
   --k 300 --batch-size 5 --concurrency 4

# Evaluate a whole subset
python -m llmgs.cli.main \
   --task-file arc-data-cleaned/mit-easy.json \
   --k 400 --batch-size 5 --concurrency 8 \
   --save-results results_mit_easy.json
Caveats & common pitfalls
Prompt overflow – Gemini Flash has a 1 048 576-token context, but requests larger than ~512 k tokens are rejected; your prompt + completion length × n must stay well under this.
Sandbox cold-starts – the first Pyodide load can take ~900 ms; keep the server alive for batch evaluations.
Colour index confusion – ARC uses integers 0-9; treat 0 as not blank, unlike some 2020 write-ups that zero out black.
Rate-limits – OpenRouter enforces per-minute and per-day token caps. Capture HTTP 429 and retry after Retry-After.
Gemini quirks – the Flash model occasionally emits badly indented code. Strip Markdown fences and run ast.parse() for a fast validity check before sandboxing.
Cost drift – Small changes in k and batch_size compound fast. Track cumulative usage.input_tokens/output_tokens to stay under your $25 budget.
Next step: run ./setup_demo.sh, open the generated stubs, and start implementing each TODO. Once cli/main.py is wired up, you’ll have a lightweight, educational replica of Greenblatt-style program search that you can scale from a single MIT-easy task to a full evaluation set.
Good hacking! 