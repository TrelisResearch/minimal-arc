# TODO List for Greenblatt ARC Demo

This document outlines suggested improvements and refactorings for the Greenblatt ARC Demo project, organized by category.

1. Architecture & Structure
   - Consolidate CLI scripts:
     * Merge `main.py`, `cli/main.py`, and `run_k_analysis.py` into a unified CLI with subcommands (e.g., `generate`, `evaluate`, `analyze`).
     * Leverage a CLI framework like `click` or `typer` for clearer argument parsing and subcommand organization.
   - Package Installation:
     * Convert the project into a Python package (`setup.py`/`pyproject.toml`) and define entry points for CLI commands.
   - Module Reorganization:
     * Move standalone scripts (`run_k_analysis.py`, `check_solutions.py`) under a `cli/` directory.
     * Rename `test-scripts/` to `tests/` and integrate with `pytest`.

2. Configuration & Environment
   - Centralize configuration:
     * Create a `config.py` or YAML/JSON config file for default timeouts, concurrency, and sandbox settings.
     * Consolidate timeout parameters and remove magic numbers.
   - `.env` handling:
     * Rename `sample.env` to `.env.example` and add support for automatic loading via `python-dotenv`.
   - Dependency Management:
     * Remove committed `node_modules/`; add to `.gitignore`.
     * Provide a `requirements.txt` or keep `pyproject.toml` as single source of truth.

3. Sandboxing & Security
   - Enhance local executor:
     * Improve AST-based sandbox to detect and block dangerous code patterns.
     * Set per-program execution time limits and memory limits.
   - MCP integration:
     * Combine MCP and local executor behind a common interface.
     * Provide clear fallback logic when the MCP server is unavailable.

4. Performance & Scalability
   - Concurrency management:
     * Use asynchronous concurrency (e.g., `asyncio`) for both LLM calls and program evaluation.
     * Implement rate limiting to avoid API throttling.
   - Caching:
     * Cache LLM responses by prompt hash to prevent duplicate API calls.
   - Bulk evaluation:
     * Evaluate candidate programs in parallel batches; collect and aggregate results.

5. Testing & Validation
   - Migrate existing scripts in `test-scripts/` to `pytest` test cases.
   - Write unit tests for `core.generate_programs` and `core.evaluate`, covering edge cases:
     * Empty or invalid grids
     * Timeouts and runtime errors
     * No valid programs found scenario
   - Add integration tests for the end-to-end pipeline on a small set of ARC tasks.

6. Documentation & Examples
   - Update root `README.md`:
     * Add a project overview with user-oriented examples.
     * Document unified CLI usage with subcommands.
   - Consolidate `greenblatt/README.md` into the root README or link appropriately.
   - Fix duplicated "Example 3" numbering and path references.

7. Logging & Monitoring
   - Replace `print` statements with Python's `logging` module.
   - Add configurable log levels and structured log outputs.
   - Instrument metrics (e.g., tasks/second, success rates) and expose via a `--metrics` flag.

8. Results & Reporting
   - Standardize output formats (e.g., JSON schema) for results files.
   - Provide a post-processing tool to aggregate `run_k_analysis` outputs and generate plots.

9. Miscellaneous
   - Handle ties in majority voting and define a fallback strategy (e.g., best-of-one).
   - Ensure path handling is cross-platform (use `pathlib`).
   - Set a fixed random seed for reproducibility where randomness is used.
