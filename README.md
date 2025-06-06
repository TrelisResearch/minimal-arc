# Minimal ARC

A repository for working with the Abstraction and Reasoning Corpus (ARC) dataset.

Apply to join the Trelis ARC AGI 2 Team [here](https://trelis.com/arc-agi-2/).

Check out Trelis' video explaining ARC [here](https://youtu.be/GbCgyY1laDU).

## Very Basic Examples

If you're new to ARC AGI, read through these very simple examples:
- [Domain Specific Language Approach = combine basic operations](https://chatgpt.com/share/68191c05-fd1c-8003-ab24-8c157a3af550)
- [Neural Net Approach = train on examples](https://chatgpt.com/share/6810cc68-cf38-8003-aace-3630952bbeb6)
- [LLM guided program search = write python solvers using LLMs](https://chatgpt.com/share/6810cc4c-e790-8003-8aac-b6f8ab2213e8)

Then, move to the comprehensive examples in the `dsl`, `llmgs` and `ttt` folders.

## Scripts

### clean_arc_data.py

This script filters ARC data files to keep only examples with single test inputs/outputs.

**Functionality:**
- Removes examples with multiple tests or solutions
- Preserves the original data structure
- Outputs statistics about removed examples

**Usage:**
```bash
uv venv
cd arc-data
uv run python clean_arc_data.py
```

**Output:**
- Creates an `arc-data-cleaned` directory
- Saves filtered versions of the original files
- Prints statistics about the cleaning process

## Folders

- `arc-data`: Original ARC data files
- `arc-data-cleaned`: Cleaned ARC data files
- `dsl`: Domain Specific Language approach (create basic programs and try to combine them to solve the training examples. Deterministic approach.).
- `llmgs`: LLM guided search (get an LLM to keep writing programs until one passes on train examples).
- `ttt`: Test-time training (train a neural net to predict outputs, then add depth first search).
