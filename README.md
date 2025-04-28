# Minimal ARC

A repository for working with the Abstraction and Reasoning Corpus (ARC) dataset.

Apply to join the Trelis ARC AGI 2 Team [here](https://trelis.com/arc-agi-2/).

Check out Trelis' video explaining ARC [here - forthcoming].

Todo:
[x] Greenblatt
    [x] [Sandboxing](https://chatgpt.com/c/6807b43f-8798-8003-87dc-79c42119a063)
    [x] [Writing a long script](https://chatgpt.com/c/6801632f-5864-8003-b9a5-3144b0bf695a)
[ ] Test time training (TTT)
    [ ] Review current script
    [ ] Add pre-training on the "train" split

## Scripts

### clean_arc_data.py

This script filters ARC data files to keep only examples with single test inputs/outputs.

**Functionality:**
- Removes examples with multiple tests or solutions
- Preserves the original data structure
- Outputs statistics about removed examples

**Usage:**
```bash
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
- `greenblatt`: The Ryan Greenblatt approach of generating many python programs and then testing them.
- `ttt`: Test-time training.
