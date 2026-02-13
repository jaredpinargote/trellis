# `experiments/` — Exploratory Analysis

Ad-hoc analysis scripts that informed design decisions but are not part of the production pipeline.

## Structure

```
experiments/
└── notebooks/
    └── compare_short_text.py   → Short-text accuracy analysis across retrieval methods
```

## Purpose

This script was used to investigate the short-text accuracy gap (45% → 73%) that led to the data augmentation strategy in `pipelines/data_prep/11_augment_data.py`. It compares how different retrieval methods handle documents under 50 characters.

## Design Decisions

- **Separated from `pipelines/`**: Experiments are one-off explorations, not reproducible pipeline steps. They may have hardcoded paths or require manual interpretation.
- **Kept in repo**: Documents the analytical process behind key decisions (augmentation, DFR selection) for reviewers who want to understand the "why" behind the model choice.
