# `pipelines/data_prep/` — Data Preparation

Scripts to transform raw data into clean, split, validated training sets.

## Pipeline Order

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `1_consolidate_data.py` | Merges multiple raw data sources into a single CSV |
| 2 | `2_split_data.py` | Stratified train/validation/test split (preserves class distribution) |
| 3 | `3_validate_csv.py` | Schema and integrity checks on the split CSVs |
| 11 | `11_augment_data.py` | Data augmentation for short-text robustness (added after initial training revealed short-text weakness) |

## Design Decisions

- **Stratified splitting**: Ensures minority classes appear proportionally in all splits — critical for reliable F1 metrics.
- **Augmentation added late** (Step 11): Short-text accuracy was 45% during initial demos. Augmentation improved it to ~73%. Numbered `11_` to indicate it was added after the main pipeline.
- **Validation as a separate step**: Catches data corruption before expensive training runs (missing labels, encoding issues, class imbalance).
