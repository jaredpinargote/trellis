# `pipelines/` — Data Preparation & Model Training

Numbered scripts for the full ML pipeline, from raw data to trained model artifacts. Scripts are numbered in execution order.

## Structure

```
pipelines/
├── data_prep/     → Steps 1-3, 11: data consolidation, splitting, validation, augmentation
└── training/      → Steps 4-10: model training, hyperparameter search, benchmarks
```

## Design Decisions

- **Numbered naming**: Scripts are prefixed `1_`, `2_`, etc. to make execution order obvious.
- **Separated from `app/`**: Training code is NOT imported by the API. It produces artifacts (`models/baseline.joblib`) consumed by the API at runtime.
- **Reproducibility**: Each script writes results to `data/` or `models/` and prints metrics to stdout for audit trails.
