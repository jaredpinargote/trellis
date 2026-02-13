# `pipelines/training/` — Model Training & Benchmarks

Scripts for training, hyperparameter optimization, and cross-method comparison.

## Pipeline Order

| Step | Script | Purpose |
|------|--------|---------|
| 4 | `4_train_baseline.py` | SGDClassifier + TF-IDF baseline (fast, ~94% F1) |
| 5 | `5_train_xgboost.py` | XGBoost with TF-IDF features |
| 6 | `6_train_setfit.py` | SetFit (few-shot sentence transformer) |
| 7 | `7_hyperparam_search.py` | Optuna hyperparameter search for the selected model |
| 8 | `8_optuna_retrieval_search.py` | Optuna search across 7 retrieval methods (TF-IDF, BM25, DFR, etc.) |
| 9 | `9_inference_benchmark.py` | Latency and throughput profiling per model |
| 10 | `10_cross_method_benchmark.py` | Head-to-head comparison of all 7 retrieval methods |

### Retraining Scripts
| Script | Purpose |
|--------|---------|
| `retrain_dfr.py` | Retrain with DFR (Divergence From Randomness) vectorizer — the winning method |
| `retrain_tfidf_augmented.py` | Retrain TF-IDF with augmented data |
| `retrain_tfidf_final.py` | Final TF-IDF model with tuned parameters |

## Design Decisions

- **7 retrieval methods tested**: TF-IDF, BM25, DFR, LMDirichlet, IB, DFI, TF-IDF+SVD. DFR won on F1 with acceptable latency.
- **Optuna over GridSearch**: More efficient hyperparameter exploration, especially for the 7-method search space (Step 8).
- **SGDClassifier over SVM**: Linear classifier with partial_fit support — can scale to larger datasets without full retraining.
- **Output**: All training scripts produce `models/baseline.joblib` — a dict with keys `pipeline`, `threshold`, `model_version`.
