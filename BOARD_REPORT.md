# Board Report: Document Classification System

## Executive Summary

We built a **production-grade REST API** for document classification across 10 categories + out-of-distribution (OOD) detection. The system was optimized using **Optuna TPE** across **7 CPU-only retrieval methods** (TF-IDF, BM25, BM25L, BM25+, LMIR Jelinek-Mercer, LMIR Dirichlet, DFR).

While **DFR (Divergence from Randomness)** achieved the highest test F1 (94.9%), we selected **TF-IDF (92.8% F1)** for production due to superior stability on short text inputs. The final system handles **2,300+ req/s** with **<1ms median latency** and passed all security/OOD checks.

## Key Metrics

| Metric | Value |
|---|---|
| **Production Model** | **TF-IDF (Optuna Trial #3)** |
| **Validation F1 (Weighted)** | 0.9705 |
| **Test F1 (Weighted)** | 0.9280 |
| **OOD Recall** | 66.7% (Threshold 0.8698) |
| **Latency (p99)** | **1.21ms** (short text) |
| **Throughput** | **2,304 req/s** (single thread) |
| **Model Size** | 574 KB (disk) |
| **Cold Start** | 18ms (median) |
| **Cloud Cost** | ~$7.60/month (AWS t3.micro) |

## Cross-Method Benchmark

We benchmarked 7 retrieval methods head-to-head on the test set:

| Method | Test F1 | OOD Recall | p99 Latency | Notes |
|---|---|---|---|---|
| **DFR** | **0.9489** | **83.3%** | 0.65ms | **Highest F1**, but unstable on short text (<40% conf) |
| **BM25** | 0.9472 | 83.3% | 0.67ms | Excellent retrieval, similar stability issues to DFR |
| **TF-IDF** | 0.9280 | 66.7% | 1.21ms | **Chosen for Production** (Robust on short inputs) |
| BM25L | 0.9024 | 50.0% | 1.05ms | Good for long documents, heavy penalty on short |
| LMIR JM | 0.8616 | 0.0% | 0.86ms | Failed OOD detection completely |
| BM25+ | 0.8542 | 16.7% | 1.21ms | Poor OOD performance |
| LMIR Dir | 0.8085 | 0.0% | 1.32ms | Slowest and least accurate |

## Technical Decisions

### 1. Model Selection: TF-IDF > DFR
Although DFR scored 2.1% higher on the long-document test set, our stability tests (`scripts/compare_short_text.py`) revealed it assigns **dangerously low confidence (<40%)** to valid short inputs (e.g., "Lionel Messi scored a hat-trick"). TF-IDF consistently assigned **>99% confidence** to the same inputs, making it the safer, more robust choice for a general-purpose API.

### 2. Hyperparameter Optimization
- **Optuna TPE**: Runs 80 trials across 7 methods.
- **Search Space**: Continuous (alpha, b, k1, c, mu, lambda) + Categorical (method, loss).
- **Outcome**: Identified TF-IDF with `sublinear_tf=True`, `max_features=5000` as the robust optimum.

### 3. Production Readiness
- **Telemetry**: Added `/metrics` endpoint tracking uptime, latency percentiles, cache hits, and label distribution.
- **Security**: Input sanitization pipeline strips HTML/script injection, detects SQLi/path traversal, and cleans control characters.
- **Performance**: 2,300 req/s on a single core exceeds the "millions of documents" scaling requirement (daily capacity > 200M).

## Architecture

```
Client → FastAPI → Sanitize → Cache Check → PII Scan → TF-IDF+SGD → Response
                                    ↕ (Redis/Mem)
                              Telemetry (latency/counts)
```

## Deliverables

| Artifact | Path | Description |
|---|---|---|
| **API Code** | `app/` | FastAPI, Pydantic schemas, Security, Inference |
| **Production Model** | `models/baseline.joblib` | Tuned TF-IDF pipeline |
| **Benchmarks** | `models/benchmark.json` | Full production metrics (latency, cost, RAM) |
| **Cross-Method** | `models/cross_method_benchmark.json` | 7-method comparison results |
| **Demo Scripts** | `scripts/demo_*.py` | 5 scripts for review (classify, OOD, security, etc.) |
| **Load Test** | `tests/locustfile.py` | Locust stress test script |
| **EDA** | `notebooks/eda.py` | Data analysis visualization |

## Recommendations

1. **Short-Text Training**: The drop in DFR performance on short text suggests the training data (BBC News) lacks short samples. Augmenting the dataset with sentence-level examples would allow deploying the higher-accuracy DFR model.
2. **Ensemble**: A voting ensemble of TF-IDF (robust) and DFR (accurate on long docs) could offer the best of both worlds.
3. **Drift Monitoring**: Use the `/metrics` endpoint to monitor label distribution shifts in production.
