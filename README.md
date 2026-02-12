# Document Classification System

A production-grade REST API for classifying documents into 10 categories with out-of-distribution (OOD) detection, optimized using **Optuna** across **7 CPU-only retrieval methods**.

**Test F1: 92.8%** | **7 Retrieval Methods** | **Optuna 80-Trial TPE** | **25/25 Tests Passing**

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (Optuna search across 7 retrieval methods)
python scripts/8_optuna_retrieval_search.py

# 4. Start API
python scripts/run_api.py
# → http://127.0.0.1:8000/docs
```

## API Usage

### Classify a Document

```bash
curl -X POST http://127.0.0.1:8000/classify_document \
  -H "Content-Type: application/json" \
  -d '{"document_text": "The quarterback threw a touchdown pass in the championship game."}'
```

**Response:**

```json
{
  "message": "Classification successful",
  "label": "sport",
  "confidence": 0.9872,
  "is_ood": false,
  "processing_time_ms": 1.23,
  "model_version": "optuna_tfidf"
}
```

## Retrieval Methods

The system explores 7 CPU-only retrieval methods as feature extractors:

| Method | Type | Key Parameters |
|---|---|---|
| **TF-IDF** | Term weighting | `max_features`, `ngram_range`, `sublinear_tf` |
| **BM25 (Okapi)** | Probabilistic | `k1`, `b` |
| **BM25L** | BM25 variant | `k1`, `b`, `delta` |
| **BM25+** | BM25 variant | `k1`, `b`, `delta` |
| **LMIR Jelinek-Mercer** | Language model | `lambda` |
| **LMIR Dirichlet** | Language model | `mu` |
| **DFR** | Divergence from Randomness | `c` |

## Categories

`business` · `entertainment` · `food` · `graphics` · `historical` · `medical` · `politics` · `space` · `sport` · `technologie` · `other` (OOD)

## Testing

```bash
# Unit + Security tests (25 tests)
python -m pytest tests/ -v

# Load test (requires API running)
python -m locust -f tests/locustfile.py --headless -u 50 -r 10 -t 30s --host http://127.0.0.1:8000
```

## Hyperparameter Search

### Optuna (Recommended)
```bash
python scripts/8_optuna_retrieval_search.py
```
- 80 trials, TPE sampler, 7 retrieval methods
- Per-trial 30s timeout with skip logging
- Results saved to `models/results.json`

### Manual Grid Search (Legacy)
```bash
python scripts/7_hyperparam_search.py
```

## Docker

```bash
docker build -t doc-classifier .
docker run -p 8000:8000 doc-classifier
```

## Project Structure

```
├── app/
│   ├── main.py          # FastAPI application
│   ├── schemas.py       # Request/response models
│   ├── inference.py     # Model wrapper + OOD detection
│   ├── security.py      # Input sanitization + PII detection
│   └── cache.py         # Redis caching layer
├── scripts/
│   ├── 8_optuna_retrieval_search.py  # Optuna + 7 retrieval methods
│   ├── 7_hyperparam_search.py        # Manual 10-config search
│   └── run_api.py       # API launcher
├── tests/
│   ├── test_api.py      # 15 API endpoint tests
│   ├── test_security.py # 10 sanitization tests
│   └── locustfile.py    # High-throughput load test
├── notebooks/
│   └── eda.py           # Exploratory data analysis
├── models/
│   ├── baseline.joblib  # Trained model artifact
│   └── results.json     # Optuna search results (80 trials)
├── data/training/       # Train/val/test CSV splits
├── Dockerfile           # Production container
├── BOARD_REPORT.md      # Executive summary
└── requirements.txt     # Python dependencies
```
