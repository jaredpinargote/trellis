# Document Classification System

A production-grade REST API for classifying documents into 10 categories with out-of-distribution (OOD) detection.

**Test F1: 96.2%** | **Inference: < 25ms** | **25/25 Tests Passing**

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model (hyperparameter search across 10 configs)
python scripts/7_hyperparam_search.py

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
  "model_version": "tuned_v9_no_stopwords"
}
```

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

## Categories

`business` · `entertainment` · `food` · `graphics` · `historical` · `medical` · `politics` · `space` · `sport` · `technologie` · `other` (OOD)

## Testing

```bash
# Unit + Security tests (25 tests)
python -m pytest tests/ -v

# Security tests only
python -m pytest tests/test_security.py -v

# Load test (requires API running)
python -m locust -f tests/locustfile.py --headless -u 50 -r 10 -t 30s --host http://127.0.0.1:8000
```

## Hyperparameter Search

The model was selected from 10 TF-IDF + SGD configurations:

```bash
python scripts/7_hyperparam_search.py
```

Results saved to `models/results.json`. Winner: **v9_no_stopwords** (F1=0.9616).

## EDA

```bash
python notebooks/eda.py
```

Generates 5 analysis charts in `notebooks/`:
- `class_distribution.png` — class balance across splits
- `document_lengths.png` — character/word distributions
- `tfidf_top_terms.png` — top TF-IDF features per class
- `ood_analysis.png` — OOD class imbalance justification
- `hyperparam_search.png` — 10-config comparison

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
│   ├── 7_hyperparam_search.py  # 10-config tuning
│   └── run_api.py       # API launcher
├── tests/
│   ├── test_api.py      # 15 API endpoint tests
│   ├── test_security.py # 10 sanitization tests
│   └── locustfile.py    # High-throughput load test
├── notebooks/
│   └── eda.py           # Exploratory data analysis
├── models/
│   ├── baseline.joblib  # Trained model artifact
│   └── results.json     # Hyperparameter search results
├── data/training/       # Train/val/test CSV splits
├── Dockerfile           # Production container
├── BOARD_REPORT.md      # Executive summary
└── requirements.txt     # Python dependencies
```
