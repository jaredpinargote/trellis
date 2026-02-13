# Document Classification API

A production-ready REST API for classifying documents into 10 categories (business, politics, sport, etc.) with out-of-distribution (OOD) detection. Optimized via Optuna across 7 retrieval methods, achieving **92.8% F1** and **2,300+ req/s**.

## 🚀 Quick Start for Reviewers

### 1. Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API
Start the server (keep this terminal open):
```bash
python scripts/run_api.py
```
> API will run at `http://127.0.0.1:8000`. Docs at `/docs`.

### 3. Run Demo Scripts (in a new terminal)

Run these scripts to verify production readiness:

| Script | Description |
|---|---|
| `python scripts/demo_classify.py` | **Accuracy Check**: Classifies samples from all 11 categories. |
| `python scripts/demo_ood.py` | **OOD Check**: Verifies that gibberish/off-topic text is rejected. |
| `python scripts/demo_security.py` | **Security Check**: Tests SQLi, XSS, and path traversal handling. |
| `python scripts/demo_stress.py` | **Load Test**: Fires 200 concurrent requests to measure latency. |
| `python scripts/demo_metrics.py` | **Telemetry**: Generates traffic and shows the `/metrics` dashboard. |

---

## 📊 Performance & Benchmarks

Full details in [BOARD_REPORT.md](BOARD_REPORT.md).

- **Accuracy**: 92.8% Test F1 (Weighted)
- **Throughput**: 2,304 requests/sec (single thread)
- **Latency**: 1.21ms (p99)
- **Cold Start**: 18ms
- **Model Size**: 574 KB
- **Cost**: ~$7.60/month (AWS t3.micro)

### Model Selection
We benchmarked 7 methods (TF-IDF, BM25, DFR, etc.). While **DFR** had higher test accuracy (+2%), it proved unstable on short text inputs (<40% confidence). We selected **TF-IDF** for production as it maintains >99% confidence on valid inputs while still achieving excellent F1 scores.

---

## 🛠️ Project Structure

```
├── app/
│   ├── main.py          # FastAPI app with /classify_document, /metrics
│   ├── inference.py     # Model loading & prediction logic
│   ├── security.py      # Input sanitization (XSS, SQLi, PII)
│   ├── schemas.py       # Pydantic data models
│   └── cache.py         # In-memory LRU cache
│
├── models/
│   ├── baseline.joblib  # PRODUCTION MODEL (TF-IDF)
│   ├── benchmark.json   # Latency/Throughput metrics
│   └── results.json     # Optuna search results
│
├── scripts/
│   ├── run_api.py       # Server launcher
│   ├── demo_*.py        # Reviewer demo scripts
│   ├── 8_optuna_*.py    # Hyperparameter search (7 methods)
│   └── 9_inference_*.py # Benchmark suite
│
├── tests/
│   ├── test_api.py      # Unit tests (25 passed)
│   └── locustfile.py    # Load testing
```

## 🔐 Security Features

- **Input Sanitization**: Strips HTML tags, script injection, and control characters.
- **Injection Detection**: Logs potential SQL injection or path traversal attempts.
- **PII Awareness**: Placeholders for PII detection logic.
- **DoS Protection**: Payload size limits (5KB) and request validation.

## 📈 Telemetry

The API exposes a `/metrics` endpoint showing:
- Uptime & Request Counts
- Error Rates & Cache Hit Rates
- Latency Percentiles (p50, p95, p99)
- Label Distribution
