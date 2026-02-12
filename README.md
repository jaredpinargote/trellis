# Legal Document Classification System

This project implements a machine learning system to classify legal documents into 10 categories (e.g., Contract, Statute, Judgment) plus an "Other" category for out-of-distribution detection.

## Quick Start

### Prerequisites
*   Python 3.10+
*   Docker (Optional)

### Installation
1.  Create a virtual environment:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Training
To reproduce the baseline model:
```bash
python scripts/4_train_baseline.py
```
This saves the model and threshold to `models/baseline.joblib`.

### Running the API
Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Or use the helper script:
```bash
python scripts/run_api.py
```
*   **Swagger UI:** http://127.0.0.1:8000/docs
*   **Health Check:** http://127.0.0.1:8000/health

### Testing
Run unit tests:
```bash
pytest
```
Run load tests:
```bash
locust -f tests/locustfile.py --headless -u 10 -r 2 -t 10s --host http://127.0.0.1:8000
```

## Docker Deployment
Build the lightweight container:
```bash
docker build -t legal-classifier:latest .
```
Run the container:
```bash
docker run -p 8000:8000 legal-classifier:latest
```

## Architecture
*   **Models:** Comparison of TF-IDF+SVM vs SetFit vs XGBoost. Selected Baseline for speed/accuracy.
*   **Security:** PII detection (Presidio) + Payload validation.
*   **Caching:** Redis support (auto-fallback if not available).
