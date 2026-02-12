# Master Project Blueprint: Legal Document Classification System

**Status:** Phase 1 (Data) done

---

## 1. Executive Summary & Decision Log
### The "Other" Class Strategy (Crucial Context)
*   **The Constraint:** The dataset contained only 6 "Other" documents vs 1,000 known documents.
*   **The Solution:** We **rejected** training on "Other". We moved 100% of "Other" to the **Test Set**.
*   **The Mechanism:** We are treating "Other" detection as an **Out-of-Distribution (OOD)** problem.
    *   We calibrated a **Confidence Threshold** using the Validation Set (5th percentile of correct predictions).
    *   **API Logic:** If `model.predict_proba() < Threshold`, the API *must* override the prediction to `"Other"`.

---

## 2. Phase 2: Model Shootout
train a tfidf model, and many other models legeraging gpu for training as well as hyperparameter finetuning. make sure all models are saved, all scripts for training are saved, and all results are saved. you should have 6 different model architectures and several hyperparameter configurations, training runs, and results reports for each model.

## 2. Phase 3: API Engineering Guidelines
**Objective:** Build a robust, production-grade REST API using `FastAPI`.

### A. Model Loading & Inference Architecture
*   **Artifact Handling:** The model is saved as a `.joblib` dictionary containing:
    1.  The Scikit-Learn Pipeline.
    2.  The calibrated `threshold` (float).
    *   *Requirement:* The API must load *both*. Do not hardcode the threshold.
*   **Inference Interface:** Create an abstract `ModelWrapper` class.
    *   Even though we are shipping TF-IDF now, the interface `predict(text) -> (label, confidence)` should be generic enough to hot-swap a Transformer model later without rewriting the API controller.

### B. Security & Validation (Middleware)
*   **PII Stripping:** The prompt explicitly mentions "production environment" concerns.
    *   *Requirement:* Integrate `presidio-analyzer` (or a lightweight regex fallback) to scan incoming text for PII (names, SSN) before logging or processing. Warn if high PII density is found.
*   **Payload Protection:**
    *   *Requirement:* Enforce strict request body limits (e.g., 2MB) to prevent "Zip Bomb" attacks or memory exhaustion.

### C. Performance & Caching
*   **Redis Layer:**
    *   *Requirement:* Implement a content-addressable cache.
    *   *Logic:* Hash the input text (SHA-256). Check Redis. If hit, return cached JSON. If miss, run inference and cache with a 24h TTL. This saves compute on duplicate document submissions.

---

## 3. Phase 4: Operational Excellence (QA & DevOps)
**Objective:** Prove the system works under pressure and package it efficiently.

### A. Testing Strategy
*   **Unit Tests:** Verify the API handles empty strings, massive strings, and invalid JSON gracefully (400/422 errors).
*   **Integration Tests:** Verify the Model Wrapper correctly loads the `.joblib` and applies the threshold logic (mock a low-confidence prediction and assert it returns "Other").
*   **Load Testing (Locust):**
    *   *Requirement:* Simulate 50-100 concurrent users.
    *   *Goal:* Demonstrate that the TF-IDF model maintains sub-50ms latency at scale (this is the key advantage of our model choice).

### B. Docker Optimization
*   **The "Slim" Advantage:**
    *   Since we rejected SetFit/Torch, we do **not** need the massive PyTorch binaries in our production image.
    *   *Requirement:* Build a lightweight Docker image (Python Slim). It should be <500MB. This is a major "Senior Engineer" win for deployment costs.

---

## 4. Phase 5: Final Deliverables & Reporting
**Objective:** Prepare the artifacts for the "Board Presentation."

### A. The Board Report (`BOARD_REPORT.md`)
*   **Executive Summary:** "We chose the lightweight model because it outperformed the heavy model in accuracy (94% vs 92%) and is 5000x faster to train."
*   **The OOD Defense:** Explain the "Confidence Floor" strategy clearly.
*   **Cost Analysis:** Highlight the cost savings of running a CPU-only Scikit-Learn model vs. a GPU-dependent Transformer model for millions of documents.

### B. Repository Cleanup
*   Ensure `requirements.txt` is updated (remove `torch`/`transformers` if we aren't using them in production, or move them to `requirements-dev.txt`).
*   Ensure `README.md` provides one-command instructions to:
    1.  `make train` (Runs the Baseline script).
    2.  `make run` (Starts the Docker container).
    3.  `make test` (Runs Pytest + Locust).