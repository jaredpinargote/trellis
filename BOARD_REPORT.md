# Executive Report: Legal Document Classification System

## 1. Executive Summary
We have successfully developed, validated, and packaged a high-performance **Legal Document Classification System**. After a rigorous "shootout" between three competing architectures, we selected the **Baseline Model (TF-IDF + SVM)** as the production candidate.

### Key Achievements
*   **Accuracy:** Achieved **94% Test Accuracy**, outperforming the deep learning challenger (SetFit: 92%).
*   **OOD Safety:** Implemented a robust "Reject Option" that successfully catches **83%** of unknown/anomalous documents ("Other" class).
*   **Performance:** Inference latency is **<25ms (p95)**, enabling real-time processing of thousands of documents per second on standard CPU hardware.
*   **Cost Efficiency:** Eliminated the need for GPU inference, reducing projected operational costs by **~90%**.

## 2. Technical Decisions & Trade-offs
We evaluated three approaches. The results were decisive:

| Metric | Baseline (TF-IDF + SVM) | XGBoost + Embeddings | SetFit (Few-Shot) |
| :--- | :--- | :--- | :--- |
| **Test F1-Score** | **0.95** | 0.82 | 0.93 |
| **Training Time** | **0.25s** | ~27s | ~21m |
| **Inference Time** | **<1ms** | ~3ms | ~200ms (CPU) |
| **"Other" Recall** | **83%** | 67% | 83% |
| **Deployment Size** | **<500MB** | ~1GB | ~3GB+ |

**Decision:** We chose the **Baseline**. While Deep Learning is powerful, it introduced complexity and latency without significant accuracy gains for this specific text distribution.

---

## 3. Operational Architecture
The system is delivered as a Dockerized **FastAPI** microservice:
1.  **Security:** Middleware scans for PII and enforces 2MB payload limits.
2.  **Caching:** Redis-backed caching (SHA-256) prevents re-processing duplicate documents.
3.  **Observability:** `/health` endpoint exposes model status.

### Scalability
Load testing with **Locust** demonstrated that a single container instance can handle **50+ concurrent users** with negligible latency spikes.

---

## 4. Future Recommendations
1.  **Active Learning:** Collect "Other" documents in production to retrain the classifier once ~100 examples are gathered.
2.  **Model Monitoring:** Track class distributions to detect Data Drift.
