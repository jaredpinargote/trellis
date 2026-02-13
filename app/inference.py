import joblib
import time
import numpy as np
from pathlib import Path
from .schemas import PredictionResponse

# Import custom transformers so joblib can deserialize them
# from .transformers import DFRVectorizer  # noqa: F401 (Reverted to TF-IDF for stability)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "baseline.joblib"


class ModelWrapper:
    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5
        self.classes_ = []
        self.model_version = "unknown"
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")

        print(f"Loading model from {MODEL_PATH}...")
        artifact = joblib.load(MODEL_PATH)

        self.pipeline = artifact.get('pipeline')
        self.threshold = artifact.get('threshold', 0.5)
        self.model_version = artifact.get('model_version', 'unknown')

        if self.pipeline is None:
            raise ValueError("Joblib artifact missing 'pipeline' key")

        if hasattr(self.pipeline, 'classes_'):
            self.classes_ = self.pipeline.classes_

        print(f"Model loaded. Threshold: {self.threshold:.4f}, Version: {self.model_version}")

    def predict(self, document_text: str) -> PredictionResponse:
        start_time = time.time()

        probs = self.pipeline.predict_proba([document_text])[0]

        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = self.classes_[pred_idx]

        # Reject Option: OOD Detection
        is_ood = max_prob < self.threshold
        if is_ood:
            final_label = "other"
        else:
            final_label = pred_label

        duration = (time.time() - start_time) * 1000

        return PredictionResponse(
            message="Classification successful",
            label=final_label,
            confidence=round(float(max_prob), 4),
            is_ood=is_ood,
            processing_time_ms=round(duration, 2),
            model_version=self.model_version
        )


# Global singleton
model_wrapper = ModelWrapper()
