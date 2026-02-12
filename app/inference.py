import joblib
import os
import time
import numpy as np
from pathlib import Path
from .schemas import PredictionResponse

# Correct path relative to this file
# This assumes running from project root or similar. But safer to use absolute path logic relative to THIS file.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "baseline.joblib"

class ModelWrapper:
    def __init__(self):
        self.pipeline = None
        self.threshold = 0.5
        self.classes_ = []
        self._load_model()

    def _load_model(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
        
        print(f"Loading model from {MODEL_PATH}...")
        artifact = joblib.load(MODEL_PATH)
        
        self.pipeline = artifact.get('pipeline')
        self.threshold = artifact.get('threshold', 0.5)
        
        if self.pipeline is None:
             raise ValueError("Joblib artifact missing 'pipeline' key")
             
        # Cache classes for faster lookup if possible, though pipeline has them
        if hasattr(self.pipeline, 'classes_'):
            self.classes_ = self.pipeline.classes_
            
        print(f"Model loaded successfully. Confidence Threshold: {self.threshold}")

    def predict(self, text: str) -> PredictionResponse:
        start_time = time.time()
        
        # Get Probabilities
        # predict_proba returns [n_samples, n_classes], we have 1 sample
        probs = self.pipeline.predict_proba([text])[0]
        
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = self.classes_[pred_idx]
        
        # Applying the "Reject Option" (OOD Detection)
        if max_prob < self.threshold:
            final_label = "other"
            # We keep the confidence score as is, or maybe clamp it? 
            # Per plan, we just return "Other" if below threshold.
            # The confidence is still the max_prob of the *incorrect* class, which is fine, 
            # or we could return the threshold as confidence? 
            # Usually returning the raw max_prob is honest: "I was 40% sure it was Sport, which is < 60% threshold, so I say Other".
        else:
            final_label = pred_label

        duration = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            category=final_label,
            confidence=float(max_prob),
            processing_time_ms=duration,
            model_version="baseline_v1_tfidf"
        )

# Global singleton for import
model_wrapper = ModelWrapper()
