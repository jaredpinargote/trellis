import joblib
import logging
import time
import numpy as np
from pathlib import Path
from app.schemas import PredictionResponse
from app.core.config import Settings

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, settings: Settings):
        self.pipeline = None
        self.threshold = settings.MODEL_THRESHOLD
        self.classes_ = []
        self.model_version = "unknown"
        
        # Resolve model path from settings (can be overridden by env var)
        self.model_path = Path(settings.MODEL_PATH)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent.parent.parent / self.model_path
        
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")

        logger.info(f"Loading model from {self.model_path}...")
        # Shim for unpickling: 'app.transformers' might be missing if we moved it.
        # But we decided to keep app/transformers.py as a shim, so it should work.
        artifact = joblib.load(self.model_path)

        self.pipeline = artifact.get('pipeline')
        # Allow artifact to override config threshold if present? 
        # Or enforce config? Let's respect artifact for now as it's model-specific.
        self.threshold = artifact.get('threshold', self.threshold)
        self.model_version = artifact.get('model_version', 'unknown')

        if self.pipeline is None:
            raise ValueError("Joblib artifact missing 'pipeline' key")

        if hasattr(self.pipeline, 'classes_'):
            self.classes_ = self.pipeline.classes_

        logger.info(f"Model loaded. Threshold: {self.threshold:.4f}, Version: {self.model_version}")

    def predict(self, document_text: str) -> PredictionResponse:
        start_time = time.time()

        if self.pipeline is None:
             raise ValueError("Model pipeline is not loaded.")

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
