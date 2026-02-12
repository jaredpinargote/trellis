import pandas as pd
import numpy as np
import joblib
import time
import os
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Configuration
DATA_DIR = 'data/training'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost.joblib') # Saving as joblib dict for consistency
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Small, fast, effective

def load_data():
    """Load the split CSVs."""
    print("Loading data...")
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    # check for null or na, if found raise error, else return
    if train.isnull().values.any() or val.isnull().values.any() or test.isnull().values.any():
        raise ValueError("Null or NA values found in the data")
    return (
        train['text'], train['category'],
        val['text'], val['category'],
        test['text'], test['category']
    )

def train_xgboost():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    X_train_txt, y_train, X_val_txt, y_val, X_test_txt, y_test = load_data()
    
    # 2. Encode Text to Vectors (The Expensive Part)
    print(f"Loading SentenceTransformer: {EMBEDDING_MODEL_NAME}...")
    # We do NOT save this model in the joblib. The API will load it from huggingface cache.
    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("Encoding training data (this may take a moment)...")
    t0 = time.time()
    X_train_emb = encoder.encode(X_train_txt.tolist(), show_progress_bar=True)
    X_val_emb = encoder.encode(X_val_txt.tolist(), show_progress_bar=False)
    X_test_emb = encoder.encode(X_test_txt.tolist(), show_progress_bar=False)
    encoding_time = time.time() - t0
    print(f"Encoding finished in {encoding_time:.2f}s")

    # 3. Encode Labels (Strings -> Ints) for XGBoost
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    # Note: Test set has 'other', which might not be in LabelEncoder if we only fit on Train.
    # However, we only need y_test for the report. 
    # We will decode predictions back to strings later.

    # 4. Train XGBoost
    print("Training XGBoost Classifier...")
    t1 = time.time()
    
    clf = xgb.XGBClassifier(
        objective='multi:softprob', # Essential for probabilities
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        num_class=len(le.classes_)
    )
    
    clf.fit(X_train_emb, y_train_enc)
    training_time = time.time() - t1
    print(f"XGBoost training complete in {training_time:.2f}s")

    # 5. Calibration (Finding the Threshold)
    print("\n--- Calibration Phase (Validation Set) ---")
    val_probs = clf.predict_proba(X_val_emb)
    val_preds_idx = np.argmax(val_probs, axis=1)
    
    max_probs = np.max(val_probs, axis=1)
    
    # Filter: Look only at CORRECT predictions
    correct_indices = (val_preds_idx == y_val_enc)
    correct_confidences = max_probs[correct_indices]
    
    # Heuristic: 5th percentile of correct hits
    if len(correct_confidences) > 0:
        threshold = np.percentile(correct_confidences, 5)
    else:
        threshold = 0.5
        
    print(f"Validation Accuracy: {np.mean(correct_indices):.2%}")
    print(f"Confidence Threshold set to: {threshold:.4f}")

    # 6. Evaluation (Test Set + 'Other')
    print("\n--- Evaluation Phase (Test Set) ---")
    start_infer = time.time()
    
    test_probs = clf.predict_proba(X_test_emb)
    test_preds_idx = np.argmax(test_probs, axis=1)
    test_max_probs = np.max(test_probs, axis=1)
    
    # Convert Int predictions back to Strings
    raw_pred_labels = le.inverse_transform(test_preds_idx)
    
    # Apply Threshold Logic
    final_preds = []
    for pred, prob in zip(raw_pred_labels, test_max_probs):
        if prob < threshold:
            final_preds.append('other')
        else:
            final_preds.append(pred)
            
    inference_time_ms = (time.time() - start_infer) * 1000

    # Metrics
    unique_labels = sorted(list(set(y_test) | set(final_preds)))
    print(classification_report(y_test, final_preds, labels=unique_labels))

    # Check OOD specifically
    other_mask = (y_test == 'other')
    if np.sum(other_mask) > 0:
        other_caught = np.sum(np.array(final_preds)[other_mask] == 'other')
        print(f"OOD Detection: Caught {other_caught} / {np.sum(other_mask)} 'other' documents.")

    # 7. Save Artifacts
    # We MUST save the LabelEncoder, otherwise we can't decode the output later.
    artifact = {
        'model_type': 'xgboost-embeddings',
        'classifier': clf,
        'label_encoder': le,
        'threshold': threshold,
        'embedding_model_name': EMBEDDING_MODEL_NAME, # Store string, load later
        'metrics': {
            'encoding_time_s': encoding_time,
            'train_time_s': training_time,
            'inference_time_ms_per_batch': inference_time_ms
        }
    }
    
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_xgboost()