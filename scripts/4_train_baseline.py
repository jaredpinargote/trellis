import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

# Configuration
DATA_DIR = 'data/training'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.joblib')

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

def train_baseline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # 2. Build Pipeline
    # Tfidf: Standard english stop words, max 10k features to keep it lightweight
    # SGDClassifier: loss='modified_huber' enables predict_proba()
    print("Training Baseline Model (TF-IDF + SVM)...")
    start_train = time.time()
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
        ('clf', SGDClassifier(
            loss='modified_huber', 
            penalty='l2',
            alpha=1e-4, 
            random_state=42, 
            max_iter=1000, 
            tol=1e-3,
            n_jobs=-1
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_train
    print(f"Training complete in {train_time:.2f}s")

    # 3. Calibration (Finding the Threshold)
    print("\n--- Calibration Phase (Validation Set) ---")
    # Predict probabilities on Validation set (Known classes only)
    val_probs = pipeline.predict_proba(X_val)
    val_preds = pipeline.predict(X_val)
    
    # Get the confidence score of the CHOSEN class
    max_probs = np.max(val_probs, axis=1)
    
    # Filter: Look only at CORRECT predictions to set the "Safety Floor"
    correct_indices = (val_preds == y_val)
    correct_confidences = max_probs[correct_indices]
    
    # Heuristic: Set threshold at the 5th percentile of correct predictions
    # Meaning: "95% of the time when we are right, we are at least THIS confident."
    if len(correct_confidences) > 0:
        threshold = np.percentile(correct_confidences, 5)
    else:
        print("WARNING: Model got 0 validation samples correct. Defaulting threshold to 0.5")
        threshold = 0.5
        
    print(f"Validation Accuracy: {np.mean(correct_indices):.2%}")
    print(f"Confidence Threshold set to: {threshold:.4f} (5th percentile of correct hits)")

    # 4. Evaluation (The OOD Test)
    print("\n--- Evaluation Phase (Test Set + 'Other' Class) ---")
    start_infer = time.time()
    
    # Raw predictions (10 classes)
    test_probs = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(test_probs, axis=1)]
    max_test_probs = np.max(test_probs, axis=1)
    
    # Apply Threshold Logic
    final_preds = []
    for pred, prob in zip(raw_preds, max_test_probs):
        if prob < threshold:
            final_preds.append('other') # The logic we needed!
        else:
            final_preds.append(pred)
            
    inference_time = (time.time() - start_infer) * 1000 # ms
    
    # 5. Metrics
    # We must explicitly list all labels including 'other' for the report
    unique_labels = sorted(list(set(y_test) | set(final_preds)))
    
    print(classification_report(y_test, final_preds, labels=unique_labels))
    
    # specifically check recall on 'other'
    other_mask = (y_test == 'other')
    if np.sum(other_mask) > 0:
        other_caught = np.sum(np.array(final_preds)[other_mask] == 'other')
        print(f"OOD Detection: Caught {other_caught} / {np.sum(other_mask)} 'other' documents.")
    
    # 6. Save Artifact
    # We save the pipeline AND the threshold. The API needs both.
    artifact = {
        'pipeline': pipeline,
        'threshold': threshold,
        'metrics': {
            'train_time_sec': train_time,
            'inference_time_ms': inference_time / len(X_test)
        }
    }
    
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nModel and metadata saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_baseline()