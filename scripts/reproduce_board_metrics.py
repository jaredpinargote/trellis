import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import classification_report, f1_score

# Defines
MODEL_PATH = 'models/baseline.joblib'
TEST_DATA_PATH = 'data/training/test.csv'

def reproduce_metrics():
    print("--- Reproducing Board Metrics ---")
    
    # 1. Load Model Artifact
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model artifact not found at {MODEL_PATH}")
        sys.exit(1)
        
    print(f"Loading model from {MODEL_PATH}...")
    try:
        artifact = joblib.load(MODEL_PATH)
        pipeline = artifact['pipeline']
        threshold = artifact['threshold']
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    print(f"Model loaded. Threshold: {threshold:.4f}")

    # 2. Load Test Data
    if not os.path.exists(TEST_DATA_PATH):
        print(f"Error: Test data not found at {TEST_DATA_PATH}")
        sys.exit(1)
        
    print(f"Loading test data from {TEST_DATA_PATH}...")
    try:
        df_test = pd.read_csv(TEST_DATA_PATH)
        # Ensure no NaNs
        if df_test.isnull().values.any():
            print("Warning: NaNs found in test data, dropping...")
            df_test = df_test.dropna()
            
        X_test = df_test['text']
        y_test = df_test['category']
    except Exception as e:
        print(f"Failed to load test data: {e}")
        sys.exit(1)

    print(f"Loaded {len(X_test)} test samples.")

    # 3. Run Inference
    print("Running inference...")
    # Get probabilities
    probas = pipeline.predict_proba(X_test)
    
    # Get raw predictions (argmax)
    raw_preds = pipeline.classes_[np.argmax(probas, axis=1)]
    max_probs = np.max(probas, axis=1)

    # Apply Threshold Logic for 'other' class
    final_preds = []
    for pred, prob in zip(raw_preds, max_probs):
        if prob < threshold:
            final_preds.append('other')
        else:
            final_preds.append(pred)

    # 4. Generate Report
    print("\n--- Classification Report ---")
    unique_labels = sorted(list(set(y_test) | set(final_preds)))
    report = classification_report(y_test, final_preds, labels=unique_labels)
    print(report)

    # 5. Verify "94%" Claim
    macro_f1 = f1_score(y_test, final_preds, average='macro')
    weighted_f1 = f1_score(y_test, final_preds, average='weighted')
    
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    
    if weighted_f1 >= 0.93: # Allowing slight variance
        print("\n✅ PASSED: Metrics substantiate the Board Report claim (~94%).")
    else:
        print(f"\n⚠️ WARNING: Metrics are lower than claimed (Expected ~0.94, Got {weighted_f1:.4f}).")

if __name__ == "__main__":
    reproduce_metrics()
