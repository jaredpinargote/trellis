import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

# Defines
MODEL_PATH = 'models/baseline.joblib'
TEST_DATA_PATH = 'data/training/test.csv'
OUTPUT_PLOT = 'threshold_analysis.png'

def analyze_sensitivity():
    print("--- Analyzing OOD Threshold Sensitivity ---")
    
    # 1. Load Model & Data
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TEST_DATA_PATH):
        print("Error: Model or Data not found.")
        sys.exit(1)
        
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact['pipeline']
    current_threshold = artifact['threshold']
    
    df_test = pd.read_csv(TEST_DATA_PATH).dropna()
    X_test = df_test['text']
    y_test = df_test['category']
    
    print(f"Loaded {len(X_test)} samples. Current Threshold: {current_threshold:.4f}")

    # 2. Get Probabilities
    print("Running inference to get base probabilities...")
    probas = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(probas, axis=1)]
    max_probs = np.max(probas, axis=1)

    # 3. Iterate Thresholds
    thresholds = np.linspace(0.1, 0.95, 18)
    results = []
    
    print("\n--- Sensitivity Table (Focus: 'other' class) ---")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'OOD Count':<10}")
    print("-" * 65)

    for t in thresholds:
        # Apply Threshold Logic
        final_preds = []
        ood_count = 0
        for pred, prob in zip(raw_preds, max_probs):
            if prob < t:
                final_preds.append('other')
                ood_count += 1
            else:
                final_preds.append(pred)
        
        # Calculate Metrics for 'other' class specifically
        # We need to handle the case where 'other' might not be in y_test if we are running partial data
        # but here we assume test.csv has 'other'
        
        # We start by getting metrics for ALL classes
        labels = sorted(list(set(y_test) | set(final_preds)))
        p, r, f, _ = precision_recall_fscore_support(y_test, final_preds, labels=labels, zero_division=0)
        
        # Find index of 'other'
        if 'other' in labels:
            idx = labels.index('other')
            p_other = p[idx]
            r_other = r[idx]
            f_other = f[idx]
        else:
            p_other, r_other, f_other = 0.0, 0.0, 0.0

        marker = "*" if abs(t - current_threshold) < 0.03 else ""
        print(f"{t:<10.2f} | {p_other:<10.4f} | {r_other:<10.4f} | {f_other:<10.4f} | {ood_count:<10} {marker}")
        
        results.append({
            'threshold': t,
            'precision': p_other,
            'recall': r_other,
            'f1': f_other
        })

    # 4. Optional: Plot
    try:
        ts = [r['threshold'] for r in results]
        ps = [r['precision'] for r in results]
        rs = [r['recall'] for r in results]
        fs = [r['f1'] for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.plot(ts, ps, label='Precision (Other)', marker='o')
        plt.plot(ts, rs, label='Recall (Other)', marker='x')
        plt.plot(ts, fs, label='F1 (Other)', marker='s')
        plt.axvline(x=current_threshold, color='r', linestyle='--', label=f'Current ({current_threshold:.2f})')
        
        plt.title('OOD Detection Sensitivity Analysis')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(OUTPUT_PLOT)
        print(f"\nPlot saved to {OUTPUT_PLOT}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

if __name__ == "__main__":
    analyze_sensitivity()
