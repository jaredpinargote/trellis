import pandas as pd
import numpy as np
import joblib
import os
import sys
import time

# Defines
MODEL_PATH = 'models/baseline.joblib'
TEST_DATA_PATH = 'data/training/test.csv'

def simulate_drift():
    print("--- Simulating Data Drift & Monitoring ---")
    
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        sys.exit(1)
    
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact['pipeline']
    
    # 2. Load Data
    df = pd.read_csv(TEST_DATA_PATH).dropna()
    
    # Separate Normal vs OOD (Other)
    normal_data = df[df['category'] != 'other']
    ood_data = df[df['category'] == 'other']
    
    print(f"Loaded Data: {len(normal_data)} Normal samples, {len(ood_data)} OOD samples.")
    
    if len(ood_data) < 10:
        print("Warning: Not enough OOD samples to simulate significant drift. Using subset of normal data as 'drifted' (this is just a simulation).")
        # In a real scenario we'd need actual OOD data.
        # For demo, let's just use the normal data but maybe jitter it? 
        # Actually, let's just proceed, even if small.
    
    # 3. Simulate "Normal" Traffic
    print("\nPhase 1: Normal Traffic Baseline")
    # Take a batch of normal data
    batch_size = 50
    normal_batch = normal_data.sample(n=min(len(normal_data), batch_size), random_state=42)
    
    probas = pipeline.predict_proba(normal_batch['text'])
    max_probs = np.max(probas, axis=1)
    avg_conf = np.mean(max_probs)
    
    print(f"Baseline Average Confidence: {avg_conf:.4f}")
    baseline = avg_conf

    # 4. Simulate "Drift" Traffic (Sudden appearance of OOD topics)
    print("\nPhase 2: Drift Injection (Simulating 100% OOD Traffic)")
    
    if len(ood_data) > 0:
        # Replicate OOD data if we don't have enough to fill a batch
        drift_batch = pd.concat([ood_data] * (batch_size // len(ood_data) + 1)).head(batch_size)
        
        probas_drift = pipeline.predict_proba(drift_batch['text'])
        drift_probs = np.max(probas_drift, axis=1)
        drift_conf = np.mean(drift_probs)
        
        print(f"Drifted Average Confidence: {drift_conf:.4f}")
        
        # 5. Alert Logic
        drop = baseline - drift_conf
        print(f"Confidence Drop: {drop:.4f}")
        
        if drop > 0.15: # Threshold for alert
            print("\n🚨 ALERT: Significant Concept Drift Detected!")
            print("Action: Trigger retraining or human review.")
        else:
            print("\n✅ Drift within acceptable limits (or model is overconfident on OOD).")
    else:
        print("Skipping Phase 2: No 'other' class data found in test set to simulate drift.")

if __name__ == "__main__":
    simulate_drift()
