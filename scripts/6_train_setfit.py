import pandas as pd
import numpy as np
import os
import time
import joblib
import torch
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss # <--- THE MISSING IMPORT
from datasets import Dataset
from sklearn.metrics import classification_report

# Configuration
DATA_DIR = 'data/training'
MODEL_DIR = 'models/setfit_v1'
METADATA_PATH = os.path.join(MODEL_DIR, 'metadata.joblib')

# Using MiniLM for CPU speed
BACKBONE = 'sentence-transformers/all-MiniLM-L6-v2' 

def load_data():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val_df = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    
    # HF Dataset conversion
    train_ds = Dataset.from_pandas(train_df[['text', 'category']].rename(columns={'category': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['text', 'category']].rename(columns={'category': 'label'}))
    
    return train_ds, val_ds, test_df

def train_setfit():
    train_ds, val_ds, test_df = load_data()
    
    print(f"Initializing SetFit with backbone: {BACKBONE}...")
    model = SetFitModel.from_pretrained(BACKBONE)
    
    print("Starting SetFit Training (Contrastive Fine-Tuning)...")
    print("NOTE: On CPU, this might take 2-5 minutes. Please wait.")
    start_train = time.time()
    
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        loss_class=CosineSimilarityLoss, # <--- FIXED: Passing the Class, not a string
        metric="accuracy",
        batch_size=8,        
        num_iterations=2,    # Low iterations for speed
        num_epochs=1,        
        column_mapping={"text": "text", "label": "label"}
    )
    
    # Train
    trainer.train()
    
    train_time = time.time() - start_train
    print(f"Training complete in {train_time:.2f}s")
    
    # --- Calibration & Eval ---
    print("\n--- Calibration Phase (Validation Set) ---")
    val_texts = val_ds['text']
    val_labels = val_ds['label']
    
    # Get probabilities
    val_probs = model.predict_proba(val_texts)
    if isinstance(val_probs, torch.Tensor):
        val_probs = val_probs.cpu().numpy()
        
    val_max_probs = np.max(val_probs, axis=1)
    val_preds = model.predict(val_texts)
    
    # Threshold Logic
    correct_indices = [i for i, (pred, true) in enumerate(zip(val_preds, val_labels)) if pred == true]
    
    if len(correct_indices) > 0:
        correct_confidences = val_max_probs[correct_indices]
        threshold = np.percentile(correct_confidences, 5)
    else:
        threshold = 0.5
        
    print(f"Validation Accuracy: {len(correct_indices)/len(val_labels):.2%}")
    print(f"Confidence Threshold set to: {threshold:.4f}")

    print("\n--- Evaluation Phase (Test Set) ---")
    start_infer = time.time()
    
    test_texts = test_df['text'].tolist()
    test_probs = model.predict_proba(test_texts)
    if isinstance(test_probs, torch.Tensor):
        test_probs = test_probs.cpu().numpy()
        
    test_max_probs = np.max(test_probs, axis=1)
    raw_preds = model.predict(test_texts)
    
    # Apply Threshold
    final_preds = []
    for pred, prob in zip(raw_preds, test_max_probs):
        if prob < threshold:
            final_preds.append('other')
        else:
            final_preds.append(pred)
            
    inference_time_ms = (time.time() - start_infer) * 1000
    
    # Metrics
    y_true = test_df['category'].tolist()
    unique_labels = sorted(list(set(y_true) | set(final_preds)))
    
    print(classification_report(y_true, final_preds, labels=unique_labels))
    
    # OOD Check
    other_mask = np.array(y_true) == 'other'
    if np.sum(other_mask) > 0:
        other_caught = np.sum(np.array(final_preds)[other_mask] == 'other')
        print(f"OOD Detection: Caught {other_caught} / {np.sum(other_mask)} 'other' documents.")

    # Save
    model.save_pretrained(MODEL_DIR)
    metadata = {
        'threshold': threshold,
        'metrics': {
            'train_time_s': train_time,
            'inference_time_ms_total': inference_time_ms
        }
    }
    joblib.dump(metadata, METADATA_PATH)
    print(f"\nModel saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_setfit()