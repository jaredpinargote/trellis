"""
Retrain Production Model: TF-IDF (Augmented Data)
=================================================
Trains the TF-IDF model on the augmented dataset (original + paragraphs + sentences)
to resolve the length mismatch issue.

Training: `data/training/train_augmented.csv`
Validation: `data/training/val.csv` (Original)
Test: `data/training/test.csv` (Original)

This ensures strict evaluation: we train on short text to learn patterns,
but we validate on unseen original documents to ensure no regression.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = 'data/training'
MODEL_DIR = 'models'

def main():
    print("=" * 70)
    print("RETRAINING PRODUCTION MODEL: TF-IDF (Augmented Data)")
    print("=" * 70)
    
    # Load data
    train_aug_path = os.path.join(DATA_DIR, 'train_augmented.csv')
    if not os.path.exists(train_aug_path):
        print("Error: Augmented data not found.")
        sys.exit(1)
        
    train = pd.read_csv(train_aug_path)
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    X_train, y_train = train['text'].astype(str), train['category']
    X_val, y_val = val['text'].astype(str), val['category']
    X_test, y_test = test['text'].astype(str), test['category']

    print(f"Training Samples: {len(X_train)} (Augmented)")
    print(f"Validation Samples: {len(X_val)} (Original)")

    # Build Pipeline (Using Optuna Trial #3 Params, robust)
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000, # Increased from 5000 due to augmented corpus
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', SGDClassifier(
            loss='modified_huber',
            alpha=4.233e-05,
            penalty='l2',
            random_state=42,
            max_iter=1000,
            tol=1e-3
        ))
    ])

    # Train
    print("Training...")
    pipeline.fit(X_train, y_train)

    # Calibrate Threshold (5th percentile on val set)
    # Note: Val set is long text. Threshold might be high.
    # But model should now be confident on short text too.
    val_probs = pipeline.predict_proba(X_val)
    val_preds = pipeline.predict(X_val)
    correct_mask = (val_preds == y_val)
    correct_confs = np.max(val_probs, axis=1)[correct_mask]
    threshold = float(np.percentile(correct_confs, 5)) if len(correct_confs) > 0 else 0.5
    print(f"OOD Threshold (5th percentile): {threshold:.4f}")

    # Evaluate on Test (Long Text)
    test_probs = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(test_probs, axis=1)]
    max_probs = np.max(test_probs, axis=1)
    final_preds = ['other' if p < threshold else r for r, p in zip(raw_preds, max_probs)]
    
    test_f1 = f1_score(y_test, final_preds, average='weighted')
    other_mask = (y_test == 'other')
    ood_recall = np.mean(np.array(final_preds)[other_mask] == 'other')
    
    print(f"Test F1 (Long Text Stability): {test_f1:.4f}")
    print(f"OOD Recall: {ood_recall:.2%}")

    # Save
    artifact = {
        'pipeline': pipeline,
        'threshold': threshold,
        'model_version': 'optuna_tfidf_augmented_v2',
        'metrics': {'test_f1': test_f1, 'ood_recall': ood_recall}
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, 'baseline.joblib'))
    print("Saved to models/baseline.joblib")

if __name__ == "__main__":
    main()
