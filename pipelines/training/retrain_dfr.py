"""
Retrain Production Model: DFR (Divergence from Randomness)
===========================================================
Loads best DFR config from cross_method_benchmark.json,
trains on full training set, calibrates OOD threshold, saves artifact.

Usage: python scripts/retrain_dfr.py
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib

# Import custom transformers FROM THE APP MODULE so pickle references the correct path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from app.transformers import DFRVectorizer

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

DATA_DIR = 'data/training'
MODEL_DIR = 'models'


def main():
    # ── Load best DFR config ───────────────────────────────────────
    with open(os.path.join(MODEL_DIR, 'cross_method_benchmark.json')) as f:
        bench = json.load(f)

    dfr = bench['methods']['dfr']
    params = dfr['params']
    print("=" * 70)
    print("RETRAINING PRODUCTION MODEL: DFR (Divergence from Randomness)")
    print("=" * 70)
    print(f"  Source: cross_method_benchmark.json (trial #{dfr['trial']})")
    print(f"  Val F1: {dfr['val_f1']:.4f} | Test F1: {dfr['test_f1']:.4f}")
    print(f"  Params: {json.dumps(params, indent=4)}")

    # ── Load data ──────────────────────────────────────────────────
    # Check for augmented data first (Critical for short-text performance)
    train_path = os.path.join(DATA_DIR, 'train_augmented.csv')
    if not os.path.exists(train_path):
        print("Warning: Augmented data not found. Falling back to original train.csv")
        train_path = os.path.join(DATA_DIR, 'train.csv')
    else:
        print(f"Loading augmented data from {train_path}...")
    
    train = pd.read_csv(train_path)
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

    X_train, y_train = train['text'], train['category']
    X_val, y_val = val['text'], val['category']
    X_test, y_test = test['text'], test['category']

    print(f"\n  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # ── Build pipeline ─────────────────────────────────────────────
    ngram_range = (1, params.get('custom_ngram_max', 1))
    pipeline = Pipeline([
        ('vectorizer', DFRVectorizer(
            c=params.get('dfr_c', 1.0),
            max_features=params.get('custom_max_features', 5000),
            ngram_range=ngram_range,
        )),
        ('clf', SGDClassifier(
            loss=params.get('sgd_loss', 'log_loss'),
            alpha=params.get('sgd_alpha', 1e-4),
            penalty='l2',
            random_state=42,
            max_iter=1000,
            tol=1e-3,
        ))
    ])

    # ── Train ──────────────────────────────────────────────────────
    print("\n  Training...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.1f}s")

    # ── Calibrate OOD threshold ────────────────────────────────────
    # ── Calibrate OOD threshold ────────────────────────────────────
    # Use a sample of augmented training data for calibration to include short texts
    # (val.csv only has long texts, which skews threshold too high)
    cal_subset = train.sample(min(2000, len(train)), random_state=42)
    cal_probs = pipeline.predict_proba(cal_subset['text'])
    cal_preds = pipeline.predict(cal_subset['text'])
    
    correct_mask = (cal_preds == cal_subset['category'])
    correct_confs = np.max(cal_probs, axis=1)[correct_mask]
    threshold = float(np.percentile(correct_confs, 1)) if len(correct_confs) > 0 else 0.5
    print(f"  OOD threshold: {threshold:.4f}")

    # ── Evaluate on test ───────────────────────────────────────────
    test_probs = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(test_probs, axis=1)]
    max_probs = np.max(test_probs, axis=1)
    final_preds = ['other' if p < threshold else r for r, p in zip(raw_preds, max_probs)]

    labels = sorted(set(y_test) | set(final_preds))
    test_f1 = f1_score(y_test, final_preds, labels=labels, average='weighted', zero_division=0)
    report = classification_report(y_test, final_preds, labels=labels, output_dict=True, zero_division=0)

    other_mask = (y_test == 'other')
    ood_recall = float(np.mean(np.array(final_preds)[other_mask] == 'other')) if np.sum(other_mask) > 0 else 0.0

    print(f"\n  Test F1 (weighted): {test_f1:.4f}")
    print(f"  OOD Recall: {ood_recall:.2%}")
    print(f"\n  Per-class F1:")
    for cls in sorted(report.keys()):
        if cls in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        f1 = report[cls]['f1-score']
        print(f"    {cls:20s} {f1:.4f}")

    # ── Save artifact ──────────────────────────────────────────────
    artifact = {
        'pipeline': pipeline,
        'threshold': threshold,
        'model_version': 'optuna_dfr_v1',
        'metrics': {
            'test_f1_weighted': test_f1,
            'ood_recall': ood_recall,
            'classification_report': report
        }
    }
    model_path = os.path.join(MODEL_DIR, 'baseline.joblib')
    joblib.dump(artifact, model_path)
    print(f"\n  Model saved to {model_path}")
    print(f"  Artifact size: {os.path.getsize(model_path)/1024:.0f} KB")

    print(f"\n{'='*70}")
    print(f"  DONE. Production model: DFR | F1={test_f1:.4f} | OOD={ood_recall:.2%}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
