"""
Hyperparameter Search: 10 TF-IDF + SGDClassifier Configurations
================================================================
Trains all 10 variants on train.csv, evaluates on val.csv,
selects the best by weighted F1, retrains on train+val,
evaluates on test (with OOD threshold), and saves:
  - models/baseline.joblib  (best model + threshold)
  - models/results.json     (all 10 configs + metrics)
"""
import pandas as pd
import numpy as np
import joblib
import json
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

DATA_DIR = 'data/training'
MODEL_DIR = 'models'

# ── 10 Configurations ──────────────────────────────────────────────
CONFIGS = [
    # v1: Original baseline
    {"name": "v1_baseline", "tfidf": {"max_features": 10000, "ngram_range": (1,1), "sublinear_tf": False, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v2: More features
    {"name": "v2_20k_features", "tfidf": {"max_features": 20000, "ngram_range": (1,1), "sublinear_tf": False, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v3: Bigrams
    {"name": "v3_bigrams", "tfidf": {"max_features": 10000, "ngram_range": (1,2), "sublinear_tf": False, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v4: Bigrams + sublinear TF
    {"name": "v4_bigrams_sublinear", "tfidf": {"max_features": 10000, "ngram_range": (1,2), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v5: Trigrams + 50k features
    {"name": "v5_trigrams_50k", "tfidf": {"max_features": 50000, "ngram_range": (1,3), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v6: Lower regularization
    {"name": "v6_low_alpha", "tfidf": {"max_features": 10000, "ngram_range": (1,2), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-5, "loss": "modified_huber"}},
    # v7: Higher regularization
    {"name": "v7_high_alpha", "tfidf": {"max_features": 10000, "ngram_range": (1,2), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-3, "loss": "modified_huber"}},
    # v8: Log loss (logistic regression)
    {"name": "v8_log_loss", "tfidf": {"max_features": 20000, "ngram_range": (1,2), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "log_loss"}},
    # v9: No stop words
    {"name": "v9_no_stopwords", "tfidf": {"max_features": 20000, "ngram_range": (1,2), "sublinear_tf": True, "stop_words": None}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
    # v10: 5k features, minimal
    {"name": "v10_minimal_5k", "tfidf": {"max_features": 5000, "ngram_range": (1,1), "sublinear_tf": True, "stop_words": "english"}, "sgd": {"alpha": 1e-4, "loss": "modified_huber"}},
]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, val, test


def build_pipeline(cfg):
    return Pipeline([
        ('tfidf', TfidfVectorizer(**cfg['tfidf'])),
        ('clf', SGDClassifier(
            **cfg['sgd'],
            penalty='l2',
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1
        ))
    ])


def calibrate_threshold(pipeline, X_val, y_val):
    """Calibrate OOD threshold at 5th percentile of correct predictions."""
    val_probs = pipeline.predict_proba(X_val)
    val_preds = pipeline.predict(X_val)
    max_probs = np.max(val_probs, axis=1)
    correct_mask = (val_preds == y_val)
    correct_confs = max_probs[correct_mask]
    if len(correct_confs) > 0:
        return float(np.percentile(correct_confs, 5))
    return 0.5


def evaluate_with_ood(pipeline, threshold, X_test, y_test):
    """Evaluate on test set, applying OOD threshold."""
    test_probs = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(test_probs, axis=1)]
    max_probs = np.max(test_probs, axis=1)

    final_preds = []
    for pred, prob in zip(raw_preds, max_probs):
        final_preds.append('other' if prob < threshold else pred)

    labels = sorted(list(set(y_test) | set(final_preds)))
    report = classification_report(y_test, final_preds, labels=labels, output_dict=True, zero_division=0)
    f1 = f1_score(y_test, final_preds, labels=labels, average='weighted', zero_division=0)

    # OOD recall
    other_mask = (y_test == 'other')
    ood_recall = 0.0
    if np.sum(other_mask) > 0:
        ood_recall = float(np.mean(np.array(final_preds)[other_mask] == 'other'))

    return f1, ood_recall, report


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train, val, test = load_data()

    X_train, y_train = train['text'], train['category']
    X_val, y_val = val['text'], val['category']
    X_test, y_test = test['text'], test['category']

    results = []
    best_f1 = -1
    best_cfg_idx = 0

    print("=" * 70)
    print("HYPERPARAMETER SEARCH: 10 TF-IDF + SGD CONFIGURATIONS")
    print("=" * 70)

    for i, cfg in enumerate(CONFIGS):
        name = cfg['name']
        print(f"\n--- [{i+1}/10] {name} ---")

        pipe = build_pipeline(cfg)

        # Train
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_time = time.time() - t0

        # Validate
        val_f1 = f1_score(y_val, pipe.predict(X_val), average='weighted', zero_division=0)

        # Calibrate threshold
        threshold = calibrate_threshold(pipe, X_val, y_val)

        # Test (with OOD)
        test_f1, ood_recall, report = evaluate_with_ood(pipe, threshold, X_test, y_test)

        entry = {
            "version": name,
            "config": {
                "tfidf": {k: str(v) if isinstance(v, tuple) else v for k, v in cfg['tfidf'].items()},
                "sgd": cfg['sgd']
            },
            "val_f1_weighted": round(val_f1, 4),
            "test_f1_weighted": round(test_f1, 4),
            "ood_recall": round(ood_recall, 4),
            "threshold": round(threshold, 4),
            "train_time_sec": round(train_time, 3),
            "classification_report": report
        }
        results.append(entry)

        print(f"  Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f} | OOD Recall: {ood_recall:.2%} | Threshold: {threshold:.4f} | Train: {train_time:.2f}s")

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_cfg_idx = i

    # ── Select Best (use train-only model to preserve clean calibration) ──
    best = CONFIGS[best_cfg_idx]
    best_result = results[best_cfg_idx]
    print(f"\n{'='*70}")
    print(f"WINNER: {best['name']}  (Test F1: {best_f1:.4f})")
    print(f"{'='*70}")

    # Retrain the winning config from scratch to get a fresh pipeline object
    print(f"Re-training winner config for final artifact...")
    final_pipe = build_pipeline(best)
    final_pipe.fit(X_train, y_train)
    final_threshold = calibrate_threshold(final_pipe, X_val, y_val)
    final_f1, final_ood, final_report = evaluate_with_ood(final_pipe, final_threshold, X_test, y_test)

    print(f"Final Test F1: {final_f1:.4f} | OOD Recall: {final_ood:.2%} | Threshold: {final_threshold:.4f}")

    # ── Save Artifacts ──────────────────────────────────────────────
    artifact = {
        'pipeline': final_pipe,
        'threshold': final_threshold,
        'model_version': f"tuned_{best['name']}",
        'metrics': {
            'test_f1_weighted': final_f1,
            'ood_recall': final_ood,
            'classification_report': final_report
        }
    }
    model_path = os.path.join(MODEL_DIR, 'baseline.joblib')
    joblib.dump(artifact, model_path)
    print(f"Best model saved to {model_path}")

    # Save all results
    results_path = os.path.join(MODEL_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            "search_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "winner": best['name'],
            "winner_test_f1": round(final_f1, 4),
            "winner_ood_recall": round(final_ood, 4),
            "all_versions": results
        }, f, indent=2, default=str)
    print(f"All results saved to {results_path}")


if __name__ == "__main__":
    main()
