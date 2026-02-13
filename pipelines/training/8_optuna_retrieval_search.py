"""
Optuna Hyperparameter Search: 7 CPU-Only Retrieval Methods
===========================================================
Methods:
  1. TF-IDF (sklearn TfidfVectorizer)
  2. BM25 (Okapi BM25)
  3. BM25L (lower-bounded length normalization)
  4. BM25+ (additive TF floor)
  5. LMIR Jelinek-Mercer (language model smoothing)
  6. LMIR Dirichlet (Dirichlet smoothing)
  7. DFR (Divergence from Randomness)

Each method is a custom sklearn transformer producing sparse term-document
matrices, piped into SGDClassifier. Optuna's TPE sampler optimizes both
retrieval parameters and classifier hyperparameters.

Usage: python scripts/8_optuna_retrieval_search.py
"""
import os
import json
import time
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np
import pandas as pd
import optuna
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

DATA_DIR = 'data/training'
MODEL_DIR = 'models'
N_TRIALS = 80
TRIAL_TIMEOUT = 30  # seconds per trial

# ╔══════════════════════════════════════════════════════════════════╗
# ║               CUSTOM RETRIEVAL TRANSFORMERS                     ║
# ╚══════════════════════════════════════════════════════════════════╝

class BM25Vectorizer(BaseEstimator, TransformerMixin):
    """
    Okapi BM25 weighting as a sparse document-term matrix.
    Score(t,d) = IDF(t) * (tf * (k1+1)) / (tf + k1*(1-b+b*|d|/avgdl))
    """
    def __init__(self, k1=1.5, b=0.75, max_features=10000, ngram_range=(1,1)):
        self.k1 = k1
        self.b = b
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range
        )
        tf_raw = self.cv_.fit_transform(X)
        n_docs = tf_raw.shape[0]
        # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        df = np.diff(tf_raw.tocsc().indptr)  # doc frequency per term
        self.idf_ = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        # Average document length
        self.avgdl_ = tf_raw.sum(axis=1).mean()
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        # Doc lengths
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        # BM25 score per term
        tf = tf_raw.toarray().astype(np.float64)
        norm = 1.0 - self.b + self.b * (dl[:, None] / self.avgdl_)
        numerator = tf * (self.k1 + 1.0)
        denominator = tf + self.k1 * norm
        bm25 = self.idf_[None, :] * (numerator / denominator)
        return csr_matrix(bm25)


class BM25LVectorizer(BaseEstimator, TransformerMixin):
    """
    BM25L: Lv & Zhai (2011). Adds a delta to penalize long documents less.
    ctf = tf / (1 - b + b*|d|/avgdl)
    ctf' = ctf + delta
    Score = IDF * (k1+1)*ctf' / (k1+ctf')
    """
    def __init__(self, k1=1.5, b=0.75, delta=0.5, max_features=10000, ngram_range=(1,1)):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        tf_raw = self.cv_.fit_transform(X)
        n_docs = tf_raw.shape[0]
        df = np.diff(tf_raw.tocsc().indptr)
        self.idf_ = np.log((n_docs + 1.0) / (df + 0.5))
        self.avgdl_ = tf_raw.sum(axis=1).mean()
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        # Corrected TF
        ctf = tf / (1.0 - self.b + self.b * (dl[:, None] / self.avgdl_))
        ctf_prime = ctf + self.delta
        score = self.idf_[None, :] * ((self.k1 + 1.0) * ctf_prime) / (self.k1 + ctf_prime)
        return csr_matrix(score)


class BM25PlusVectorizer(BaseEstimator, TransformerMixin):
    """
    BM25+: Lv & Zhai (2011). Adds delta floor to BM25 to prevent zero scores.
    Score = IDF * ((k1+1)*tf/(k1*(1-b+b*|d|/avgdl)+tf) + delta)
    """
    def __init__(self, k1=1.5, b=0.75, delta=1.0, max_features=10000, ngram_range=(1,1)):
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        tf_raw = self.cv_.fit_transform(X)
        n_docs = tf_raw.shape[0]
        df = np.diff(tf_raw.tocsc().indptr)
        self.idf_ = np.log((n_docs + 1.0) / (df + 0.5))
        self.avgdl_ = tf_raw.sum(axis=1).mean()
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        norm = self.k1 * (1.0 - self.b + self.b * (dl[:, None] / self.avgdl_))
        bm25_component = ((self.k1 + 1.0) * tf) / (norm + tf)
        score = self.idf_[None, :] * (bm25_component + self.delta)
        return csr_matrix(score)


class LMIRJMVectorizer(BaseEstimator, TransformerMixin):
    """
    Language Model IR: Jelinek-Mercer Smoothing.
    P(w|d) = lambda * P_ml(w|d) + (1-lambda) * P(w|C)
    Score = log(P(w|d)) for each term present.
    """
    def __init__(self, lambda_param=0.1, max_features=10000, ngram_range=(1,1)):
        self.lambda_param = lambda_param
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        tf_raw = self.cv_.fit_transform(X)
        # Collection probability: P(w|C) = cf(w) / total_tokens
        total = tf_raw.sum()
        self.p_collection_ = np.asarray(tf_raw.sum(axis=0)).flatten() / total
        # Avoid log(0)
        self.p_collection_ = np.maximum(self.p_collection_, 1e-10)
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        # P_ml(w|d) = tf(w,d) / |d|
        dl_safe = np.maximum(dl, 1.0)
        p_ml = tf / dl_safe[:, None]
        # Jelinek-Mercer smoothed probability
        p_smooth = self.lambda_param * p_ml + (1.0 - self.lambda_param) * self.p_collection_[None, :]
        # Log-probability score (only for terms that appear)
        score = np.log(p_smooth + 1e-10)
        # Zero out terms that don't appear in the document (optional, keeps sparsity intent)
        score = score * (tf > 0).astype(np.float64)
        return csr_matrix(score)


class LMIRDirichletVectorizer(BaseEstimator, TransformerMixin):
    """
    Language Model IR: Dirichlet Smoothing.
    P(w|d) = (tf(w,d) + mu * P(w|C)) / (|d| + mu)
    Score = log(P(w|d)) for present terms.
    """
    def __init__(self, mu=2000.0, max_features=10000, ngram_range=(1,1)):
        self.mu = mu
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        tf_raw = self.cv_.fit_transform(X)
        total = tf_raw.sum()
        self.p_collection_ = np.asarray(tf_raw.sum(axis=0)).flatten() / total
        self.p_collection_ = np.maximum(self.p_collection_, 1e-10)
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        # Dirichlet smoothed
        numerator = tf + self.mu * self.p_collection_[None, :]
        denominator = dl[:, None] + self.mu
        p_smooth = numerator / denominator
        score = np.log(p_smooth + 1e-10)
        score = score * (tf > 0).astype(np.float64)
        return csr_matrix(score)


class DFRVectorizer(BaseEstimator, TransformerMixin):
    """
    Divergence from Randomness (DFR) weighting.
    Implements the InexpB2 model:
      info = -log2(P_random(tf|cf)) ≈ tf * log2(1 + cf/N) + log2(e) * (1/(tf+1) - 1)
      normalization = tf * log2(1 + avgdl/dl)
      score = info * normalization * (tf/(tf+1))  [Laplace after-effect]
    Simplified to tf_norm * idf_dfr for stability.
    """
    def __init__(self, c=1.0, max_features=10000, ngram_range=(1,1)):
        self.c = c  # normalization constant
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(max_features=self.max_features, ngram_range=self.ngram_range)
        tf_raw = self.cv_.fit_transform(X)
        n_docs = tf_raw.shape[0]
        # Collection frequency per term
        cf = np.asarray(tf_raw.sum(axis=0)).flatten().astype(np.float64)
        # DFR IDF: log2(1 + (N+1)/(cf+0.5))
        self.idf_dfr_ = np.log2(1.0 + (n_docs + 1.0) / (cf + 0.5))
        self.avgdl_ = tf_raw.sum(axis=1).mean()
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        # Length normalization
        dl_safe = np.maximum(dl, 1.0)
        tf_norm = tf * self.c * np.log2(1.0 + self.avgdl_ / dl_safe)[:, None]
        # After-effect: Laplace
        after_effect = tf_norm / (tf_norm + 1.0)
        score = after_effect * self.idf_dfr_[None, :]
        return csr_matrix(score)


# ╔══════════════════════════════════════════════════════════════════╗
# ║                    OPTUNA OBJECTIVE                             ║
# ╚══════════════════════════════════════════════════════════════════╝

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, val, test


def create_vectorizer(trial):
    """Optuna trial suggests a retrieval method + its hyperparameters."""
    method = trial.suggest_categorical('method', [
        'tfidf', 'bm25', 'bm25l', 'bm25plus', 'lmir_jm', 'lmir_dirichlet', 'dfr'
    ])

    if method == 'tfidf':
        # TF-IDF stays sparse — can handle 50k features and trigrams
        max_features = trial.suggest_categorical('tfidf_max_features', [5000, 10000, 20000, 50000])
        ngram_max = trial.suggest_int('tfidf_ngram_max', 1, 3)
        ngram_range = (1, ngram_max)
        sublinear_tf = trial.suggest_categorical('tfidf_sublinear', [True, False])
        stop_words = trial.suggest_categorical('tfidf_stopwords', ['english', None])
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            stop_words=stop_words
        ), method
    else:
        # Custom vectorizers use dense intermediates — cap at 10k features and bigrams
        max_features = trial.suggest_categorical('custom_max_features', [5000, 10000])
        ngram_max = trial.suggest_int('custom_ngram_max', 1, 2)
        ngram_range = (1, ngram_max)

    if method == 'bm25':
        k1 = trial.suggest_float('bm25_k1', 0.5, 3.0)
        b = trial.suggest_float('bm25_b', 0.0, 1.0)
        return BM25Vectorizer(k1=k1, b=b, max_features=max_features, ngram_range=ngram_range), method

    elif method == 'bm25l':
        k1 = trial.suggest_float('bm25l_k1', 0.5, 3.0)
        b = trial.suggest_float('bm25l_b', 0.0, 1.0)
        delta = trial.suggest_float('bm25l_delta', 0.0, 2.0)
        return BM25LVectorizer(k1=k1, b=b, delta=delta, max_features=max_features, ngram_range=ngram_range), method

    elif method == 'bm25plus':
        k1 = trial.suggest_float('bm25plus_k1', 0.5, 3.0)
        b = trial.suggest_float('bm25plus_b', 0.0, 1.0)
        delta = trial.suggest_float('bm25plus_delta', 0.1, 3.0)
        return BM25PlusVectorizer(k1=k1, b=b, delta=delta, max_features=max_features, ngram_range=ngram_range), method

    elif method == 'lmir_jm':
        lam = trial.suggest_float('jm_lambda', 0.01, 0.9)
        return LMIRJMVectorizer(lambda_param=lam, max_features=max_features, ngram_range=ngram_range), method

    elif method == 'lmir_dirichlet':
        mu = trial.suggest_float('dirichlet_mu', 100.0, 10000.0, log=True)
        return LMIRDirichletVectorizer(mu=mu, max_features=max_features, ngram_range=ngram_range), method

    elif method == 'dfr':
        c = trial.suggest_float('dfr_c', 0.1, 10.0, log=True)
        return DFRVectorizer(c=c, max_features=max_features, ngram_range=ngram_range), method


def _run_trial_inner(vectorizer, method, alpha, loss, X_train, y_train, X_val, y_val):
    """Inner function that does the actual fit+predict (runs in thread)."""
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('clf', SGDClassifier(
            loss=loss,
            alpha=alpha,
            penalty='l2',
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1
        ))
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    return f1_score(y_val, preds, average='weighted', zero_division=0)


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective: trains pipeline with per-trial timeout."""
    vectorizer, method = create_vectorizer(trial)

    # Classifier hyperparameters
    alpha = trial.suggest_float('sgd_alpha', 1e-6, 1e-1, log=True)
    loss = trial.suggest_categorical('sgd_loss', ['modified_huber', 'log_loss'])

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _run_trial_inner, vectorizer, method, alpha, loss,
                X_train, y_train, X_val, y_val
            )
            f1 = future.result(timeout=TRIAL_TIMEOUT)
        elapsed = time.time() - t0
        logging.info(f"Trial {trial.number:3d} | {method:20s} | F1={f1:.4f} | {elapsed:.1f}s")
        return f1
    except FuturesTimeoutError:
        elapsed = time.time() - t0
        logging.warning(
            f"Trial {trial.number:3d} | {method:20s} | SKIPPED (timeout {TRIAL_TIMEOUT}s exceeded, ran {elapsed:.1f}s) "
            f"| params: max_feat={trial.params.get('custom_max_features', trial.params.get('tfidf_max_features'))}, "
            f"ngram={trial.params.get('custom_ngram_max', trial.params.get('tfidf_ngram_max'))}"
        )
        return 0.0
    except Exception as e:
        elapsed = time.time() - t0
        logging.warning(
            f"Trial {trial.number:3d} | {method:20s} | FAILED ({type(e).__name__}: {e}) | {elapsed:.1f}s"
        )
        return 0.0


# ╔══════════════════════════════════════════════════════════════════╗
# ║              OOD THRESHOLD CALIBRATION & EVAL                   ║
# ╚══════════════════════════════════════════════════════════════════╝

def calibrate_threshold(pipeline, X_val, y_val):
    val_probs = pipeline.predict_proba(X_val)
    val_preds = pipeline.predict(X_val)
    max_probs = np.max(val_probs, axis=1)
    correct_mask = (val_preds == y_val)
    correct_confs = max_probs[correct_mask]
    if len(correct_confs) > 0:
        return float(np.percentile(correct_confs, 5))
    return 0.5


def evaluate_with_ood(pipeline, threshold, X_test, y_test):
    test_probs = pipeline.predict_proba(X_test)
    raw_preds = pipeline.classes_[np.argmax(test_probs, axis=1)]
    max_probs = np.max(test_probs, axis=1)

    final_preds = []
    for pred, prob in zip(raw_preds, max_probs):
        final_preds.append('other' if prob < threshold else pred)

    labels = sorted(list(set(y_test) | set(final_preds)))
    report = classification_report(y_test, final_preds, labels=labels, output_dict=True, zero_division=0)
    f1 = f1_score(y_test, final_preds, labels=labels, average='weighted', zero_division=0)

    other_mask = (y_test == 'other')
    ood_recall = 0.0
    if np.sum(other_mask) > 0:
        ood_recall = float(np.mean(np.array(final_preds)[other_mask] == 'other'))

    return f1, ood_recall, report


# ╔══════════════════════════════════════════════════════════════════╗
# ║                        MAIN                                     ║
# ╚══════════════════════════════════════════════════════════════════╝

def main():
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    train, val, test = load_data()

    X_train, y_train = train['text'], train['category']
    X_val, y_val = val['text'], val['category']
    X_test, y_test = test['text'], test['category']

    print("=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH: 7 CPU-ONLY RETRIEVAL METHODS")
    print(f"  Methods: TF-IDF, BM25, BM25L, BM25+, LMIR-JM, LMIR-Dirichlet, DFR")
    print(f"  Trials: {N_TRIALS} | Sampler: TPE")
    print("=" * 70)

    # ── Optuna Study ───────────────────────────────────────────────
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='retrieval_method_search'
    )

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        timeout=N_TRIALS * TRIAL_TIMEOUT,
        show_progress_bar=True
    )
    search_time = time.time() - t0

    print(f"\nSearch completed in {search_time:.1f}s")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best Val F1: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    # ── Rebuild Best Pipeline ──────────────────────────────────────
    best = study.best_trial
    vectorizer, method = create_vectorizer(best)
    best_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('clf', SGDClassifier(
            loss=best.params['sgd_loss'],
            alpha=best.params['sgd_alpha'],
            penalty='l2',
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            n_jobs=-1
        ))
    ])

    print(f"\nRetraining best config ({method}) ...")
    best_pipeline.fit(X_train, y_train)

    # Calibrate OOD threshold
    threshold = calibrate_threshold(best_pipeline, X_val, y_val)
    test_f1, ood_recall, test_report = evaluate_with_ood(best_pipeline, threshold, X_test, y_test)

    print(f"Test F1: {test_f1:.4f} | OOD Recall: {ood_recall:.2%} | Threshold: {threshold:.4f}")

    # ── Save Model ─────────────────────────────────────────────────
    artifact = {
        'pipeline': best_pipeline,
        'threshold': threshold,
        'model_version': f"optuna_{method}",
        'metrics': {
            'test_f1_weighted': test_f1,
            'ood_recall': ood_recall,
            'classification_report': test_report
        }
    }
    model_path = os.path.join(MODEL_DIR, 'baseline.joblib')
    joblib.dump(artifact, model_path)
    print(f"Best model saved to {model_path}")

    # ── Per-Method Summary ─────────────────────────────────────────
    method_best = {}
    for t in study.trials:
        if t.value is not None and t.value > 0:
            m = t.params.get('method', 'unknown')
            if m not in method_best or t.value > method_best[m]['val_f1']:
                method_best[m] = {
                    'val_f1': t.value,
                    'trial': t.number,
                    'params': {k: v for k, v in t.params.items() if k != 'method'}
                }

    print(f"\n{'='*70}")
    print("BEST RESULT PER RETRIEVAL METHOD:")
    print(f"{'='*70}")
    for m in ['tfidf', 'bm25', 'bm25l', 'bm25plus', 'lmir_jm', 'lmir_dirichlet', 'dfr']:
        if m in method_best:
            info = method_best[m]
            marker = " ← WINNER" if m == method else ""
            print(f"  {m:20s} Val F1={info['val_f1']:.4f}  (trial #{info['trial']}){marker}")

    # ── Save Results ───────────────────────────────────────────────
    all_trials = []
    for t in study.trials:
        if t.value is not None:
            all_trials.append({
                'trial': t.number,
                'method': t.params.get('method', 'unknown'),
                'val_f1': round(t.value, 4),
                'params': {k: (str(v) if isinstance(v, tuple) else v) for k, v in t.params.items()},
                'duration_sec': round(t.duration.total_seconds(), 2) if t.duration else None
            })

    results = {
        'search_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'search_type': 'optuna_tpe',
        'n_trials': N_TRIALS,
        'search_time_sec': round(search_time, 1),
        'methods_searched': ['tfidf', 'bm25', 'bm25l', 'bm25plus', 'lmir_jm', 'lmir_dirichlet', 'dfr'],
        'winner': {
            'method': method,
            'trial': best.number,
            'val_f1': round(study.best_value, 4),
            'test_f1': round(test_f1, 4),
            'ood_recall': round(ood_recall, 4),
            'threshold': round(threshold, 4),
            'params': {k: (str(v) if isinstance(v, tuple) else v) for k, v in best.params.items()},
            'classification_report': test_report
        },
        'per_method_best': {
            m: {'val_f1': round(info['val_f1'], 4), 'trial': info['trial']}
            for m, info in method_best.items()
        },
        'all_trials': sorted(all_trials, key=lambda x: x['val_f1'], reverse=True)
    }

    results_path = os.path.join(MODEL_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"DONE. Winner: {method} | Test F1: {test_f1:.4f} | OOD: {ood_recall:.2%}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
