"""
Cross-Method Inference Benchmark: All 7 Retrieval Methods Compared
===================================================================
Rebuilds the Optuna-best config for each method, trains it,
and benchmarks accuracy vs speed vs memory vs cost.

This answers: "If two methods have similar accuracy but one is
significantly cheaper, which should we pick for production?"

Usage: python scripts/10_cross_method_benchmark.py
"""
import os
import sys
import gc
import time
import json
import numpy as np
import pandas as pd
import psutil
import joblib
from pathlib import Path

# Add project root so we can import the custom vectorizers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
_mod = import_module('8_optuna_retrieval_search')
BM25Vectorizer = _mod.BM25Vectorizer
BM25LVectorizer = _mod.BM25LVectorizer
BM25PlusVectorizer = _mod.BM25PlusVectorizer
LMIRJMVectorizer = _mod.LMIRJMVectorizer
LMIRDirichletVectorizer = _mod.LMIRDirichletVectorizer
DFRVectorizer = _mod.DFRVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = 'data/training'
N_LATENCY = 500
N_THROUGHPUT = 1000

MEDIUM_TEXT = (
    "The government announced new tax reforms and the prime minister "
    "addressed parliament about the budget deficit and public spending "
    "cuts in education and healthcare policy."
)
LONG_TEXT = (
    "NASA's Perseverance rover has successfully collected its tenth rock sample "
    "from the Jezero Crater on Mars. Scientists believe these samples could contain "
    "evidence of ancient microbial life. The rover uses a sophisticated drill system "
    "to extract core samples, which are then sealed in special titanium tubes for "
    "future retrieval by the Mars Sample Return mission. This international effort "
    "between NASA and ESA aims to bring Martian samples back to Earth by 2033."
)


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    val = pd.read_csv(os.path.join(DATA_DIR, 'val.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, val, test


def get_best_configs():
    """Extract the best Optuna trial for each method from results.json."""
    with open('models/results.json') as f:
        data = json.load(f)

    best = {}
    for trial in data['all_trials']:
        m = trial['method']
        if trial['val_f1'] > 0 and (m not in best or trial['val_f1'] > best[m]['val_f1']):
            best[m] = trial

    return best


def build_pipeline(method, params):
    """Build a pipeline from method name and Optuna params."""
    if method == 'tfidf':
        ngram_range = (1, params.get('tfidf_ngram_max', 2))
        vec = TfidfVectorizer(
            max_features=params.get('tfidf_max_features', 10000),
            ngram_range=ngram_range,
            sublinear_tf=params.get('tfidf_sublinear', True),
            stop_words=params.get('tfidf_stopwords', 'english') or None,
        )
    elif method == 'bm25':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = BM25Vectorizer(
            k1=params.get('bm25_k1', 1.5),
            b=params.get('bm25_b', 0.75),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    elif method == 'bm25l':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = BM25LVectorizer(
            k1=params.get('bm25l_k1', 1.5),
            b=params.get('bm25l_b', 0.75),
            delta=params.get('bm25l_delta', 0.5),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    elif method == 'bm25plus':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = BM25PlusVectorizer(
            k1=params.get('bm25plus_k1', 1.5),
            b=params.get('bm25plus_b', 0.75),
            delta=params.get('bm25plus_delta', 1.0),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    elif method == 'lmir_jm':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = LMIRJMVectorizer(
            lambda_param=params.get('jm_lambda', 0.1),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    elif method == 'lmir_dirichlet':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = LMIRDirichletVectorizer(
            mu=params.get('dirichlet_mu', 2000),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    elif method == 'dfr':
        ngram_range = (1, params.get('custom_ngram_max', 1))
        vec = DFRVectorizer(
            c=params.get('dfr_c', 1.0),
            max_features=params.get('custom_max_features', 10000),
            ngram_range=ngram_range,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return Pipeline([
        ('vectorizer', vec),
        ('clf', SGDClassifier(
            loss=params.get('sgd_loss', 'modified_huber'),
            alpha=params.get('sgd_alpha', 1e-4),
            penalty='l2', random_state=42, max_iter=1000, tol=1e-3,
        ))
    ])


def calibrate_threshold(pipeline, X_val, y_val):
    probs = pipeline.predict_proba(X_val)
    preds = pipeline.predict(X_val)
    correct = (preds == y_val)
    confs = np.max(probs, axis=1)[correct]
    return float(np.percentile(confs, 5)) if len(confs) > 0 else 0.5


def evaluate_with_ood(pipeline, threshold, X_test, y_test):
    probs = pipeline.predict_proba(X_test)
    raw = pipeline.classes_[np.argmax(probs, axis=1)]
    maxp = np.max(probs, axis=1)
    final = ['other' if p < threshold else r for r, p in zip(raw, maxp)]
    labels = sorted(set(y_test) | set(final))
    f1 = f1_score(y_test, final, labels=labels, average='weighted', zero_division=0)
    other_mask = (y_test == 'other')
    ood = float(np.mean(np.array(final)[other_mask] == 'other')) if np.sum(other_mask) > 0 else 0.0
    return f1, ood


def benchmark_inference(pipeline, threshold, text, n_samples):
    """Measure single-request latency."""
    times = []
    for _ in range(n_samples):
        t0 = time.perf_counter()
        probs = pipeline.predict_proba([text])
        pred = pipeline.classes_[np.argmax(probs)]
        conf = np.max(probs)
        _ = 'other' if conf < threshold else pred
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    return {
        'mean_ms': round(np.mean(arr), 3),
        'p50_ms': round(np.median(arr), 3),
        'p95_ms': round(np.percentile(arr, 95), 3),
        'p99_ms': round(np.percentile(arr, 99), 3),
    }


def benchmark_throughput(pipeline, text, n_total):
    """Measure requests/sec at batch_size=1."""
    t0 = time.perf_counter()
    for _ in range(n_total):
        pipeline.predict_proba([text])
    elapsed = time.perf_counter() - t0
    return {
        'total_sec': round(elapsed, 3),
        'req_per_sec': round(n_total / elapsed, 1),
    }


def measure_model_size(pipeline):
    """Save to tmp, measure disk size, delete."""
    tmp = '__tmp_model.joblib'
    joblib.dump({'pipeline': pipeline, 'threshold': 0.5}, tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def main():
    train, val, test = load_data()
    X_train, y_train = train['text'], train['category']
    X_val, y_val = val['text'], val['category']
    X_test, y_test = test['text'], test['category']

    configs = get_best_configs()
    method_order = ['tfidf', 'bm25', 'bm25l', 'bm25plus', 'lmir_jm', 'lmir_dirichlet', 'dfr']

    print("=" * 100)
    print("CROSS-METHOD INFERENCE BENCHMARK: 7 CPU-ONLY RETRIEVAL METHODS")
    print("=" * 100)

    all_results = {}

    for method in method_order:
        if method not in configs:
            print(f"\n  ⚠ {method} — no successful trial, skipping")
            continue

        trial_info = configs[method]
        params = trial_info['params']
        val_f1 = trial_info['val_f1']

        print(f"\n{'─'*100}")
        print(f"  METHOD: {method.upper()}")
        print(f"  Best Optuna Val F1: {val_f1:.4f} (trial #{trial_info['trial']})")
        print(f"{'─'*100}")

        # ── Train ──────────────────────────────────────────────────
        pipeline = build_pipeline(method, params)
        t0 = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_time = (time.perf_counter() - t0) * 1000

        # ── Evaluate ───────────────────────────────────────────────
        threshold = calibrate_threshold(pipeline, X_val, y_val)
        test_f1, ood_recall = evaluate_with_ood(pipeline, threshold, X_test, y_test)
        print(f"  Test F1: {test_f1:.4f} | OOD Recall: {ood_recall:.2%} | Threshold: {threshold:.4f}")

        # ── Model Size ─────────────────────────────────────────────
        disk_bytes = measure_model_size(pipeline)
        disk_kb = disk_bytes / 1024

        # ── Memory ─────────────────────────────────────────────────
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        _ = pipeline.predict_proba([LONG_TEXT] * 50)
        mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
        gc.collect()

        # ── Latency ────────────────────────────────────────────────
        latency = benchmark_inference(pipeline, threshold, MEDIUM_TEXT, N_LATENCY)
        print(f"  Latency: p50={latency['p50_ms']:.2f}ms  p95={latency['p95_ms']:.2f}ms  p99={latency['p99_ms']:.2f}ms")

        # ── Throughput ─────────────────────────────────────────────
        tp = benchmark_throughput(pipeline, MEDIUM_TEXT, N_THROUGHPUT)
        print(f"  Throughput: {tp['req_per_sec']:,.0f} req/s")
        print(f"  Train time: {train_time:.0f}ms | Disk: {disk_kb:.0f} KB | Mem spike: {mem_after-mem_before:.1f} MB")

        # ── Cost at $0.01/hour ─────────────────────────────────────
        cost_per_1m = (1_000_000 / tp['req_per_sec'] / 3600) * 0.01
        daily_capacity = tp['req_per_sec'] * 86400

        all_results[method] = {
            'val_f1': val_f1,
            'test_f1': round(test_f1, 4),
            'ood_recall': round(ood_recall, 4),
            'threshold': round(threshold, 4),
            'train_time_ms': round(train_time, 0),
            'disk_kb': round(disk_kb, 0),
            'mem_spike_mb': round(mem_after - mem_before, 1),
            'latency_p50_ms': latency['p50_ms'],
            'latency_p95_ms': latency['p95_ms'],
            'latency_p99_ms': latency['p99_ms'],
            'throughput_rps': tp['req_per_sec'],
            'cost_per_1m_req': round(cost_per_1m, 6),
            'daily_capacity': round(daily_capacity),
            'trial': trial_info['trial'],
            'params': params,
        }

        del pipeline
        gc.collect()

    # ── COMPARISON TABLE ───────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("CROSS-METHOD COMPARISON (sorted by Test F1)")
    print(f"{'='*100}")

    header = (f"{'Method':<20s} {'Val F1':>7s} {'Test F1':>7s} {'OOD':>6s} "
              f"{'p50':>7s} {'p95':>7s} {'p99':>7s} {'req/s':>8s} "
              f"{'Disk':>7s} {'Train':>7s} {'$/1M':>8s}")
    print(header)
    print("─" * 100)

    sorted_methods = sorted(all_results.items(), key=lambda x: -x[1]['test_f1'])
    best_f1 = sorted_methods[0][1]['test_f1']

    for method, r in sorted_methods:
        delta_f1 = r['test_f1'] - best_f1
        marker = " ★" if delta_f1 == 0 else f" ({delta_f1:+.2%})"
        print(
            f"{method:<20s} {r['val_f1']:>7.4f} {r['test_f1']:>7.4f} {r['ood_recall']:>5.1%} "
            f"{r['latency_p50_ms']:>6.2f}ms {r['latency_p95_ms']:>6.2f}ms {r['latency_p99_ms']:>6.2f}ms "
            f"{r['throughput_rps']:>7,.0f} "
            f"{r['disk_kb']:>5.0f}KB {r['train_time_ms']:>5.0f}ms "
            f"${r['cost_per_1m_req']:>7.5f}{marker}"
        )

    # ── COST-ACCURACY FRONTIER ─────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("COST vs ACCURACY TRADEOFF ANALYSIS")
    print(f"{'='*100}")

    for method, r in sorted_methods:
        f1_diff = abs(r['test_f1'] - best_f1)
        speed_ratio = all_results[sorted_methods[0][0]]['latency_p50_ms'] / max(r['latency_p50_ms'], 0.001)
        if f1_diff < 0.01:
            verdict = "✅ VIABLE — Similar accuracy"
        elif f1_diff < 0.03:
            verdict = "⚠️  MARGINAL — Slightly lower accuracy"
        else:
            verdict = "❌ NOT RECOMMENDED — Accuracy gap too large"

        print(f"\n  {method.upper()}")
        print(f"    F1 gap from best: {f1_diff:+.4f} ({f1_diff:.2%})")
        print(f"    Speed vs winner: {speed_ratio:.2f}x")
        print(f"    Cost per 1M requests: ${r['cost_per_1m_req']:.5f}")
        print(f"    → {verdict}")

    # ── RECOMMENDATION ─────────────────────────────────────────────
    cheapest_viable = None
    for method, r in sorted(all_results.items(), key=lambda x: x[1]['cost_per_1m_req']):
        if abs(r['test_f1'] - best_f1) < 0.01:
            cheapest_viable = (method, r)
            break

    winner_method, winner_r = sorted_methods[0]
    print(f"\n\n{'='*100}")
    print("RECOMMENDATION")
    print(f"{'='*100}")
    print(f"  Best accuracy: {winner_method} (F1={winner_r['test_f1']:.4f})")
    if cheapest_viable:
        cm, cr = cheapest_viable
        print(f"  Cheapest viable: {cm} (F1={cr['test_f1']:.4f}, ${cr['cost_per_1m_req']:.5f}/1M)")
        if cm != winner_method:
            speed_gain = winner_r['latency_p50_ms'] / cr['latency_p50_ms']
            print(f"  Speed gain: {speed_gain:.2f}x faster")
            print(f"  F1 gap: {abs(cr['test_f1'] - winner_r['test_f1']):.4f} ({abs(cr['test_f1'] - winner_r['test_f1']):.2%})")
        else:
            print(f"  → Winner IS the cheapest viable option!")

    # ── Save ───────────────────────────────────────────────────────
    output = {
        'benchmark_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_latency_samples': N_LATENCY,
        'n_throughput_samples': N_THROUGHPUT,
        'methods': all_results,
        'recommendation': {
            'best_accuracy': winner_method,
            'cheapest_viable': cheapest_viable[0] if cheapest_viable else None,
        }
    }
    out_path = Path('models/cross_method_benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == '__main__':
    main()
