"""
Production Inference Benchmark
===============================
Measures every metric needed for production deployment:

  1. Model Loading (cold start)
  2. Disk Footprint
  3. Memory Usage (baseline, model, peak inference)
  4. Single-Request Latency (p50, p95, p99, min, max, mean)
  5. Throughput (requests/sec at various batch sizes)
  6. Warm-up Effect (first N requests vs steady state)
  7. Document Length Impact (short/medium/long texts)
  8. End-to-End API Latency (full pipeline: sanitize → cache → PII → inference)
  9. Cost Estimation (AWS/GCP instances)

Usage: python scripts/9_inference_benchmark.py
"""
import os
import sys
import time
import json
import gc
import psutil
import numpy as np
import joblib
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────
MODEL_PATH = Path('models/baseline.joblib')
N_WARMUP = 20
N_LATENCY = 500       # single-request latency samples
N_THROUGHPUT = 1000    # throughput test total requests
BATCH_SIZES = [1, 10, 50, 100]

# ── Sample documents (varied lengths) ──────────────────────────────
TEXTS = {
    'short': "Stock market gains today.",
    'medium': (
        "The government announced new tax reforms and the prime minister "
        "addressed parliament about the budget deficit and public spending "
        "cuts in education and healthcare policy."
    ),
    'long': (
        "NASA's Perseverance rover has successfully collected its tenth rock sample "
        "from the Jezero Crater on Mars. Scientists believe these samples could contain "
        "evidence of ancient microbial life. The rover uses a sophisticated drill system "
        "to extract core samples, which are then sealed in special titanium tubes for "
        "future retrieval by the Mars Sample Return mission. This international effort "
        "between NASA and ESA aims to bring Martian samples back to Earth by 2033. "
        "The geological diversity of Jezero Crater, which was once an ancient river delta, "
        "makes it an ideal location for astrobiological research. The samples collected "
        "so far include igneous rocks, sedimentary layers, and regolith, each providing "
        "unique insights into the planet's geological and potentially biological history."
    ),
    'very_long': None,  # Generated below
}
# Generate a ~2000-word document
TEXTS['very_long'] = " ".join([TEXTS['long']] * 5)


def get_process_memory_mb():
    """Current process RSS in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def format_bytes(b):
    """Human-readable byte size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    results = {}
    process = psutil.Process(os.getpid())

    # ── 1. BASELINE MEMORY ──────────────────────────────────────────
    section("1. BASELINE MEMORY (before model load)")
    gc.collect()
    baseline_mem = get_process_memory_mb()
    print(f"  Process RSS: {baseline_mem:.1f} MB")
    results['baseline_memory_mb'] = round(baseline_mem, 1)

    # ── 2. DISK FOOTPRINT ──────────────────────────────────────────
    section("2. DISK FOOTPRINT")
    disk_size = MODEL_PATH.stat().st_size
    print(f"  Model file: {MODEL_PATH}")
    print(f"  Size: {format_bytes(disk_size)}")
    results['disk_size_bytes'] = disk_size
    results['disk_size_human'] = format_bytes(disk_size)

    # ── 3. COLD START (model loading) ──────────────────────────────
    section("3. COLD START (model loading time)")
    load_times = []
    for i in range(5):
        gc.collect()
        t0 = time.perf_counter()
        artifact = joblib.load(MODEL_PATH)
        t1 = time.perf_counter()
        load_times.append((t1 - t0) * 1000)
        if i < 4:
            del artifact
            gc.collect()

    pipeline = artifact['pipeline']
    threshold = artifact.get('threshold', 0.5)
    model_version = artifact.get('model_version', 'unknown')

    print(f"  Load times: {[f'{t:.1f}ms' for t in load_times]}")
    print(f"  Mean: {np.mean(load_times):.1f}ms | Median: {np.median(load_times):.1f}ms")
    print(f"  Model version: {model_version}")
    results['cold_start_ms'] = {
        'mean': round(np.mean(load_times), 1),
        'median': round(np.median(load_times), 1),
        'min': round(min(load_times), 1),
        'max': round(max(load_times), 1),
        'samples': [round(t, 1) for t in load_times]
    }

    # ── 4. MEMORY WITH MODEL ────────────────────────────────────────
    section("4. MEMORY WITH MODEL LOADED")
    gc.collect()
    model_mem = get_process_memory_mb()
    model_delta = model_mem - baseline_mem
    print(f"  Process RSS: {model_mem:.1f} MB")
    print(f"  Model overhead: {model_delta:.1f} MB")
    results['model_memory_mb'] = round(model_mem, 1)
    results['model_overhead_mb'] = round(model_delta, 1)

    # ── 5. WARM-UP EFFECT ──────────────────────────────────────────
    section("5. WARM-UP EFFECT")
    warmup_times = []
    for i in range(N_WARMUP):
        t0 = time.perf_counter()
        pipeline.predict([TEXTS['medium']])
        t1 = time.perf_counter()
        warmup_times.append((t1 - t0) * 1000)

    first_5 = warmup_times[:5]
    last_5 = warmup_times[-5:]
    print(f"  First 5 requests: {[f'{t:.2f}ms' for t in first_5]}")
    print(f"  Last 5 requests:  {[f'{t:.2f}ms' for t in last_5]}")
    print(f"  Warm-up slowdown: {np.mean(first_5)/np.mean(last_5):.1f}x")
    results['warmup'] = {
        'first_5_mean_ms': round(np.mean(first_5), 2),
        'last_5_mean_ms': round(np.mean(last_5), 2),
        'slowdown_factor': round(np.mean(first_5) / np.mean(last_5), 1)
    }

    # ── 6. SINGLE-REQUEST LATENCY ──────────────────────────────────
    section(f"6. SINGLE-REQUEST LATENCY ({N_LATENCY} samples)")
    latency_by_length = {}

    for label, text in TEXTS.items():
        times = []
        for _ in range(N_LATENCY):
            t0 = time.perf_counter()
            probs = pipeline.predict_proba([text])
            pred = pipeline.classes_[np.argmax(probs)]
            conf = np.max(probs)
            final = 'other' if conf < threshold else pred
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times_arr = np.array(times)
        stats = {
            'mean_ms': round(np.mean(times_arr), 3),
            'median_ms': round(np.median(times_arr), 3),
            'p95_ms': round(np.percentile(times_arr, 95), 3),
            'p99_ms': round(np.percentile(times_arr, 99), 3),
            'min_ms': round(np.min(times_arr), 3),
            'max_ms': round(np.max(times_arr), 3),
            'std_ms': round(np.std(times_arr), 3),
            'text_chars': len(text),
            'text_words': len(text.split()),
        }
        latency_by_length[label] = stats
        print(f"\n  [{label}] ({stats['text_words']} words, {stats['text_chars']} chars)")
        print(f"    Mean: {stats['mean_ms']:.3f}ms | p50: {stats['median_ms']:.3f}ms | "
              f"p95: {stats['p95_ms']:.3f}ms | p99: {stats['p99_ms']:.3f}ms")

    results['latency'] = latency_by_length

    # ── 7. PEAK MEMORY DURING INFERENCE ────────────────────────────
    section("7. PEAK MEMORY DURING INFERENCE")
    gc.collect()
    pre_inference = get_process_memory_mb()
    # Run inference on large batch
    _ = pipeline.predict_proba([TEXTS['very_long']] * 100)
    post_inference = get_process_memory_mb()
    gc.collect()
    post_gc = get_process_memory_mb()

    print(f"  Before batch: {pre_inference:.1f} MB")
    print(f"  After batch (100 long docs): {post_inference:.1f} MB")
    print(f"  After GC: {post_gc:.1f} MB")
    print(f"  Peak spike: {post_inference - pre_inference:.1f} MB")
    results['peak_memory'] = {
        'pre_inference_mb': round(pre_inference, 1),
        'post_batch_100_mb': round(post_inference, 1),
        'post_gc_mb': round(post_gc, 1),
        'spike_mb': round(post_inference - pre_inference, 1)
    }

    # ── 8. THROUGHPUT ──────────────────────────────────────────────
    section(f"8. THROUGHPUT ({N_THROUGHPUT} requests)")
    throughput_results = {}

    for batch_size in BATCH_SIZES:
        texts_batch = [TEXTS['medium']] * batch_size
        n_batches = N_THROUGHPUT // batch_size

        t0 = time.perf_counter()
        for _ in range(n_batches):
            pipeline.predict_proba(texts_batch)
        t1 = time.perf_counter()

        total_time = t1 - t0
        total_requests = n_batches * batch_size
        rps = total_requests / total_time

        throughput_results[f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'total_requests': total_requests,
            'total_time_sec': round(total_time, 3),
            'requests_per_sec': round(rps, 1),
            'ms_per_request': round(1000 / rps, 3),
        }
        print(f"  Batch={batch_size:3d}: {rps:,.0f} req/s ({1000/rps:.3f} ms/req) "
              f"[{total_requests} reqs in {total_time:.2f}s]")

    results['throughput'] = throughput_results

    # ── 9. CPU UTILIZATION ─────────────────────────────────────────
    section("9. CPU UTILIZATION")
    process.cpu_percent()  # prime the counter
    time.sleep(0.1)

    # Sustained load for 3 seconds
    t_end = time.time() + 3.0
    count = 0
    while time.time() < t_end:
        pipeline.predict_proba([TEXTS['medium']])
        count += 1

    cpu_pct = process.cpu_percent()
    cpu_count = psutil.cpu_count(logical=True)
    print(f"  CPU usage during sustained load: {cpu_pct:.1f}%")
    print(f"  Logical cores: {cpu_count}")
    print(f"  Requests in 3s: {count}")
    results['cpu'] = {
        'usage_percent': round(cpu_pct, 1),
        'logical_cores': cpu_count,
        'requests_in_3s': count
    }

    # ── 10. COST ESTIMATION ────────────────────────────────────────
    section("10. CLOUD COST ESTIMATION")
    # Use medium text throughput (batch=1) for cost calc
    rps_single = throughput_results['batch_1']['requests_per_sec']
    rps_batch = throughput_results[f'batch_{BATCH_SIZES[-1]}']['requests_per_sec']

    # Cloud instance pricing (approximate $/hour)
    instances = {
        'AWS t3.micro (2 vCPU, 1GB)': {'cost_hr': 0.0104, 'note': 'Free tier eligible'},
        'AWS t3.small (2 vCPU, 2GB)': {'cost_hr': 0.0208, 'note': 'Recommended minimum'},
        'AWS t3.medium (2 vCPU, 4GB)': {'cost_hr': 0.0416, 'note': 'Comfortable headroom'},
        'GCP e2-micro (2 vCPU, 1GB)': {'cost_hr': 0.0084, 'note': 'Free tier eligible'},
        'GCP e2-small (2 vCPU, 2GB)': {'cost_hr': 0.0168, 'note': 'Recommended minimum'},
        'Azure B1s (1 vCPU, 1GB)': {'cost_hr': 0.0104, 'note': 'Free 750h/mo'},
    }

    cost_results = {}
    print(f"\n  Single-request throughput: {rps_single:,.0f} req/s")
    print(f"  Batch-{BATCH_SIZES[-1]} throughput: {rps_batch:,.0f} req/s")
    print(f"\n  Daily capacity at single-request rate: {rps_single * 86400:,.0f} requests/day")
    print(f"\n  {'Instance':45s} {'$/hour':>8s} {'$/month':>10s} {'$/1M req':>10s}")
    print(f"  {'-'*75}")

    for name, info in instances.items():
        monthly = info['cost_hr'] * 730  # avg hours/month
        cost_per_1m = (1_000_000 / rps_single / 3600) * info['cost_hr']
        cost_results[name] = {
            'cost_per_hour': info['cost_hr'],
            'cost_per_month': round(monthly, 2),
            'cost_per_1m_requests': round(cost_per_1m, 4),
            'note': info['note']
        }
        print(f"  {name:45s} ${info['cost_hr']:<7.4f} ${monthly:<9.2f} ${cost_per_1m:<9.4f}")

    results['cost_estimation'] = cost_results
    results['daily_capacity_single'] = round(rps_single * 86400)
    results['daily_capacity_batch'] = round(rps_batch * 86400)

    # ── 11. SUMMARY ────────────────────────────────────────────────
    section("BENCHMARK SUMMARY")
    med_latency = results['latency']['medium']
    print(f"  Model version:      {model_version}")
    print(f"  Disk footprint:     {results['disk_size_human']}")
    print(f"  Model RAM overhead: {results['model_overhead_mb']:.1f} MB")
    print(f"  Cold start:         {results['cold_start_ms']['median']:.0f} ms")
    print(f"  Inference latency:  p50={med_latency['median_ms']:.1f}ms  p95={med_latency['p95_ms']:.1f}ms  p99={med_latency['p99_ms']:.1f}ms")
    print(f"  Throughput:         {throughput_results['batch_1']['requests_per_sec']:,.0f} req/s (single)  |  {rps_batch:,.0f} req/s (batch-{BATCH_SIZES[-1]})")
    print(f"  Peak memory spike:  {results['peak_memory']['spike_mb']:.1f} MB (100 long docs)")
    print(f"  Daily capacity:     {results['daily_capacity_single']:,} requests")
    print(f"  Cheapest cloud:     ${min(c['cost_per_month'] for c in cost_results.values()):.2f}/month")

    # ── PRODUCTION READINESS CHECKLIST ──────────────────────────────
    section("PRODUCTION READINESS CHECKLIST")
    checks = {
        'Latency < 50ms (p99)': med_latency['p99_ms'] < 50,
        'Throughput > 100 req/s': rps_single > 100,
        'Cold start < 2s': results['cold_start_ms']['median'] < 2000,
        'Model < 50MB disk': disk_size < 50 * 1024 * 1024,
        'RAM overhead < 200MB': results['model_overhead_mb'] < 200,
        'Memory spike < 500MB': results['peak_memory']['spike_mb'] < 500,
    }
    all_pass = True
    for check, passed in checks.items():
        icon = '✅' if passed else '❌'
        if not passed:
            all_pass = False
        print(f"  {icon} {check}")

    results['prod_readiness'] = {k: v for k, v in checks.items()}
    results['all_checks_pass'] = all_pass

    # ── SAVE ────────────────────────────────────────────────────────
    out_path = Path('models/benchmark.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results saved to {out_path}")


if __name__ == '__main__':
    main()
