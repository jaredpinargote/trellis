"""
Demo: Quick Stress Test.
Fires concurrent requests and reports latency percentiles + throughput.

Usage: python scripts/demo_stress.py
  (requires API running on localhost:8000)
"""
import requests
import time
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"
N_REQUESTS = 200
N_WORKERS = 20

TEXTS = [
    "The stock market reached an all-time high today as investor confidence surged.",
    "Scientists discovered water ice beneath the surface of Mars using ground-penetrating radar.",
    "The quarterback completed a game-winning drive in the final two minutes of the Super Bowl.",
    "New immunotherapy treatments show promise for patients with advanced melanoma cancer.",
    "The new programming language offers memory safety without garbage collection overhead.",
]


def send_request(text):
    """Send one classify request, return latency in ms."""
    t0 = time.perf_counter()
    try:
        r = requests.post(
            f"{API}/classify_document",
            json={"document_text": text},
            timeout=10
        )
        latency = (time.perf_counter() - t0) * 1000
        return latency, r.status_code
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return latency, 0


def main():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ API not reachable at {API}: {e}")
        print("   Start it with: python scripts/run_api.py")
        sys.exit(1)

    print("=" * 80)
    print(f"DEMO: Stress Test ({N_REQUESTS} requests, {N_WORKERS} concurrent workers)")
    print("=" * 80)
    print(f"  Model: {r.json()['model_version']}\n")
    print("  Sending requests...")

    latencies = []
    errors = 0
    texts = [TEXTS[i % len(TEXTS)] for i in range(N_REQUESTS)]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(send_request, t): t for t in texts}
        for future in as_completed(futures):
            lat, code = future.result()
            latencies.append(lat)
            if code != 200:
                errors += 1
    total_time = time.time() - t0

    arr = np.array(latencies)
    rps = N_REQUESTS / total_time

    print(f"\n{'─'*60}")
    print(f"  ⚡ STRESS TEST RESULTS")
    print(f"{'─'*60}")
    print(f"  Total Requests:    {N_REQUESTS}")
    print(f"  Concurrent Workers: {N_WORKERS}")
    print(f"  Total Time:        {total_time:.2f}s")
    print(f"  Throughput:        {rps:.0f} req/s")
    print(f"  Errors:            {errors} ({errors/N_REQUESTS:.1%})")
    print(f"\n  LATENCY (end-to-end, including network):")
    print(f"    Mean:   {np.mean(arr):.1f}ms")
    print(f"    p50:    {np.median(arr):.1f}ms")
    print(f"    p95:    {np.percentile(arr, 95):.1f}ms")
    print(f"    p99:    {np.percentile(arr, 99):.1f}ms")
    print(f"    Min:    {np.min(arr):.1f}ms")
    print(f"    Max:    {np.max(arr):.1f}ms")

    # Pass/Fail criteria
    print(f"\n  PRODUCTION CHECKS:")
    checks = [
        (f"Error rate < 1%", errors / N_REQUESTS < 0.01),
        (f"p99 latency < 200ms", np.percentile(arr, 99) < 200),
        (f"Throughput > 50 req/s", rps > 50),
    ]
    all_pass = True
    for desc, ok in checks:
        print(f"    {'✅' if ok else '❌'} {desc}")
        if not ok:
            all_pass = False

    print(f"\n{'─'*60}")
    if all_pass:
        print("  ✅ ALL STRESS TEST CHECKS PASSED")
    else:
        print("  ❌ Some checks failed — investigate")
    print(f"{'─'*60}\n")


if __name__ == '__main__':
    main()
