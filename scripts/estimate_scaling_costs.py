"""
Estimate Scaling Costs: Benchmark API latency and project cloud costs.
=====================================================================
Sends requests to a running API and estimates AWS Lambda/EC2 costs
for processing 1M documents at observed throughput.

Usage: python scripts/estimate_scaling_costs.py
(Ensure API is running: uvicorn app.main:app --port 8000)
"""
import time
import requests
import statistics
import concurrent.futures
import sys

sys.stdout.reconfigure(encoding='utf-8')

# --- Configuration ---
API_URL = "http://localhost:8000/classify_document"
HEALTH_URL = "http://localhost:8000/health"
SEQUENTIAL_BATCH = 20    # Sequential requests for true latency measurement
CONCURRENT_BATCH = 100   # Concurrent requests for throughput measurement
CONCURRENCY = 5          # Thread pool size
TARGET_DOCS = 1_000_000

# Cloud Cost Assumptions (Approximate Retail Rates)
LAMBDA_GB_SEC_RATE = 0.0000166667  # per GB-second
LAMBDA_REQ_RATE = 0.20 / 1_000_000
EC2_HOURLY_RATE = 0.0416  # t3.medium (2 vCPU, 4GB)


def measure_sequential_latency(session):
    """Measure true per-request latency (no queueing effects)."""
    print(f"--- Phase 1: Sequential Latency ({SEQUENTIAL_BATCH} requests) ---")

    payload = {"document_text": "The stock market rallied yesterday as the tech sector showed strong quarterly earnings."}
    times = []

    for i in range(SEQUENTIAL_BATCH):
        t0 = time.time()
        r = session.post(API_URL, json=payload)
        lat = time.time() - t0
        if r.status_code == 200:
            times.append(lat)
            # Also capture the server-reported processing time
            server_time = r.json().get("processing_time_ms", 0)

    if not times:
        print("❌ All requests failed.")
        return None

    avg = statistics.mean(times)
    p99 = sorted(times)[int(len(times) * 0.99)] if len(times) >= 20 else max(times)

    print(f"   Avg Latency:  {avg*1000:.1f} ms (includes network round-trip)")
    print(f"   P99 Latency:  {p99*1000:.1f} ms")
    print(f"   Server Time:  {server_time:.1f} ms (model inference only)")
    return {"avg_latency": avg, "p99_latency": p99}


def measure_throughput(session):
    """Measure throughput under concurrency."""
    print(f"\n--- Phase 2: Throughput ({CONCURRENT_BATCH} requests, {CONCURRENCY} threads) ---")

    payload = {"document_text": "The stock market rallied yesterday as the tech sector showed strong quarterly earnings."}

    def send_request():
        try:
            r = session.post(API_URL, json=payload)
            return r.status_code == 200
        except Exception:
            return False

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        results = list(executor.map(lambda _: send_request(), range(CONCURRENT_BATCH)))
    duration = time.time() - start

    success = sum(results)
    tps = success / duration

    print(f"   Completed:    {success}/{CONCURRENT_BATCH} in {duration:.1f}s")
    print(f"   Throughput:   {tps:.1f} req/s")
    print(f"   Success Rate: {success/CONCURRENT_BATCH:.0%}")

    return {"tps": tps, "duration": duration}


def estimate_costs(latency_metrics, throughput_metrics):
    avg_lat = latency_metrics["avg_latency"]
    tps = throughput_metrics["tps"]

    print(f"\n--- Cost Forecast ({TARGET_DOCS:,} Documents) ---")

    # Time estimate (at measured throughput)
    seconds_needed = TARGET_DOCS / tps
    hours_needed = seconds_needed / 3600
    print(f"   Time (single node):  {hours_needed:.1f} hours")

    # Lambda cost
    total_compute_seconds = TARGET_DOCS * avg_lat
    lambda_cost = (total_compute_seconds * LAMBDA_GB_SEC_RATE * 0.5) + (TARGET_DOCS * LAMBDA_REQ_RATE)
    print(f"   AWS Lambda (512MB):  ${lambda_cost:.2f}")

    # EC2 cost
    ec2_cost = hours_needed * EC2_HOURLY_RATE
    print(f"   EC2 (t3.medium):     ${ec2_cost:.2f}")

    print(f"\n   Recommendation: {'Lambda' if lambda_cost < ec2_cost else 'EC2'} is cheaper for this volume.")


if __name__ == "__main__":
    try:
        session = requests.Session()

        # Verify API is running
        r = session.get(HEALTH_URL)
        r.raise_for_status()
        health = r.json()
        print(f"✅ API healthy | Model: {health['model_version']} | Cache: {'on' if health['cache_enabled'] else 'off'}\n")

        lat = measure_sequential_latency(session)
        tp = measure_throughput(session)

        if lat and tp:
            estimate_costs(lat, tp)

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect. Is the API running on port 8000?")
        print("   Run: uvicorn app.main:app --port 8000")
