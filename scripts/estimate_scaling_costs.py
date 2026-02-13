import time
import requests
import statistics
import concurrent.futures

# --- Configuration ---
API_URL = "http://localhost:8000/classify_document"
BATCH_SIZE = 100  # Number of requests for benchmark
TARGET_DOCS = 1_000_000

# Cloud Cost Assumptions (Approximate Retail Rates)
# AWS Lambda (x86, 512MB RAM): ~$0.0000008333 per 100ms * GB-second
# Request charge: $0.20 per 1M requests
LAMBDA_GB_SEC_RATE = 0.0000166667 # per GB-second
LAMBDA_REQ_RATE = 0.20 / 1_000_000

# EC2 t3.medium (2 vCPU, 4GB RAM): ~$0.0416 per hour
EC2_HOURLY_RATE = 0.0416

def benchmark_api():
    print(f"--- Benchmarking API ({BATCH_SIZE} requests) ---")
    
    payload = {"document_text": "The stock market rallied yesterday as the tech sector showed strong quarterly earnings."}
    
    times = []
    success_count = 0
    
    start_global = time.time()
    
    # Simple synchronous benchmark for baseline latency
    # For throughput, we'd ideally use async/threads, let's use a small thread pool
    # to simulate "realistic" ingestion client.
    
    # Define a task that measures its own latency
    def measured_request():
        t0 = time.time()
        try:
            r = requests.post(API_URL, json=payload)
            lat = time.time() - t0
            return r, lat
        except Exception as e:
            return None, 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(measured_request) for _ in range(BATCH_SIZE)]
        
        for f in concurrent.futures.as_completed(futures):
            try:
                resp, lat = f.result()
                if resp and resp.status_code == 200:
                    success_count += 1
                    times.append(lat)
            except Exception as e:
                print(f"Request failed: {e}")

    total_time = time.time() - start_global
    
    if not times:
        print("❌ Benchmark failed. Is the API running?")
        return None

    avg_latency = statistics.mean(times)
    p99_latency = sorted(times)[int(len(times)*0.99)] if len(times) >= 100 else max(times)
    tps = success_count / total_time
    
    print(f"[PASS] Benchmark Complete.")
    print(f"   Avg Latency: {avg_latency*1000:.2f} ms")
    print(f"   Est. Throughput (local): {tps:.2f} docs/sec")
    
    return {
        "tps": tps,
        "avg_latency": avg_latency
    }

def estimate_costs(metrics):
    tps = metrics["tps"]
    avg_lat = metrics["avg_latency"]
    
    print(f"\n--- Cost & Time Forecast (1,000,000 Documents) ---")
    
    # 1. Time
    # If we sustain this TPS
    seconds_needed = TARGET_DOCS / tps
    hours_needed = seconds_needed / 3600
    print(f"[TIME] Time to Process 1M Docs (Single Node): {hours_needed:.2f} hours")
    
    # 2. AWS Lambda Cost
    # Cost = (Total Requests * Req Rate) + (Total Compute Seconds * GB-Sec Rate * MemoryGB)
    # Assuming 512MB RAM (0.5 GB)
    # Compute Seconds = 1M * avg_latency
    total_compute_seconds = TARGET_DOCS * avg_lat
    lambda_compute_cost = total_compute_seconds * LAMBDA_GB_SEC_RATE * 0.5
    lambda_req_cost = TARGET_DOCS * LAMBDA_REQ_RATE
    total_lambda = lambda_compute_cost + lambda_req_cost
    
    print(f"[COST] Est. AWS Lambda Cost (Serverless): ${total_lambda:.2f}")
    print(f"   (Assumes 512MB RAM, avg latency {avg_lat*1000:.1f}ms per doc)")

    # 3. AWS EC2 Cost
    # Cost = Hours Needed * Hourly Rate
    # Note: This implies we run perfectly for N hours. 
    # In reality, we could scale horizontally. 
    # E.g. 10 instances = 1/10th time, but same total cost (roughly).
    ec2_cost = hours_needed * EC2_HOURLY_RATE
    print(f"[COST] Est. EC2 Cost (t3.medium): ${ec2_cost:.2f}")
    print(f"   (Assumes single instance running for {hours_needed:.1f} hours)")
    
    print("\n--------------------------------------------------")
    print("Recommendation:")
    if total_lambda < ec2_cost:
         print("[REC] Lambda is cheaper for this volume (and easier to scale).")
    else:
         print("[REC] EC2 is cheaper for this volume (steady state).")

if __name__ == "__main__":
    try:
        metrics = benchmark_api()
        if metrics:
            estimate_costs(metrics)
    except requests.exceptions.ConnectionError:
        print("[FAIL] Error: Could not connect to API. Is it running on port 8000?")
        print("Run: uvicorn app.main:app --port 8000")
