import asyncio
import httpx
import time
import subprocess
import sys
import os
import signal

# Configuration
API_URL = "http://127.0.0.1:8000"
ENDPOINT = "/classify_document"
CONCURRENCY = 50
TOTAL_REQUESTS = 200
LARGE_PAYLOAD_SIZE_MB = 10

async def wait_for_api(client, retries=10, delay=1):
    """Wait for the API to be healthy."""
    for i in range(retries):
        try:
            resp = await client.get(f"{API_URL}/health")
            if resp.status_code == 200:
                print("✅ API is online.")
                return True
        except httpx.ConnectError:
            pass
        print(f"Waiting for API... ({i+1}/{retries})")
        await asyncio.sleep(delay)
    return False

async def stress_concurrency(client):
    """Test 1: High Concurrency."""
    print(f"\n--- Test 1: High Concurrency ({CONCURRENCY} concurrent reqs) ---")
    
    payload = {"document_text": "This is a standard legal document regarding contract law."}
    
    start_time = time.time()
    tasks = []
    
    # Create a semaphore to limit concurrency if needed, but we want to stress it
    sem = asyncio.Semaphore(CONCURRENCY)
    
    async def make_request():
        async with sem:
            try:
                resp = await client.post(f"{API_URL}{ENDPOINT}", json=payload)
                return resp.status_code
            except Exception as e:
                return str(e)

    for _ in range(TOTAL_REQUESTS):
        tasks.append(make_request())
        
    results = await asyncio.gather(*tasks)
    duration = time.time() - start_time
    
    success_count = results.count(200)
    print(f"Completed {TOTAL_REQUESTS} requests in {duration:.2f}s")
    print(f"TPS: {TOTAL_REQUESTS/duration:.2f}")
    print(f"Success Rate: {success_count}/{TOTAL_REQUESTS} ({success_count/TOTAL_REQUESTS:.1%})")
    
    if success_count < TOTAL_REQUESTS:
        print("⚠️ Warning: Some requests failed or dropped.")
    else:
        print("✅ Passed Concurrency Test.")

async def stress_large_payload(client):
    """Test 2: Large Payload (DoS attempt)."""
    print(f"\n--- Test 2: Large Payload ({LARGE_PAYLOAD_SIZE_MB}MB) ---")
    
    # Generate large string
    large_text = "legal " * (LARGE_PAYLOAD_SIZE_MB * 1024 * 1024 // 6) # approx size
    payload = {"document_text": large_text}
    
    try:
        start_time = time.time()
        resp = await client.post(f"{API_URL}{ENDPOINT}", json=payload, timeout=30.0)
        duration = time.time() - start_time
        
        print(f"Status Code: {resp.status_code}")
        print(f"Response Time: {duration:.2f}s")
        
        if resp.status_code == 413:
             print("✅ Passed: API correctly rejected payload too large.")
        elif resp.status_code == 422:
             print("✅ Passed: API validated input size (Validation Error).")
        elif resp.status_code == 200:
             print("⚠️ Warning: API accepted 10MB payload. Check memory usage/limits.")
        else:
             print(f"❓ Unexpected status: {resp.status_code}")
             
    except httpx.TimeoutException:
        print("❌ Failed: Request timed out. API might be hanging processing large text.")
    except Exception as e:
        print(f"❌ Error: {e}")

async def stress_malformed(client):
    """Test 3: Malformed Data."""
    print(f"\n--- Test 3: Malformed Data ---")
    
    # Case A: Invalid JSON
    try:
        resp = await client.post(f"{API_URL}{ENDPOINT}", content="{ 'bad': json }", headers={"Content-Type": "application/json"})
        print(f"Malformed JSON Status: {resp.status_code} (Expected 422 or 400)")
    except Exception as e:
        print(f"Malformed JSON Error: {e}")

    # Case B: Wrong Data Type
    try:
        resp = await client.post(f"{API_URL}{ENDPOINT}", json={"document_text": 12345})
        print(f"Wrong Type Status: {resp.status_code} (Expected 422)")
    except Exception as e:
        print(f"Wrong Type Error: {e}")

async def main():
    # 1. Start Server (if not running)
    server_process = None
    server_started_by_script = False
    
    # Check if port 8000 is open
    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{API_URL}/health")
            print("Running against existing API instance...")
        except:
            print("Target API not found. Starting local instance...")
            # Start uvicorn
            server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            server_started_by_script = True
            if not await wait_for_api(client):
                print("❌ Failed to start API.")
                if server_process: server_process.kill()
                sys.exit(1)

    # 2. Run Tests
    async with httpx.AsyncClient(timeout=10.0) as client:
        await stress_concurrency(client)
        await stress_large_payload(client)
        await stress_malformed(client)

    # 3. Cleanup
    if server_started_by_script and server_process:
        print("\nStopping temporary API instance...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
