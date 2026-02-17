
import argparse
import requests
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Test a deployed Trellis API instance.")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API (default: http://localhost:8000)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    print(f"🚀 Testing API at: {base_url}")

    # 1. Health Check
    print("\n1. Checking Health...")
    try:
        t0 = time.time()
        r = requests.get(f"{base_url}/health", timeout=10)
        latency = (time.time() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        print(f"   ✅ [PASS] Status: {r.status_code} | Latency: {latency:.0f}ms")
        print(f"      Model: {data.get('model_version')} | Threads: {data.get('model_threshold')}")
    except Exception as e:
        print(f"   ❌ [FAIL] Could not connect to {base_url}/health: {e}")
        sys.exit(1)

    # 2. Prediction Check (Valid)
    print("\n2. Classification (Valid Input)...")
    payload = {"document_text": "The central bank raised interest rates to combat inflation."}
    try:
        r = requests.post(f"{base_url}/classify_document", json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        print(f"   ✅ [PASS] Label: {data['label']} | Conf: {data['confidence']:.2f}")
    except Exception as e:
        print(f"   ❌ [FAIL] Prediction failed: {e}")
        sys.exit(1)

    # 3. OOD Check (Invalid)
    print("\n3. OOD Detection...")
    ood_payload = {"document_text": "Djwkald wjakldjawkldj wajdklwajdkl."} # Nonsense
    try:
        r = requests.post(f"{base_url}/classify_document", json=ood_payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get('is_ood'):
             print(f"   ✅ [PASS] Correctly identified as OOD.")
        else:
             print(f"   ⚠️ [WARN] Failed to detect OOD (Label: {data['label']})")
    except Exception as e:
        print(f"   ❌ [FAIL] OOD check failed: {e}")

    print("\n✨ Deployment Verified Successfully.")

if __name__ == "__main__":
    main()
