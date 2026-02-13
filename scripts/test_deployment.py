"""
Test Deployment: Verify a live Trellis deployment end-to-end.
==============================================================
Usage:
    python scripts/test_deployment.py --url https://your-deployment-url.up.railway.app
"""
import argparse
import sys
import requests

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

SAMPLES = [
    ("sport", "The Olympics 100m sprint final was won by the American athlete who broke the world record."),
    ("technologie", "Apple released a new iPhone with an improved A17 processor and enhanced camera capabilities."),
    ("politics", "The Senate passed the bipartisan infrastructure bill with a vote of 69 to 30."),
    ("other", "Wxy plmk zrtq bvnf. Qkjh mnop wertyuiop asdfghjkl."),
]

GUIDE = """
🔧 Deployment Test — Configuration Required

  You need the deployment URL. Here's how to get it:

  If you're the developer:
    1. Go to https://railway.app/dashboard
    2. Select the "trellis" project
    3. Click on the service → Settings → Networking
    4. Copy the "Public URL" (e.g. https://trellis-production-XXXX.up.railway.app)

  If you're NOT the developer:
    Ask your team lead for the production URL.

  Usage:
    python scripts/test_deployment.py --url <DEPLOYMENT_URL>

  Optional flags:
    --timeout 10    Request timeout in seconds (default: 10)
    --verbose       Print full response JSON
"""


def run_tests(base_url: str, timeout: int, verbose: bool):
    url = base_url.rstrip("/")
    print(f"🚀 Testing deployment: {url}\n")

    # 1. Health Check
    print("1. Health Check...")
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        health = r.json()
        print(f"   ✅ Status: {health['status']}")
        print(f"   Model: {health['model_version']} | Threshold: {health['model_threshold']:.4f}")
        print(f"   Cache: {'enabled' if health['cache_enabled'] else 'disabled'}")
    except requests.ConnectionError:
        print(f"   ❌ Cannot connect to {url}. Is the service running?")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        sys.exit(1)

    # 2. Classification Tests
    print(f"\n2. Classification ({len(SAMPLES)} samples)...")
    print(f"   {'EXPECTED':<14} {'PREDICTED':<14} {'CONF':<6} {'RESULT'}")
    print(f"   {'-'*50}")

    correct = 0
    for expected, text in SAMPLES:
        try:
            r = requests.post(f"{url}/classify_document", json={"document_text": text}, timeout=timeout)
            r.raise_for_status()
            data = r.json()

            pred = "other(OOD)" if data["is_ood"] else data["label"]
            conf = f"{data['confidence']:.2f}"
            is_correct = (expected == "other" and data["is_ood"]) or (expected == data["label"])
            status = "✅" if is_correct else "❌"
            if is_correct:
                correct += 1

            print(f"   {expected:<14} {pred:<14} {conf:<6} {status}")
            if verbose:
                print(f"      {data}")
        except Exception as e:
            print(f"   {expected:<14} {'ERROR':<14} {'0.00':<6} ❌ {e}")

    # 3. Metrics
    print(f"\n3. Metrics Endpoint...")
    try:
        r = requests.get(f"{url}/metrics", timeout=timeout)
        r.raise_for_status()
        metrics = r.json()
        print(f"   Uptime: {metrics['uptime_human']} | Requests: {metrics['total_requests']} | RPS: {metrics['requests_per_second']}")
    except Exception as e:
        print(f"   ⚠️ Could not fetch metrics: {e}")

    # 4. Summary
    acc = correct / len(SAMPLES) if SAMPLES else 0
    print(f"\n{'='*50}")
    print(f"Accuracy: {correct}/{len(SAMPLES)} ({acc:.0%})")
    if correct == len(SAMPLES):
        print("✅ ALL TESTS PASSED — Deployment is healthy.")
    else:
        print("⚠️  Some tests failed. Review results above.")


def main():
    parser = argparse.ArgumentParser(
        description="Test a live Trellis deployment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=GUIDE
    )
    parser.add_argument("--url", type=str, help="Deployment URL (e.g. https://trellis-xyz.up.railway.app)")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds (default: 10)")
    parser.add_argument("--verbose", action="store_true", help="Print full response JSON")

    args = parser.parse_args()

    if not args.url:
        print(GUIDE)
        sys.exit(0)

    run_tests(args.url, args.timeout, args.verbose)


if __name__ == "__main__":
    main()
