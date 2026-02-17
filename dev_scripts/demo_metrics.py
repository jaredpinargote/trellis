"""
Demo: Telemetry & Metrics Dashboard.
Generates traffic, then fetches /metrics to show production telemetry.

Usage: python scripts/demo_metrics.py
  (requires API running on localhost:8000)
"""
import requests
import time
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"

TRAFFIC = [
    "The quarterback threw a 60-yard touchdown pass in the championship game.",
    "Apple unveiled its latest iPhone with a revolutionary new chip architecture.",
    "Scientists discovered a new exoplanet in the habitable zone of a nearby star.",
    "The Senate voted on the new healthcare reform bill along party lines.",
    "Researchers published promising results for a new cancer immunotherapy drug.",
    "Global markets rallied as the Federal Reserve announced no interest rate hike.",
    "The new Marvel movie grossed over 200 million dollars in its opening weekend.",
    "Traditional Japanese ramen requires a rich tonkotsu broth simmered for 12 hours.",
    "GPU rendering with ray tracing transformed the quality of real-time 3D graphics.",
    "The French Revolution of 1789 fundamentally changed the political landscape of Europe.",
    # Repeat for volume
    "Manchester United signed a record-breaking transfer deal worth 150 million euros.",
    "The tech startup raised 50 million in Series B funding for its AI platform.",
    "NASA confirmed the successful landing of a probe on Jupiter's moon Europa.",
    "The prime minister called for snap elections amid the political crisis in parliament.",
    "A randomized controlled trial showed the vaccine was 95 percent effective.",
]


def main():
    try:
        r = requests.get(f"{API}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ API not reachable at {API}: {e}")
        print("   Start it with: python scripts/run_api.py")
        sys.exit(1)

    print("=" * 80)
    print("DEMO: Telemetry & Metrics Dashboard")
    print("=" * 80)

    # Generate traffic
    print(f"\n  Generating traffic ({len(TRAFFIC)} requests)...")
    for text in TRAFFIC:
        requests.post(f"{API}/classify_document", json={"document_text": text})
    # Hit one cached (duplicate)
    requests.post(f"{API}/classify_document", json={"document_text": TRAFFIC[0]})

    time.sleep(0.5)

    # Fetch metrics
    r = requests.get(f"{API}/metrics")
    m = r.json()

    print(f"\n{'─'*60}")
    print(f"  📊 PRODUCTION METRICS DASHBOARD")
    print(f"{'─'*60}")
    print(f"  Uptime:              {m['uptime_human']}")
    print(f"  Total Requests:      {m['total_requests']}")
    print(f"  Total Errors:        {m['total_errors']}")
    print(f"  Error Rate:          {m['error_rate']:.2%}")
    print(f"  Cache Hit Rate:      {m['cache_hit_rate']:.2%}")
    print(f"  Requests/sec:        {m['requests_per_second']:.1f}")
    print(f"  Model Version:       {m['model_version']}")
    print(f"  OOD Threshold:       {m['ood_threshold']:.4f}")

    print(f"\n  📈 LATENCY")
    lat = m['latency']
    print(f"    Mean:   {lat['mean_ms']:.2f}ms")
    print(f"    p50:    {lat['p50_ms']:.2f}ms")
    print(f"    p95:    {lat['p95_ms']:.2f}ms")
    print(f"    p99:    {lat['p99_ms']:.2f}ms")
    print(f"    Max:    {lat['max_ms']:.2f}ms")
    print(f"    Samples: {lat['samples']}")

    print(f"\n  🏷️  LABEL DISTRIBUTION")
    for label, count in m['label_distribution'].items():
        bar = "█" * count
        print(f"    {label:<16s} {count:>3d}  {bar}")

    print(f"\n{'─'*60}")
    print(f"  ✅ Telemetry is live and collecting.")
    print(f"     GET {API}/metrics  to see it anytime.")
    print(f"{'─'*60}\n")


if __name__ == '__main__':
    main()
