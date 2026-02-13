"""
Demo: Security & Input Sanitization.
Shows that HTML, script injection, SQL injection, and path traversal
are handled safely without crashing the model.

Usage: python scripts/demo_security.py
  (requires API running on localhost:8000)
"""
import requests
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"

ATTACK_VECTORS = [
    ("HTML injection",
     "<b>Bold</b> <i>italic</i> stock market gains <a href='evil.com'>click here</a>"),
    ("Script injection",
     "<script>alert('xss')</script> NASA launched a new satellite into orbit around Mars."),
    ("SQL injection",
     "'; DROP TABLE documents; -- The president announced new healthcare policy reforms."),
    ("Path traversal",
     "../../../../etc/passwd The genome sequencing technique revolutionized medical diagnostics."),
    ("Null bytes",
     "Sports\x00news: The team won\x00 the championship\x00 game in overtime."),
    ("Control characters",
     "Technology\x01\x02\x03 advances in quantum\x04 computing\x05 are accelerating."),
    ("Unicode attack",
     "Ṫḧḕ ṗṛḕṡḭḋḕṅṫ signed a new trade agreement with European nations today."),
    ("Oversized spaces",
     "The     stock     market     crashed     today     amid     fears     of     recession."),
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
    print("DEMO: Security & Input Sanitization")
    print("=" * 80)
    print("  All attack vectors should be sanitized safely, producing valid predictions.\n")

    print(f"{'Attack Type':<22s} {'Status':>8s} {'Label':<14s} {'Conf':>6s} {'HTTP':>5s}  Input Preview")
    print("─" * 80)

    all_safe = True
    for attack_type, text in ATTACK_VECTORS:
        try:
            r = requests.post(f"{API}/classify_document", json={"document_text": text})
            if r.status_code == 200:
                d = r.json()
                print(f"  {attack_type:<20s} {'✅ SAFE':>8s} {d['label']:<14s} {d['confidence']:>5.1%} {r.status_code:>5d}  {repr(text[:40])}...")
            elif r.status_code == 422:
                print(f"  {attack_type:<20s} {'✅ BLOCK':>8s} {'(rejected)':<14s} {'n/a':>6s} {r.status_code:>5d}  {repr(text[:40])}...")
            else:
                print(f"  {attack_type:<20s} {'❌ FAIL':>8s} {'error':<14s} {'n/a':>6s} {r.status_code:>5d}  {repr(text[:40])}...")
                all_safe = False
        except Exception as e:
            print(f"  {attack_type:<20s} {'❌ ERR':>8s} {str(e)[:30]}")
            all_safe = False

    print()
    if all_safe:
        print("  ✅ ALL ATTACK VECTORS HANDLED SAFELY — no crashes, no 500 errors.")
    else:
        print("  ❌ Some vectors caused failures — investigate.")
    print()


if __name__ == '__main__':
    main()
