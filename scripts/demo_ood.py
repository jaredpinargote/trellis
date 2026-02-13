"""
Demo: Out-Of-Distribution (OOD) detection.
Shows that nonsense, off-topic, and adversarial texts are rejected as 'other'.

Usage: python scripts/demo_ood.py
  (requires API running on localhost:8000)
"""
import requests
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"

OOD_SAMPLES = [
    ("Random gibberish",         "Qwerty plmk zxcv bnmq asdf ghjkl poiuy trewq."),
    ("Single word",              "Hello"),
    ("Numbers only",             "123456789 987654321 555 777 999"),
    ("Foreign language",         "日本語のテスト文章です。これは分類されるべきではありません。"),
    ("Emoji spam",               "🎉🎊🎈🎁🎀🎉🎊🎈🎁🎀🎉🎊🎈"),
    ("Cooking recipe (edge)",    "Mix flour and eggs, bake at 350F for 30 minutes until golden brown."),
    ("Weather report",           "Tomorrow will be partly cloudy with a high of 72 degrees Fahrenheit and a 30 percent chance of afternoon thunderstorms."),
    ("Philosophical",            "What is the meaning of consciousness and does free will truly exist in a deterministic universe?"),
]

IN_DOMAIN_SAMPLES = [
    ("sport",        "The Lakers defeated the Celtics 112-108 in overtime, with LeBron scoring 41 points in a thrilling NBA Finals game seven."),
    ("medical",      "The FDA approved a new immunotherapy drug for stage IV lung cancer patients who have not responded to conventional chemotherapy treatments."),
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
    print("DEMO: Out-Of-Distribution (OOD) Detection")
    print("=" * 80)
    print(f"  Model: {r.json()['model_version']} | OOD Threshold: {r.json()['model_threshold']:.4f}\n")

    print("── OUT-OF-DOMAIN TEXTS (should be rejected → 'other') ──")
    print(f"{'Description':<25s} {'Label':<12s} {'Conf':>6s} {'OOD':>5s}  Text Preview")
    print("─" * 80)
    for desc, text in OOD_SAMPLES:
        r = requests.post(f"{API}/classify_document", json={"document_text": text})
        d = r.json()
        icon = "✅" if d['is_ood'] else "⚠️ "
        print(f"{icon} {desc:<23s} {d['label']:<12s} {d['confidence']:>5.1%} {str(d['is_ood']):>5s}  {text[:40]}...")

    print(f"\n── IN-DOMAIN TEXTS (should be classified correctly) ──")
    print(f"{'Expected':<25s} {'Label':<12s} {'Conf':>6s} {'OOD':>5s}  Text Preview")
    print("─" * 80)
    for expected, text in IN_DOMAIN_SAMPLES:
        r = requests.post(f"{API}/classify_document", json={"document_text": text})
        d = r.json()
        icon = "✅" if d['label'] == expected else "❌"
        print(f"{icon} {expected:<23s} {d['label']:<12s} {d['confidence']:>5.1%} {str(d['is_ood']):>5s}  {text[:40]}...")

    print("\n  Done.\n")


if __name__ == '__main__':
    main()
