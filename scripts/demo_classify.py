"""
Demo: Classify documents from each category.
Shows the model working across all 10 categories + OOD.

Usage: python scripts/demo_classify.py
  (requires API running on localhost:8000)
"""
import requests
import sys

# Force UTF-8 output for Windows to support emojis/box-drawing
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"

SAMPLES = [
    ("business",        "Global stock markets surged today as quarterly earnings reports exceeded analyst expectations across the technology and finance sectors."),
    ("entertainment",   "The latest Marvel movie broke box office records this weekend, earning $350 million domestically in its opening three days."),
    ("food",            "A traditional Italian risotto requires Arborio rice, fresh Parmesan cheese, chicken broth, and white wine for the perfect creamy texture."),
    ("graphics",        "OpenGL shaders allow real-time rendering of 3D scenes using vertex and fragment programs that execute on the GPU pipeline."),
    ("historical",      "The Roman Empire reached its maximum territorial extent under Emperor Trajan in 117 AD, spanning from Britain to Mesopotamia."),
    ("medical",         "Researchers at Johns Hopkins published a clinical trial showing that mRNA vaccines reduced severe COVID-19 hospitalizations by 94 percent."),
    ("politics",        "The Senate passed the bipartisan infrastructure bill with a vote of 69 to 30, sending it to the House for final approval."),
    ("space",           "NASA's James Webb Space Telescope captured unprecedented images of distant galaxies formed just 300 million years after the Big Bang."),
    ("sport",           "Lionel Messi scored a hat-trick in the Champions League final, leading his team to a dramatic 4-3 victory over Manchester City."),
    ("technologie",     "Apple unveiled the M3 chip with a 3-nanometer process, delivering 40 percent faster CPU performance and 65 percent faster GPU."),
    ("other (OOD)",     "Wxy plmk zrtq bvnf. Qkjh mnop wertyuiop asdfghjkl."),
]


def main():
    # Check API is running
    try:
        r = requests.get(f"{API}/health", timeout=3)
        r.raise_for_status()
    except Exception as e:
        print(f"❌ API not reachable at {API}: {e}")
        print("   Start it with: python scripts/run_api.py")
        sys.exit(1)

    health = r.json()
    print("=" * 80)
    print("DEMO: Document Classification Across All Categories")
    print("=" * 80)
    print(f"  Model: {health['model_version']} | Threshold: {health['model_threshold']:.4f}")
    print()

    print(f"{'Expected':<16s} {'Predicted':<16s} {'Conf':>6s} {'OOD':>5s} {'ms':>6s}  Text Preview")
    print("─" * 80)

    correct = 0
    total = len(SAMPLES)

    for expected, text in SAMPLES:
        r = requests.post(f"{API}/classify_document", json={"document_text": text})
        d = r.json()
        match = "✅" if d['label'] == expected.split(" ")[0] else "❌"
        if d['label'] == expected.split(" ")[0]:
            correct += 1
        print(f"{expected:<16s} {d['label']:<16s} {d['confidence']:>5.1%} {str(d['is_ood']):>5s} {d['processing_time_ms']:>5.1f}ms  {text[:50]}...")

    print(f"\n  Accuracy: {correct}/{total} ({correct/total:.0%})")
    print("  Done.\n")


if __name__ == '__main__':
    main()
