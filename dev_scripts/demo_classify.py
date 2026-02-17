"""
DEMO: Document Classification (Concise)
=======================================
Classifies sample documents and prints a token-efficient report.
Focuses on correctness and confidence for AI/Reviewer consumption.
"""
import requests
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

API = "http://127.0.0.1:8000"

SAMPLES = [
    ("business", "Global stock markets surged today as quarterly earnings reports exceeded analyst expectations across the technology and finance sectors."),
    ("entertainment", "The latest Marvel movie broke box office records this weekend, grossing over $200 million globally."),
    ("food", "A traditional Italian risotto requires Arborio rice, saffron, and careful stirring to achieve the perfect creamy texture."),
    ("graphics", "OpenGL shaders allow real-time rendering of 3D scenes with complex lighting and shadow effects."),
    ("historical", "The Roman Empire reached its maximum territorial extent under Trajan in 117 AD."),
    ("medical", "Researchers at Johns Hopkins published a clinical trial showing that mRNA vaccines reduced severe cases."),
    ("politics", "The Senate passed the bipartisan infrastructure bill with a vote of 69 to 30 sending it to the House for final approval."),
    ("space", "NASA's James Webb Space Telescope captured unprecedented images of distant galaxies formed shortly after the Big Bang."),
    ("sport", "Lionel Messi scored a hat-trick in the Champions League final, leading his team to a dramatic 4-3 victory over Manchester City."),
    ("technologie", "Apple unveiled the M3 chip with a 3-nanometer process, promising 20% faster CPU performance."),
    ("other", "Wxy plmk zrtq bvnf. Qkjh mnop wertyuiop asdfghjkl."),
]

def main():
    try:
        health = requests.get(f"{API}/health", timeout=5).json()
        print(f"Model: {health['model_version']} | Threshold: {health['model_threshold']:.4f}")
    except Exception as e:
        print(f"ERROR: API not reachable at {API} ({e})")
        sys.exit(1)

    print(f"{'EXPECTED':<14} {'PREDICTED':<14} {'CONF':<6} {'TEXT (Truncated)'}")
    print("-" * 65)

    correct = 0
    total = len(SAMPLES)

    for expected, text in SAMPLES:
        try:
            r = requests.post(f"{API}/classify_document", json={"document_text": text})
            r.raise_for_status()
            data = r.json()
            
            pred = data['label']
            if data['is_ood']:
                pred = "other(OOD)"
                
            conf = f"{data['confidence']:.2f}"
            is_correct = (expected == 'other' and data['is_ood']) or (expected == pred)
            
            status = " " if is_correct else "!"
            if is_correct: correct += 1
            
            # Print concise row
            print(f"{status}{expected:<13} {pred:<14} {conf:<6} {text[:30]}...")
            
        except Exception as e:
            print(f"!{expected:<13} ERROR          0.00   {str(e)[:30]}")

    acc = correct / total
    print("-" * 65)
    print(f"Accuracy: {correct}/{total} ({acc:.0%})")

if __name__ == "__main__":
    main()
