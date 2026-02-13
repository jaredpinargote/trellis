"""
DEMO: OOD Detection (Concise)
=============================
Tests Out-of-Distribution handling with nonsense inputs.
Concise output for AI/Reviewer.
"""
import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')
API = "http://127.0.0.1:8000"

SAMPLES = [
    ("OOD:Gibberish", "Qwerty plmk zxcv bnmq asdf ghjkl poiuy treerewq"),
    ("OOD:Foreign", "日本語のテスト文章です。これは分類されるべきではありません。"),
    ("OOD:Emoji", "🎉🎊🎈🎁🎀🎉🎊🎈🎁🎀🎉🎊🎈"),
    ("OOD:Numbers", "123456789 987654321 555 777 999"),
    ("OOD:Short", "Hello world"),
    ("Valid:Medical", "The FDA approved a new immunotherapy drug for cancer treatment."),
    ("Valid:Sport", "The Lakers defeated the Celtics 112-108 in overtime."),
]

def main():
    try:
        r = requests.get(f"{API}/health", timeout=5).json()
        print(f"Model: {r['model_version']} Threshold: {r['model_threshold']:.4f}")
    except:
        sys.exit(f"API down at {API}")

    print(f"{'TYPE':<15} {'TEXT (Brief)':<20} {'CONF':<6} {'OOD?':<5} {'PRED'}")
    print("-" * 60)

    for label, text in SAMPLES:
        try:
            res = requests.post(f"{API}/classify_document", json={"document_text": text}).json()
            is_ood = res['is_ood']
            conf = res['confidence']
            pred = res['label']
            
            # Correctness Check
            expected_ood = label.startswith("OOD")
            is_correct = (is_ood == expected_ood)
            if not expected_ood and pred != label.split(":")[1].lower(): is_correct = False
            
            mark = " " if is_correct else "!"
            ood_mark = "YES" if is_ood else "NO"
            
            print(f"{mark}{label:<14} {text[:19]:<20} {conf:.2f}   {ood_mark:<5} {pred}")
            
        except Exception as e:
            print(f"!{label:<14} ERROR")

if __name__ == "__main__":
    main()
