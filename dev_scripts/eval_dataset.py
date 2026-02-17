"""
Evaluate the deployed model against the held-out test set.

Usage:
    python scripts/eval_dataset.py                          # localhost
    python scripts/eval_dataset.py --url https://...        # remote
    python scripts/eval_dataset.py --dataset data/training/train.csv  # different split
"""
import argparse
import sys
import time
import pandas as pd
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

def classify_one(base_url: str, text: str, true_label: str) -> dict:
    """Send a single classification request. Returns result dict."""
    try:
        r = requests.post(
            f"{base_url}/classify_document",
            json={"document_text": text[:5000]},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return {"true": true_label, "pred": data["label"], "ood": data.get("is_ood", False), "error": False}
    except Exception:
        return {"true": true_label, "pred": None, "ood": False, "error": True}

def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy on a CSV dataset.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--dataset", default="data/training/test.csv", help="Path to CSV with 'text' and 'category' columns")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to evaluate (for speed)")
    parser.add_argument("--workers", type=int, default=20, help="Concurrent workers (default: 20)")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    df = pd.read_csv(args.dataset)

    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=42)

    total = len(df)
    correct = 0
    ood_count = 0
    errors = 0
    mismatches: list[dict] = []
    t0 = time.time()

    print(f"📊 Evaluating {total} samples from {args.dataset}")
    print(f"   API: {base_url} | Workers: {args.workers}\n")

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(classify_one, base_url, str(row["text"]), row["category"]): i
            for i, row in df.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            done += 1

            if result["error"]:
                errors += 1
            elif result["ood"]:
                ood_count += 1
            elif result["pred"] == result["true"]:
                correct += 1
            else:
                mismatches.append(result)

            if done % 50 == 0:
                elapsed = time.time() - t0
                print(f"   ... {done}/{total} ({done/total*100:.0f}%) — {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    classified = total - ood_count - errors
    accuracy = (correct / classified * 100) if classified > 0 else 0

    print(f"\n{'='*50}")
    print(f"📈 RESULTS")
    print(f"{'='*50}")
    print(f"   Total Samples:       {total}")
    print(f"   Correctly Classified: {correct}/{classified} ({accuracy:.1f}%)")
    print(f"   OOD Rejected:        {ood_count} ({ood_count/total*100:.1f}%)")
    print(f"   Errors:              {errors}")
    print(f"   Time:                {elapsed:.1f}s ({total/elapsed:.0f} samples/sec)")

    if mismatches:
        confusion = Counter((m["true"], m["pred"]) for m in mismatches)
        print(f"\n   Top Misclassifications:")
        for (true, pred), count in confusion.most_common(5):
            print(f"     {true} → {pred}: {count}")

    print()
    sys.exit(0 if accuracy >= 90 else 1)

if __name__ == "__main__":
    main()
