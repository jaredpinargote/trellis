"""
Data Splitting Script - Pure Python Implementation (Zero Dependencies)
1. Stratified Split
2. 'Other' Isolation
3. Speed
"""

import json
import csv
import random
import os
from collections import defaultdict
from pathlib import Path

# Configuration
INPUT_FILE = Path('data/dataset.jsonl')
OUTPUT_DIR = Path('data/training')
RANDOM_SEED = 16  # Reproducibility (got married in 2016 so also for fun :))
TRAIN_RATIO = 0.8 # 80% Train
VAL_RATIO = 0.1   # 10% Val
# Remaining 10% is Test

def load_jsonl(filepath):
    """Generator to read file line by line for memory efficiency in case the dataset was larger"""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def save_csv(filepath, data, headers=['category', 'file_name', 'text']):
    """Writes list of dicts to CSV properly handling quotes/commas."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(data)

def run_split():
    # 1. Setup
    random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading data from {INPUT_FILE}...")

    # 2. In-Memory Grouping (The "Manual Stratification" Step)
    # We group data by category to ensure we split each class evenly.
    categorized_data = defaultdict(list)
    other_data = []

    try:
        raw_data = list(load_jsonl(INPUT_FILE))
    except FileNotFoundError:
        print(f"Error: File {INPUT_FILE} not found.")
        return

    for item in raw_data:
        # Standardize keys if needed, assuming 'category' and 'text' exist
        cat = item.get('category')
        
        if cat == 'other':
            other_data.append(item)
        else:
            categorized_data[cat].append(item)

    print(f"Found {len(categorized_data)} distinct categories + 'Other'.")
    print(f"Isolating {len(other_data)} 'Other' samples for the Test set (Out of Distribution Test).")

    # 3. Perform Stratified Split
    train_set = []
    val_set = []
    test_set = []

    # Iterate over each category (Sport, Tech, etc.) to ensure distribution
    for category, items in categorized_data.items():
        # Shuffle THIS category specifically
        random.shuffle(items)
        
        n_total = len(items)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # n_test is the remainder
        
        # Slicing
        train_chunk = items[:n_train]
        val_chunk = items[n_train : n_train + n_val]
        test_chunk = items[n_train + n_val:]
        
        # Add to main buckets
        train_set.extend(train_chunk)
        val_set.extend(val_chunk)
        test_set.extend(test_chunk)

    # 4. Inject the "Other" class into Test Set ONLY
    # "Other" is for OOD testing only.
    test_set.extend(other_data)

    # 5. Final Shuffle (so the model doesn't see all of one category in a row)
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # 6. Sanity Checks
    train_cats = set(x['category'] for x in train_set)
    val_cats = set(x['category'] for x in val_set)
    test_cats = set(x['category'] for x in test_set)

    print(f"\n--- Split Summary ---")
    print(f"Train: {len(train_set)} docs | Classes: {len(train_cats)} (Should be 10)")
    print(f"Val:   {len(val_set)} docs  | Classes: {len(val_cats)} (Should be 10)")
    print(f"Test:  {len(test_set)} docs  | Classes: {len(test_cats)} (Should be 11 - incl 'other')")

    if 'other' in train_cats:
        raise ValueError("CRITICAL: 'Other' class leaked into Training set!")
    
    # 7. Save
    save_csv(OUTPUT_DIR / 'train.csv', train_set)
    save_csv(OUTPUT_DIR / 'val.csv', val_set)
    save_csv(OUTPUT_DIR / 'test.csv', test_set)
    
    print(f"\nSuccess. Data split and saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    run_split()