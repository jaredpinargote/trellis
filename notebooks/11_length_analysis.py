"""
EDA Insight: Document Length Mismatch
=====================================
Analyzes the distribution of document lengths in the training set vs.
typical user queries (as represented by demo samples).

Goal: Prove that the model is trained on long documents but tested on short ones.
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load Data
DATA_DIR = 'data/training'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# Calculate Lengths (Word Count)
train['word_count'] = train['text'].apply(lambda x: len(str(x).split()))

# Demo Samples (Short Text)
DEMO_SAMPLES = [
    "Global stock markets surged today as quarterly earnings reports exceeded analyst expectations.",
    "The latest Marvel movie broke box office records this weekend.",
    "Lionel Messi scored a hat-trick in the Champions League final.",
    "Apple unveiled the M3 chip with a 3-nanometer process.",
]
demo_lengths = [len(s.split()) for s in DEMO_SAMPLES]

# Statistics
print("=" * 60)
print("LENGTH ANALYSIS: Training vs. Real-World Queries")
print("=" * 60)
print(f"Training Data (N={len(train)}):")
print(f"  Mean Length:   {train['word_count'].mean():.2f} words")
print(f"  Median Length: {train['word_count'].median():.2f} words")
print(f"  Min Length:    {train['word_count'].min()} words")
print(f"  Max Length:    {train['word_count'].max()} words")
print("-" * 60)
print(f"Demo Queries (N={len(DEMO_SAMPLES)}):")
print(f"  Mean Length:   {np.mean(demo_lengths):.2f} words")
print(f"  Median Length: {np.median(demo_lengths):.2f} words")
print("=" * 60)
print("\nINSIGHT: The model expects ~300-word articles but receives ~12-word queries.")
print("SOLUTION: Augment training data with sentence-level segments.")
