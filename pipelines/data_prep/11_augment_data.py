"""
Augment Training Data: Solve Length Mismatch
============================================
The model fails on short queries because it was trained on full articles.
This script slices the training documents into paragraphs and sentences
to create a dataset that includes short, high-confidence examples.

Strategy:
1. Load train.csv.
2. For each document:
   - Keep the original (for long-form accuracy).
   - Extract paragraphs (splitting by double newline).
   - Extract sentences (splitting by period).
3. Filter segments:
   - Paragraphs > 5 words.
   - Sentences > 3 words (aggressive for short queries like "Stock market today").
4. Save to data/training/train_augmented.csv.
"""
import os
import sys
import pandas as pd
import re

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATA_DIR = 'data/training'
OUTPUT_FILE = os.path.join(DATA_DIR, 'train_augmented.csv')

def augment_text(df):
    augmented_rows = []
    
    print(f"Original Docs: {len(df)}")
    
    for _, row in df.iterrows():
        text = str(row['text'])
        category = row['category']
        
        # 1. Keep Original
        augmented_rows.append({'text': text, 'category': category, 'source': 'original'})
        
        # 2. Paragraphs (BBC format often double newline)
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.split()) > 5]
        for p in paragraphs:
            augmented_rows.append({'text': p, 'category': category, 'source': 'paragraph'})
            
        # 3. Sentences (Simple splitting)
        # Using regex to handle ., !, ?
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = [s.strip() for s in sentences if len(s.split()) > 3]
        for s in sentences:
            augmented_rows.append({'text': s, 'category': category, 'source': 'sentence'})
            
    aug_df = pd.DataFrame(augmented_rows)
    # Remove duplicates
    aug_df.drop_duplicates(subset=['text'], inplace=True)
    
    return aug_df

def main():
    train_path = os.path.join(DATA_DIR, 'train.csv')
    if not os.path.exists(train_path):
        print("Error: train.csv not found.")
        sys.exit(1)
        
    print("Loading data...")
    df = pd.read_csv(train_path)
    
    print("Augmenting data...")
    aug_df = augment_text(df)
    
    print("-" * 40)
    print(f"Resulting Dataset Size: {len(aug_df)}")
    print(f"Sources Breakdown:\n{aug_df['source'].value_counts()}")
    print("-" * 40)
    
    # Save (drop 'source' column for training compatibility)
    final_df = aug_df[['text', 'category']]
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
