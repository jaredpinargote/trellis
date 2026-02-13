import os
import sys
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from app.transformers import DFRVectorizer

# Demo texts
SAMPLES = [
    ("business", "Global stock markets surged today as quarterly earnings reports exceeded analyst expectations."),
    ("medical", "Researchers at Johns Hopkins published a clinical trial showing that mRNA vaccines reduced severe cases."),
    ("sport", "Lionel Messi scored a hat-trick in the Champions League final, leading his team to victory."),
    ("OOD", "Wxy plmk zrtq bvnf. Qkjh mnop wertyuiop asdfghjkl."),
    ("OOD_real", "Hello world this is a test."),
]

def main():
    df = pd.read_csv('data/training/train.csv')
    X, y = df['text'], df['category']

    # TF-IDF (Best Optuna params)
    tfidf = Pipeline([
        ('vec', TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True)),
        ('clf', SGDClassifier(alpha=4e-5, loss='modified_huber', random_state=42))
    ])
    
    # DFR (Best Optuna params)
    dfr = Pipeline([
        ('vec', DFRVectorizer(c=1.0, max_features=5000, ngram_range=(1,1))),
        ('clf', SGDClassifier(alpha=1e-4, loss='log_loss', random_state=42))
    ])

    print("Training TF-IDF...")
    tfidf.fit(X, y)
    print("Training DFR...")
    dfr.fit(X, y)

    print(f"\n{'Text':<20} {'TF-IDF Conf':<12} {'DFR Conf':<12}")
    print("-" * 50)
    
    for cat, text in SAMPLES:
        # TF-IDF
        prob_t = tfidf.predict_proba([text])[0]
        conf_t = np.max(prob_t)
        pred_t = tfidf.classes_[np.argmax(prob_t)]
        
        # DFR
        prob_d = dfr.predict_proba([text])[0]
        conf_d = np.max(prob_d)
        pred_d = dfr.classes_[np.argmax(prob_d)]
        
        print(f"{cat[:10]:<20} {conf_t:.4f} ({pred_t[:3]})  {conf_d:.4f} ({pred_d[:3]})")

if __name__ == "__main__":
    main()
