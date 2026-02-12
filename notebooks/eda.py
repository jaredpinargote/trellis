"""
Exploratory Data Analysis (EDA) for Document Classification
============================================================
This script justifies the technical decisions made in the project:
1. Class distribution → justifies stratified splitting & OOD handling
2. Document length distribution → justifies max_length=5000
3. TF-IDF top terms per class → justifies TF-IDF as a strong approach
4. Hyperparameter search results → justifies final model selection

Run: python notebooks/eda.py
Output: PNG charts saved to notebooks/ directory
"""
import pandas as pd
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_DIR = 'notebooks'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load Data ──────────────────────────────────────────────────────
train = pd.read_csv('data/training/train.csv')
val = pd.read_csv('data/training/val.csv')
test = pd.read_csv('data/training/test.csv')
all_data = pd.concat([train, val, test], ignore_index=True)

print(f"Total samples: {len(all_data)} (Train: {len(train)}, Val: {len(val)}, Test: {len(test)})")
print(f"Categories: {sorted(all_data['category'].unique())}")

# ── 1. Class Distribution ─────────────────────────────────────────
print("\n1. CLASS DISTRIBUTION")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, df) in zip(axes, [("Train", train), ("Validation", val), ("Test", test)]):
    counts = df['category'].value_counts().sort_index()
    colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor='gray', linewidth=0.5)
    ax.set_title(f'{name} Set (n={len(df)})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Count')
    for bar, val_count in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(val_count), va='center', fontsize=9)

plt.suptitle('Class Distribution Across Splits', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved class_distribution.png")

# ── 2. Document Length Distribution ────────────────────────────────
print("\n2. DOCUMENT LENGTH DISTRIBUTION")
all_data['text_length'] = all_data['text'].str.len()
all_data['word_count'] = all_data['text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Character length
for cat in sorted(all_data['category'].unique()):
    subset = all_data[all_data['category'] == cat]
    axes[0].hist(subset['text_length'], bins=30, alpha=0.5, label=cat)
axes[0].set_title('Character Length by Category', fontweight='bold')
axes[0].set_xlabel('Character Length')
axes[0].set_ylabel('Count')
axes[0].legend(fontsize=7, loc='upper right')
axes[0].axvline(x=5000, color='red', linestyle='--', alpha=0.7, label='max_length=5000')

# Word count
for cat in sorted(all_data['category'].unique()):
    subset = all_data[all_data['category'] == cat]
    axes[1].hist(subset['word_count'], bins=30, alpha=0.5, label=cat)
axes[1].set_title('Word Count by Category', fontweight='bold')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Count')
axes[1].legend(fontsize=7, loc='upper right')

plt.suptitle('Document Length Analysis', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'document_lengths.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved document_lengths.png")

# Print length stats
print("\n  Length Statistics:")
for cat in sorted(all_data['category'].unique()):
    subset = all_data[all_data['category'] == cat]
    print(f"    {cat:15s}: mean={subset['text_length'].mean():6.0f} chars, "
          f"max={subset['text_length'].max():5d}, "
          f"words_mean={subset['word_count'].mean():.0f}")

# ── 3. TF-IDF Top Terms Per Class ──────────────────────────────────
print("\n3. TF-IDF TOP TERMS PER CLASS")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))
X = tfidf.fit_transform(train['text'])
feature_names = np.array(tfidf.get_feature_names_out())

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
categories = sorted(train['category'].unique())

for i, cat in enumerate(categories):
    mask = (train['category'] == cat).values
    cat_mean = np.asarray(X[mask].mean(axis=0)).flatten()
    top_indices = cat_mean.argsort()[-10:][::-1]
    top_terms = feature_names[top_indices]
    top_scores = cat_mean[top_indices]

    axes[i].barh(range(10), top_scores, color=plt.cm.Set3(i/len(categories)))
    axes[i].set_yticks(range(10))
    axes[i].set_yticklabels(top_terms, fontsize=8)
    axes[i].set_title(cat, fontweight='bold', fontsize=11)
    axes[i].invert_yaxis()

# Hide unused subplots
for j in range(len(categories), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Top 10 TF-IDF Terms Per Category', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'tfidf_top_terms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved tfidf_top_terms.png")

# ── 4. "Other" Class Analysis (OOD Justification) ──────────────────
print("\n4. OOD / 'OTHER' CLASS ANALYSIS")
other_count = len(all_data[all_data['category'] == 'other'])
non_other_count = len(all_data[all_data['category'] != 'other'])
print(f"  'other' samples: {other_count} ({100*other_count/len(all_data):.1f}%)")
print(f"  non-'other' samples: {non_other_count} ({100*non_other_count/len(all_data):.1f}%)")
print(f"  → Severe class imbalance justifies treating 'other' as OOD problem")

fig, ax = plt.subplots(figsize=(8, 5))
cats = all_data['category'].value_counts().sort_values()
colors = ['#ff6b6b' if c == 'other' else '#4ecdc4' for c in cats.index]
cats.plot.barh(ax=ax, color=colors, edgecolor='gray', linewidth=0.5)
ax.set_title("Class Imbalance: 'Other' vs Rest", fontweight='bold', fontsize=13)
ax.set_xlabel('Sample Count')
# Annotate
for i, (val_c, cat) in enumerate(zip(cats.values, cats.index)):
    label = f" {val_c}" + (" ← OOD" if cat == 'other' else "")
    ax.text(val_c + 1, i, label, va='center', fontsize=9, fontweight='bold' if cat == 'other' else 'normal')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ood_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  → Saved ood_analysis.png")

# ── 5. Hyperparameter Search Results Visualization ──────────────────
print("\n5. HYPERPARAMETER SEARCH RESULTS")
results_path = os.path.join('models', 'results.json')
if os.path.exists(results_path):
    with open(results_path) as f:
        results = json.load(f)

    versions = results['all_versions']
    names = [v['version'] for v in versions]
    test_f1s = [v['test_f1_weighted'] for v in versions]
    val_f1s = [v['val_f1_weighted'] for v in versions]
    ood_recalls = [v['ood_recall'] for v in versions]
    train_times = [v['train_time_sec'] for v in versions]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # F1 scores comparison
    x = np.arange(len(names))
    w = 0.35
    axes[0,0].bar(x - w/2, val_f1s, w, label='Val F1', color='#3498db', alpha=0.8)
    axes[0,0].bar(x + w/2, test_f1s, w, label='Test F1', color='#e74c3c', alpha=0.8)
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    axes[0,0].set_ylabel('Weighted F1')
    axes[0,0].set_title('Validation vs Test F1 Score', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].set_ylim(0.85, 1.0)

    # OOD Recall
    colors_ood = ['#2ecc71' if o >= 0.8 else '#e74c3c' for o in ood_recalls]
    axes[0,1].bar(x, ood_recalls, color=colors_ood, alpha=0.8, edgecolor='gray')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    axes[0,1].set_ylabel('OOD Recall')
    axes[0,1].set_title('OOD Detection Recall', fontweight='bold')
    axes[0,1].axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    axes[0,1].set_ylim(0, 1.1)

    # Training time
    axes[1,0].bar(x, train_times, color='#9b59b6', alpha=0.8, edgecolor='gray')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    axes[1,0].set_ylabel('Seconds')
    axes[1,0].set_title('Training Time', fontweight='bold')

    # Combined score (F1 * OOD Recall)
    combined = [f1 * ood for f1, ood in zip(test_f1s, ood_recalls)]
    best_idx = np.argmax(combined)
    bar_colors = ['#f39c12' if i != best_idx else '#27ae60' for i in range(len(combined))]
    axes[1,1].bar(x, combined, color=bar_colors, alpha=0.8, edgecolor='gray')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels([n.replace('_', '\n') for n in names], fontsize=7)
    axes[1,1].set_ylabel('F1 × OOD Recall')
    axes[1,1].set_title(f'Combined Score (Winner: {names[best_idx]})', fontweight='bold', color='green')

    plt.suptitle(f"Hyperparameter Search: 10 TF-IDF Configurations\nWinner: {results['winner']} (F1={results['winner_test_f1']})",
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'hyperparam_search.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  → Saved hyperparam_search.png")

    # Print leaderboard
    print("\n  LEADERBOARD:")
    sorted_versions = sorted(versions, key=lambda v: v['test_f1_weighted'], reverse=True)
    for rank, v in enumerate(sorted_versions, 1):
        marker = " ← WINNER" if v['version'] == results['winner'] else ""
        print(f"    #{rank}: {v['version']:25s} F1={v['test_f1_weighted']:.4f}  OOD={v['ood_recall']:.2%}  Time={v['train_time_sec']:.2f}s{marker}")
else:
    print("  ⚠ models/results.json not found, skipping hyperparam visualization.")

print("\n" + "="*60)
print("EDA COMPLETE — All charts saved to notebooks/")
print("="*60)
