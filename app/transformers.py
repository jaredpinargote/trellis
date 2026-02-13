"""
Custom retrieval-model transformers used by the production pipeline.
These must be importable at API startup so joblib can deserialize them.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class DFRVectorizer(BaseEstimator, TransformerMixin):
    """
    Divergence from Randomness (DFR) weighting.
    Implements the InexpB2 model:
      tf_norm = tf * c * log2(1 + avgdl/dl)
      after_effect = tf_norm / (tf_norm + 1)   [Laplace]
      score = after_effect * idf_dfr
    """
    def __init__(self, c=1.0, max_features=10000, ngram_range=(1, 1)):
        self.c = c
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, X, y=None):
        self.cv_ = CountVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range
        )
        tf_raw = self.cv_.fit_transform(X)
        n_docs = tf_raw.shape[0]
        cf = np.asarray(tf_raw.sum(axis=0)).flatten().astype(np.float64)
        self.idf_dfr_ = np.log2(1.0 + (n_docs + 1.0) / (cf + 0.5))
        self.avgdl_ = tf_raw.sum(axis=1).mean()
        return self

    def transform(self, X):
        tf_raw = self.cv_.transform(X)
        dl = np.asarray(tf_raw.sum(axis=1)).flatten()
        tf = tf_raw.toarray().astype(np.float64)
        dl_safe = np.maximum(dl, 1.0)
        tf_norm = tf * self.c * np.log2(1.0 + self.avgdl_ / dl_safe)[:, None]
        after_effect = tf_norm / (tf_norm + 1.0)
        score = after_effect * self.idf_dfr_[None, :]
        return csr_matrix(score)
