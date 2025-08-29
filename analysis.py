import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from utils import logger

def compute_tfidf_matrix(texts, max_features=5000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1,2), dtype=np.float32)
    X = vec.fit_transform(texts)
    logger.info("TF-IDF matrix %s", X.shape)
    return X, vec

def reduce_dimensionality(X, n_components=100):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    return svd.fit_transform(X), svd

def weighted_signal_from_matrix(X, weights):
    if sparse.issparse(X):
        W = sparse.diags(weights)
        agg = np.array(W.dot(X).sum(axis=0)).ravel()
    else:
        agg = (X * weights[:,None]).sum(axis=0)
    return agg / (weights.sum() + 1e-9)
