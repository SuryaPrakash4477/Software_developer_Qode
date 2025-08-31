import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed
import config
from utils import logger

# ---------- Vectorization ----------

def _hashing_tfidf(texts, n_features, n_jobs):
    """HashingVectorizer (no vocab in RAM) + TfidfTransformer (IDF)"""
    hv = HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2)
    )
    X_counts = hv.transform(texts)          # sparse matrix
    tfidf = TfidfTransformer()
    X = tfidf.fit_transform(X_counts)
    return X, {"type": "hashing", "n_features": n_features}

def _tfidf(texts, max_features):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), dtype=np.float32)
    X = vec.fit_transform(texts)
    return X, {"type": "tfidf", "vocab_size": len(vec.vocabulary_)}

def vectorize_texts(texts):
    """Choose memory-efficient hashing TF-IDF or classic TF-IDF."""
    if config.USE_HASHING_VECTOR:
        X, meta = _hashing_tfidf(texts, config.HASHING_N_FEATURES, config.N_JOBS)
    else:
        X, meta = _tfidf(texts, config.TFIDF_MAX_FEATURES)
    logger.info("Vectorized texts -> shape=%s, meta=%s", X.shape, meta)
    return X, meta

# ---------- Dimensionality reduction ----------

def reduce_dimensionality(X, n_components=config.N_COMPONENTS):
    svd = TruncatedSVD(
        n_components=min(n_components, min(X.shape) - 1),
        random_state=42,
        n_iter=5
    )
    Z = svd.fit_transform(X)
    logger.info("SVD reduced: %s -> %s (explained var sum=%.4f)",
                X.shape, Z.shape, svd.explained_variance_ratio_.sum())
    return Z, svd

# ---------- Scoring + aggregation ----------

def compute_per_tweet_scores(Z):
    """
    Turn an embedding Z (n_samples x k) into a single per-tweet scalar score.
    We use the first component (market "momentum" axis) as a lightweight signal.
    """
    if Z.ndim != 2 or Z.shape[1] == 0:
        raise ValueError("Reduced matrix Z has no components.")
    scores = Z[:, 0]
    return scores

def _safe_to_float(series, default=0.0):
    # convert possibly string counts (like "1,234") to float
    return pd.to_numeric(series.astype(str).str.replace(",", ""), errors="coerce").fillna(default).to_numpy()

def compute_weights(df):
    """
    Engagement-aware weights (optional). If columns are missing, fall back to 1.0.
    """
    w = np.ones(len(df), dtype=np.float32)
    if "replies" in df.columns:
        w += config.WEIGHT_REPLIES * _safe_to_float(df["replies"])
    if "retweets" in df.columns:
        w += config.WEIGHT_RETWEETS * _safe_to_float(df["retweets"])
    if "likes" in df.columns:
        w += config.WEIGHT_LIKES * _safe_to_float(df["likes"])
    if "bookmarks" in df.columns:
        w += config.WEIGHT_BOOKMARKS * _safe_to_float(df["bookmarks"])
    # avoid zeros
    w = np.maximum(w, 1.0)
    return w

def aggregate_signal(scores, weights=None):
    """
    Weighted mean of per-tweet scores -> single composite signal.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if weights is None:
        return scores.mean()
    w = np.asarray(weights, dtype=np.float64)
    return np.average(scores, weights=w)

# ---------- Confidence intervals via bootstrap (parallel) ----------

def _bootstrap_mean(scores, weights, rng_state):
    rng = np.random.RandomState(rng_state)
    n = len(scores)
    idx = rng.randint(0, n, size=n)  # sample with replacement
    if weights is None:
        return scores[idx].mean()
    return np.average(scores[idx], weights=weights[idx])

def bootstrap_ci(scores, weights=None, n_boot=config.BOOTSTRAP_SAMPLES, alpha=config.ALPHA, n_jobs=config.N_JOBS):
    """
    Parallel bootstrap of the weighted mean. Returns (lo, hi).
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = None if weights is None else np.asarray(weights, dtype=np.float64)

    seeds = np.random.randint(0, 2**31 - 1, size=n_boot)
    boot_vals = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_bootstrap_mean)(scores, weights, int(s)) for s in seeds
    )
    lo = np.percentile(boot_vals, 100 * (alpha/2))
    hi = np.percentile(boot_vals, 100 * (1 - alpha/2))
    return float(lo), float(hi)
