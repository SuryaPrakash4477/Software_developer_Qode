import os
import pandas as pd
import numpy as np

import config
from utils import logger, count_rows
from analysis import (
    vectorize_texts,
    reduce_dimensionality,
    compute_per_tweet_scores,
    compute_weights,
    aggregate_signal,
    bootstrap_ci,
)
from visualizer import plot_signal_streaming

# If you want to re-run scraping each time, keep your import:
from scraper import scrape_tweets_selenium

def run():
    tweets = scrape_tweets_selenium(max_tweets=config.MAX_TWEETS)
    # 1) Load (or scrape then load)
    if not os.path.exists(config.PARQUET_FILE):
        logger.error("No parquet found at %s. Run the scraper first.", config.PARQUET_FILE)
        return

    df = pd.read_parquet(config.PARQUET_FILE)
    if "content" not in df.columns or len(df) == 0:
        logger.error("No 'content' available in parquet. Rows=%d", len(df))
        return

    logger.info("Loaded %d tweets", count_rows(df))

    # Optional quick clean
    texts = df["content"].astype(str).fillna("").tolist()

    # 2) Vectorize (Hashing TF-IDF by default; memory-efficient)
    X, meta = vectorize_texts(texts)

    # 3) Dimensionality reduction
    Z, svd = reduce_dimensionality(X, n_components=config.N_COMPONENTS)

    # 4) Per-tweet scores & weights
    scores = compute_per_tweet_scores(Z)
    weights = compute_weights(df)  # uses engagement if columns present

    # 5) Aggregate signal + CI (bootstrap in parallel)
    signal = aggregate_signal(scores, weights)
    lo, hi = bootstrap_ci(scores, weights, n_boot=config.BOOTSTRAP_SAMPLES, alpha=config.ALPHA)
    logger.info("Signal=%.5f  95%% CI=[%.5f, %.5f]", signal, lo, hi)

    # 6) Memory-efficient visualization (downsample/bin)
    timestamps = df["timestamp"] if "timestamp" in df.columns else np.arange(len(scores))
    plot_signal_streaming(
        timestamps=timestamps,
        per_tweet_scores=scores,
        sample_every=config.PLOT_SAMPLE_EVERY,
        bin_minutes=config.PLOT_BIN_MINUTES,
        out_path=config.PLOT_OUTPUT,
    )

    logger.info("Analysis complete.")

if __name__ == "__main__":
    run()
