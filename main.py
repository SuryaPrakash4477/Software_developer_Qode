import pandas as pd, numpy as np, os
import config
from scraper import scrape_tweets_playwright
from storage import save_to_parquet
from analysis import compute_tfidf_matrix, reduce_dimensionality, weighted_signal_from_matrix
from utils import logger

def run():
    tweets = scrape_tweets_playwright(max_tweets=config.MAX_TWEETS)
    save_to_parquet(tweets, config.PARQUET_FILE)

    df = pd.read_parquet(config.PARQUET_FILE)
    logger.info("Loaded %d tweets for analysis", len(df))

    texts = df["content"].fillna("").tolist()
    X, vec = compute_tfidf_matrix(texts)
    X_red, svd = reduce_dimensionality(X)

    # weight all tweets equally (since Playwright doesn’t give likes/retweets easily)
    weights = np.ones(len(df), dtype=np.float32)
    agg = weighted_signal_from_matrix(X, weights)

    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/agg_tfidf_signal.npy", agg)
    logger.info("Saved aggregate signal to outputs/agg_tfidf_signal.npy")

if __name__ == "__main__":
    run()
