from datetime import timedelta

HASHTAGS = ["#nifty50", "#sensex", "#intraday", "#banknifty"]
MAX_TWEETS = 2000
CHUNK_SIZE = 200
DATA_DIR = "data"
PARQUET_FILE = f"{DATA_DIR}/tweets.parquet"
SINCE_LAST_N_HOURS = 24
