# --- scraping (you already have these) ---
HASHTAGS = ["nifty50", "sensex", "intraday", "banknifty"]
MAX_TWEETS = 2000
SINCE_LAST_N_HOURS = 24
DATA_DIR = "data"
PARQUET_FILE = f"{DATA_DIR}/tweets.parquet"

# --- analysis ---
USE_HASHING_VECTOR = True          # memory-efficient text vectorization
HASHING_N_FEATURES = 2 ** 20       # ~1M hashed features (sparse)
TFIDF_MAX_FEATURES = 5000          # used only when USE_HASHING_VECTOR=False
N_COMPONENTS = 100                 # SVD dimensionality
BOOTSTRAP_SAMPLES = 1000           # for CI; tune up/down
ALPHA = 0.05                       # 95% CI

# engagement weights (optional)
WEIGHT_REPLIES = 1.0
WEIGHT_RETWEETS = 2.0
WEIGHT_LIKES = 1.0
WEIGHT_BOOKMARKS = 0.5

# --- concurrency ---
N_JOBS = -1                        # use all cores where supported

# --- visualization ---
PLOT_SAMPLE_EVERY = 5              # downsample factor (keep every k-th point)
PLOT_BIN_MINUTES = 5               # optional time bin for aggregation
PLOT_OUTPUT = f"{DATA_DIR}/signal_preview.png"
