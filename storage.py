import os, pandas as pd
from utils import logger
import config

os.makedirs(config.DATA_DIR, exist_ok=True)

def save_to_parquet(records, path=config.PARQUET_FILE):
    if not records: return
    new_df = pd.DataFrame(records)
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined.drop_duplicates(subset=["tweet_id"], inplace=True)
    else:
        combined = new_df.drop_duplicates(subset=["tweet_id"])
    combined.to_parquet(path, index=False)
    logger.info("Saved %d tweets to %s", len(combined), path)
