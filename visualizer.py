import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import logger
import config

def _downsample_by_stride(ts, ys, stride):
    if stride <= 1:
        return ts, ys
    return ts[::stride], ys[::stride]

def _bin_time(ts, ys, minutes):
    if minutes is None or minutes <= 0:
        return ts, ys
    df = pd.DataFrame({"ts": ts, "y": ys})
    # ts expected as pandas datetime; if string, convert:
    if not np.issubdtype(df["ts"].dtype, np.datetime64):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
    df = df.dropna(subset=["ts"])
    grp = df.resample(f"{int(minutes)}min", on="ts").mean(numeric_only=True)
    grp = grp.dropna()
    return grp.index.to_pydatetime(), grp["y"].to_numpy()

def plot_signal_streaming(
    timestamps,
    per_tweet_scores,
    sample_every=config.PLOT_SAMPLE_EVERY,
    bin_minutes=config.PLOT_BIN_MINUTES,
    out_path=config.PLOT_OUTPUT,
):
    """
    Memory-efficient: we optionally bin by time and/or downsample by stride.
    """
    ts = np.asarray(timestamps)
    ys = np.asarray(per_tweet_scores, dtype=float)

    # sort by time if available
    try:
        ts_dt = pd.to_datetime(ts, errors="coerce", utc=True)
        order = np.argsort(ts_dt.fillna(pd.Timestamp.min))
        ts, ys = ts[order], ys[order]
    except Exception:
        pass

    # optional binning
    ts, ys = _bin_time(ts, ys, bin_minutes)
    # optional stride downsample
    ts, ys = _downsample_by_stride(np.array(ts), ys, sample_every)

    plt.figure(figsize=(10, 4))
    plt.plot(ts, ys)
    plt.title("Composite Tweet Signal (downsampled)")
    plt.xlabel("Time")
    plt.ylabel("Score (unitless)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info("Saved plot to %s (points=%d)", out_path, len(ys))
