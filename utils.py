import logging, html, unicodedata, re
from datetime import datetime, timezone

def setup_logger(name="scraper", level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(ch)
        logger.setLevel(level)
    return logger

logger = setup_logger()

def normalize_text(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    return s.strip()

URL_RE = re.compile(r"http\S+|https\S+")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@(\w+)")

def extract_hashtags(text): return [h.lower() for h in HASHTAG_RE.findall(text or "")]
def extract_mentions(text): return [m.lower() for m in MENTION_RE.findall(text or "")]
def strip_urls(text): return URL_RE.sub("", text or "").strip()
def to_utc(dt): return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
