import time
from datetime import datetime, timedelta, timezone
from playwright.sync_api import sync_playwright
from utils import normalize_text, strip_urls, extract_hashtags, extract_mentions, logger
import config

def scrape_tweets_playwright(max_tweets=config.MAX_TWEETS, hours=config.SINCE_LAST_N_HOURS):
    query = " OR ".join(config.HASHTAGS)
    url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
    since_dt = datetime.now(timezone.utc) - timedelta(hours=hours)

    tweets = {}
    logger.info("Opening Twitter search: %s", url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        time.sleep(5)  # wait for tweets to load

        while len(tweets) < max_tweets:
            articles = page.query_selector_all("article")
            for article in articles:
                try:
                    tweet_id = article.get_attribute("data-testid") or str(hash(article.inner_text()))
                    if tweet_id in tweets:
                        continue

                    text = article.inner_text()
                    if not text: 
                        continue

                    # crude parsing (Playwright doesn’t expose structured tweet info like snscrape)
                    content = normalize_text(strip_urls(text))
                    hashtags = extract_hashtags(content)
                    mentions = extract_mentions(content)

                    tweets[tweet_id] = {
                        "tweet_id": tweet_id,
                        "username": None,
                        "displayname": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),  # Playwright doesn’t expose per-tweet timestamp easily
                        "content": content,
                        "raw_content": text,
                        "reply_count": 0,
                        "retweet_count": 0,
                        "like_count": 0,
                        "quote_count": 0,
                        "hashtags": hashtags,
                        "mentions": mentions,
                        "lang": "und",
                        "url": None,
                    }
                except Exception as e:
                    logger.warning("Parse error: %s", e)

            if len(tweets) >= max_tweets:
                break

            logger.info("Collected %d tweets so far, scrolling...", len(tweets))
            page.mouse.wheel(0, 2000)  # scroll down
            time.sleep(3)

        browser.close()

    logger.info("Collected %d tweets", len(tweets))
    return list(tweets.values())
