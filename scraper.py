import json
import os
import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import config

COOKIE_FILE = "twitter_cookies.json"


def load_cookies(driver, cookie_file):
    """Load cookies into Selenium session."""
    with open(cookie_file, "r", encoding="utf-8") as f:
        cookies = json.load(f)
    for cookie in cookies:
        try:
            driver.add_cookie({
                "name": cookie.get("name"),
                "value": cookie.get("value"),
                "domain": cookie.get("domain"),
                "path": cookie.get("path"),
                "secure": cookie.get("secure", False),
                "httpOnly": cookie.get("httpOnly", False),
            })
        except Exception:
            continue


def build_time_windows(hours=24, step=1):
    """Build hourly windows for pagination."""
    now = datetime.utcnow()
    windows = []
    for i in range(0, hours, step):
        end = now - timedelta(hours=i)
        start = now - timedelta(hours=i+step)
        windows.append((start, end))
    return windows


def extract_tweet_data(tweet):
    """Extract fields from tweet HTML."""
    try:
        content_tag = tweet.find("div", {"data-testid": "tweetText"})
        content = content_tag.get_text(" ", strip=True) if content_tag else None

        user_block = tweet.find("div", {"dir": "ltr"})
        username = user_block.get_text(strip=True) if user_block else None

        handle_tag = tweet.find("a", href=True)
        handle = handle_tag["href"] if handle_tag else None

        timestamp_tag = tweet.find("time")
        timestamp = timestamp_tag["datetime"] if timestamp_tag else None

        # Engagement metrics
        replies = tweet.find("div", {"data-testid": "reply"})
        retweets = tweet.find("div", {"data-testid": "retweet"})
        likes = tweet.find("div", {"data-testid": "like"})

        replies_count = replies.get_text(strip=True) if replies else "0"
        retweets_count = retweets.get_text(strip=True) if retweets else "0"
        likes_count = likes.get_text(strip=True) if likes else "0"

        # Mentions + Hashtags
        mentions = [a.get_text(strip=True) for a in tweet.find_all("a") if a.get("href", "").startswith("/")]
        hashtags = [span.get_text(strip=True) for span in tweet.find_all("span") if span.get_text().startswith("#")]

        return {
            "username": username,
            "handle": handle,
            "timestamp": timestamp,
            "content": content,
            "replies": replies_count,
            "retweets": retweets_count,
            "likes": likes_count,
            "mentions": ", ".join(mentions) if mentions else None,   # flatten list
            "hashtags": ", ".join(hashtags) if hashtags else None    # flatten list
        }
    except Exception:
        return None


def scrape_tweets_selenium(max_tweets=2000):
    """Scrape tweets using Selenium + time-window pagination."""
    options = Options()
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(options=options)

    # Step 1: Open Twitter & load cookies
    driver.get("https://twitter.com")
    time.sleep(5)
    load_cookies(driver, COOKIE_FILE)
    driver.refresh()
    time.sleep(5)

    all_tweets = []
    time_windows = build_time_windows(hours=config.SINCE_LAST_N_HOURS, step=1)

    for start, end in time_windows:
        if len(all_tweets) >= max_tweets:
            break

        since = start.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        until = end.strftime("%Y-%m-%d_%H:%M:%S_UTC")
        query = " OR ".join([f"%23{tag}" for tag in config.HASHTAGS])
        url = f"https://twitter.com/search?q={query}%20since%3A{since}%20until%3A{until}&f=live"

        print(f"Scraping window {since} → {until}")
        driver.get(url)

        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//article[@data-testid='tweet']"))
            )
        except:
            print("No tweets or blocked in this window")
            continue

        last_height = driver.execute_script("return document.body.scrollHeight")
        while len(all_tweets) < max_tweets:
            soup = BeautifulSoup(driver.page_source, "html.parser")
            for tweet in soup.find_all("article", {"data-testid": "tweet"}):
                data = extract_tweet_data(tweet)
                if data and data not in all_tweets:
                    all_tweets.append(data)

            # Scroll
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        print(f"Collected {len(all_tweets)} tweets so far")

    driver.quit()

    # Save
    print(len(all_tweets))
    os.makedirs(config.DATA_DIR, exist_ok=True)
    df = pd.DataFrame(all_tweets).drop_duplicates()
    print(len(df))

    if len(df) > 0:
        df.to_parquet(config.PARQUET_FILE, engine="pyarrow", index=False)
        print(f"Saved {len(df)} tweets to {config.PARQUET_FILE}")
    else:
        print("No tweets scraped. Cookies may have expired.")

    return df
