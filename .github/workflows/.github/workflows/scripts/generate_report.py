import os, requests, feedparser
from datetime import datetime, timedelta, timezone

# Cấu hình
FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed"
]
COINS = ["bitcoin", "ethereum"]
LOOKBACK_HOURS = 24
OUT_DIR = "reports"

def get_prices():
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(COINS)}&vs_currencies=usd&include_24hr_change=true"
    r = requests.get(url, timeout=20)
    data = r.json()
    return data

def fetch_news():
    since = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    items = []
    for url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries:
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                if pub >= since:
                    items.append(f"- [{e.title}]({e.link}) — {pub.strftime('%Y-%m-%d %H:%M UTC')}")
    return items

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    prices = get_prices()
    news = fetch_news()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md = [f"# Bản tin Crypto 24h (cập nhật: {ts})\n"]
    md.append("## Giá nhanh")
    for c in COINS:
        if c in prices:
            p = prices[c]["usd"]
            ch = prices[c].get("usd_24h_change", 0.0)
            md.append(f"- **{c.title()}**: ${p:,} ({ch:.2f}% / 24h)")

    md.append("\n## Tin nổi bật")
    if news:
        md.extend(news)
    else:
        md.append("- Không có tin mới trong 24h qua.")

    # Lưu ra file
    date_tag = datetime.now().strftime("%Y-%m-%d")
    with open(f"{OUT_DIR}/crypto_news_{date_tag}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # Cập nhật README.md
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

if __name__ == "__main__":
    main()
