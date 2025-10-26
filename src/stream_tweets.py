import os
import argparse
import threading
import queue
from typing import List

from sentiment_analyzer import SentimentAnalyzer


def stream_tweets(bearer_token: str, keywords: List[str], out_q: queue.Queue):
    try:
        import tweepy
    except ImportError:
        raise SystemExit("tweepy is not installed. Please install it via 'pip install tweepy'.")

    class StreamListener(tweepy.StreamingClient):
        def on_tweet(self, tweet):
            out_q.put(tweet.text)

        def on_connection_error(self):
            return super().on_connection_error()

        def on_errors(self, errors):
            return super().on_errors(errors)

    client = StreamListener(bearer_token)

    # Clean existing rules
    existing = client.get_rules()
    if existing and existing.data:
        ids = [r.id for r in existing.data]
        client.delete_rules(ids)

    # Add new rules
    rule_value = " OR ".join(keywords)
    client.add_rules(tweepy.StreamRule(rule_value))

    # Start streaming sample (tweets in English)
    client.filter(threaded=True, expansions=None, tweet_fields=["lang"], languages=["en"])  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Stream tweets in real time and print sentiment.")
    parser.add_argument("--keywords", nargs="+", required=True, help="Keywords to filter tweets (space-separated).")
    parser.add_argument("--method", type=str, default="vader", choices=["vader", "textblob"], help="Sentiment method.")
    args = parser.parse_args()

    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit("Please set TWITTER_BEARER_TOKEN environment variable.")

    out_q: queue.Queue = queue.Queue()
    t = threading.Thread(target=stream_tweets, args=(bearer_token, args.keywords, out_q), daemon=True)
    t.start()

    analyzer = SentimentAnalyzer()

    print("Streaming... Press Ctrl+C to stop.")
    try:
        while True:
            try:
                text = out_q.get(timeout=1)
            except queue.Empty:
                continue
            if args.method == "textblob":
                sentiment = analyzer.analyze_textblob(text)
                label = sentiment
                score = ""
            else:
                res = analyzer.analyze_vader(text)
                label = res["label"]
                score = f" {res['score']:.3f}"
            print(f"[{label.upper()}{score}] {text}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
