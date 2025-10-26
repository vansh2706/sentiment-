# Sentiment Analysis for Real-Time Response (Python)

Lightweight NLP project for analyzing sentiment on product reviews and tweets.

## Features

- VADER (NLTK) and TextBlob sentiment analysis
- Batch processing for reviews from CSV/TXT
- Real-time tweet streaming + sentiment (Tweepy + Twitter API v2)

## Setup

1. Install Python 3.10+
2. Create a virtual environment (recommended)
3. Install dependencies

```powershell
# From the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

First run will download NLTK resources automatically.

## Quick demo

```powershell
python .\src\main.py
```

## Analyze product reviews

Input can be:
- TXT: one review per line
- CSV: must include a header; default text column is `text`

```powershell
# TXT
python .\src\analyze_reviews.py .\data\reviews.txt --method vader --output .\out\reviews_scored.csv

# CSV (custom text column)
python .\src\analyze_reviews.py .\data\reviews.csv --text-column review --method textblob --output .\out\reviews_scored.csv
```

## Stream tweets in real time

Requires a Twitter API v2 Bearer Token.

```powershell
# Set env var in current PowerShell session
$env:TWITTER_BEARER_TOKEN = "<your-bearer-token>"

# Stream and score tweets containing any of the keywords
python .\src\stream_tweets.py --keywords "iphone" "samsung" "pixel" --method vader
```

## Notes

- VADER is tuned for social media and performs well for tweets.
- TextBlob is simple and gives a polarity-based label.
- For larger workloads or model fine-tuning, consider upgrading to transformer-based models later.
