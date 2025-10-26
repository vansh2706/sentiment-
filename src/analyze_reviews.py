import argparse
import csv
from pathlib import Path
from sentiment_analyzer import SentimentAnalyzer


def read_texts(input_path: Path, text_column: str = "text"):
    if input_path.suffix.lower() in {".txt"}:
        with input_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    else:
        # assume CSV with header
        with input_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_column in row and row[text_column]:
                    yield row[text_column]


def write_results(output_path: Path, rows):
    fieldnames = ["text", "label", "score"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            text = r["text"]
            sent = r["sentiment"]
            if isinstance(sent, dict):
                label = sent.get("label", "")
                score = sent.get("score", "")
            else:
                label = sent
                score = ""
            writer.writerow({"text": text, "label": label, "score": score})


def main():
    parser = argparse.ArgumentParser(description="Analyze sentiment for product reviews from a CSV or TXT file.")
    parser.add_argument("input", type=str, help="Path to input CSV/TXT file. For CSV, include a header.")
    parser.add_argument("--text-column", type=str, default="text", help="Column name in CSV containing the review text.")
    parser.add_argument("--method", type=str, default="vader", choices=["vader", "textblob"], help="Sentiment analysis method.")
    parser.add_argument("--output", type=str, default="reviews_with_sentiment.csv", help="Output CSV path.")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    analyzer = SentimentAnalyzer()

    rows = []
    for text in read_texts(input_path, args.text_column):
        if args.method == "textblob":
            sentiment = analyzer.analyze_textblob(text)
        else:
            sentiment = analyzer.analyze_vader(text)
        rows.append({"text": text, "sentiment": sentiment})

    write_results(output_path, rows)
    print(f"Wrote {len(rows)} rows with sentiment to {output_path}")


if __name__ == "__main__":
    main()
