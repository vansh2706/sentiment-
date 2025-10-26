from sentiment_analyzer import SentimentAnalyzer


def demo():
    analyzer = SentimentAnalyzer()
    samples = [
        "I love this product! It works great and the quality is excellent.",
        "Terrible experience. The item broke after one day of use.",
        "It's okay, nothing special but not bad either.",
        "Best purchase I've made this year!!!",
        "Worst customer service ever. I'm so disappointed.",
    ]

    print("VADER results:")
    for s in samples:
        res = analyzer.analyze_vader(s)
        print(f"{res['label'].upper():8} {res['score']:+.3f} - {s}")

    print("\nTextBlob results:")
    for s in samples:
        label = analyzer.analyze_textblob(s)
        print(f"{label.upper():8} - {s}")


if __name__ == "__main__":
    demo()
