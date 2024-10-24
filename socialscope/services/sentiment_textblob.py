from textblob import TextBlob


def analyze_sentiment_textblob(text: str) -> dict:
    """
    Perform sentiment analysis on the given text using TextBlob.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "sentiment": "neutral",
            "polarity": 0.0,
            "subjectivity": 0.0,
        }

    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        sentiment = "positive"
    elif analysis.sentiment.polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "text": text,
        "sentiment": sentiment,
        "polarity": float(analysis.sentiment.polarity),
        "subjectivity": float(analysis.sentiment.subjectivity),
    }


def main():
def main():
    texts = [
        "I really love this amazing product but the service was terrible",
        "The product quality is excellent but customer service was disappointing",
        "This is a neutral statement without much emotion",
        "Covid cases are increasing fast!",
        "😊🤣",
        "@charles_pare @28delayslater @Model3Owners Model S plaid is faster, quicker, more range, and far safer than the Taycan. But hey, it's your money.",
    ]

    for i, text in enumerate(texts, 1):
        print(f"\nAnalyzing Text {i}:")
        print(f"Text: {text}")
        print(analyze_sentiment_textblob(text))


if __name__ == "__main__":
    main()