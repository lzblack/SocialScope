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


if __name__ == "__main__":
    result = analyze_sentiment_textblob("Covid cases are increasing fast!")
    print(result)
