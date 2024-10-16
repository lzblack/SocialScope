import numpy as np
from scipy.special import softmax
from textblob import TextBlob
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    pipeline,
)


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


def analyze_dataframe_textblob(df):
    """
    Perform sentiment analysis on the text data in the given DataFrame using TextBlob.
    """
    df["sentiment_analysis"] = df["text"].apply(analyze_sentiment_textblob)
    df["sentiment"] = df["sentiment_analysis"].apply(lambda x: x["sentiment"])
    df["polarity"] = df["sentiment_analysis"].apply(lambda x: x["polarity"])
    df["subjectivity"] = df["sentiment_analysis"].apply(lambda x: x["subjectivity"])
    return df


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def analyze_sentiment_roberta(text: str) -> dict:
    """
    Perform sentiment analysis on the given text using RoBERTa.

    Reference: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)

    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")

    return scores


if __name__ == "__main__":
    print(analyze_sentiment_roberta("I love this product!"))
