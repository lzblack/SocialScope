from socialscope.services.sentiment_openai import analyze_sentiment_openai
from socialscope.services.sentiment_roberta import analyze_sentiment_roberta
from socialscope.services.sentiment_textblob import analyze_sentiment_textblob


def preprocess(text):
    """"""
    new_text = []
    for t in text.split(" "):
        # t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "" if t.startswith("@") and len(t) > 1 else t
        t = "" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def analyze_dataframe(df):
    """
    Perform all the sentiment analyses on the text data in the given DataFrame.
    """
    df["preprocess_text"] = df["text"].apply(preprocess)
    df["sentiment_analysis_textblob"] = df["preprocess_text"].apply(
        analyze_sentiment_textblob
    )
    df["sentiment_textblob"] = df["sentiment_analysis_textblob"].apply(
        lambda x: x["sentiment"]
    )
    df["polarity_textblob"] = df["sentiment_analysis_textblob"].apply(
        lambda x: x["polarity"]
    )
    df["subjectivity_textblob"] = df["sentiment_analysis_textblob"].apply(
        lambda x: x["subjectivity"]
    )

    df["sentiment_analysis_roberta"] = df["preprocess_text"].apply(
        analyze_sentiment_roberta
    )
    df["sentiment_roberta"] = df["sentiment_analysis_roberta"].apply(
        lambda x: x["label"]
    )
    df["score_roberta"] = df["sentiment_analysis_roberta"].apply(lambda x: x["score"])

    df["sentiment_analysis_openai"] = df["preprocess_text"].apply(
        analyze_sentiment_openai
    )
    df["sentiment_openai"] = df["sentiment_analysis_openai"].apply(
        lambda x: x["sentiment_openai"]
    )
    df["confidence_openai"] = df["sentiment_analysis_openai"].apply(
        lambda x: x["confidence"]
    )

    return df
