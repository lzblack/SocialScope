from socialscope.services.sentiment_roberta import analyze_sentiment_roberta
from socialscope.services.sentiment_textblob import analyze_sentiment_textblob


def analyze_dataframe(df):
    """
    Perform all the sentiment analyses on the text data in the given DataFrame.
    """
    df["sentiment_analysis_textblob"] = df["text"].apply(analyze_sentiment_textblob)
    df["sentiment_textblob"] = df["sentiment_analysis_textblob"].apply(lambda x: x["sentiment"])
    df["polarity_textblob"] = df["sentiment_analysis_textblob"].apply(lambda x: x["polarity"])
    df["subjectivity_textblob"] = df["sentiment_analysis_textblob"].apply(
        lambda x: x["subjectivity"]
    )

    df["sentiment_analysis_roberta"] = df["text"].apply(analyze_sentiment_roberta)
    df["sentiment_roberta"] = df["sentiment_analysis_roberta"].apply(lambda x: x["label"])
    df["score_roberta"] = df["sentiment_analysis_roberta"].apply(lambda x: x["score"])

    return df
