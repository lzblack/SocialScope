import re

import numpy as np
import pandas as pd


def remove_ordinal_indicator(date_string):
    """"""
    return re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_string)


def process_csv(file):
    df = pd.read_csv(file, encoding="utf-8")

    if len(df) > 10:
        raise ValueError("CSV file contains more than 10 rows")

    required_columns = [
        "author_username",
        "post_created_at",
        "raw_body_text",
        "sentiment_score",
    ]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV lacks required columns")

    def parse_date(date_string):

        try:
            date = pd.to_datetime(
                remove_ordinal_indicator(date_string), format="%B %d %Y, %H:%M %Z"
            )
            return date
        except ValueError as e:
            print(e)
            return pd.NaT

    df["post_created_at"] = df["post_created_at"].apply(parse_date)

    # Fill missing values with empty string
    df["raw_body_text"] = df["raw_body_text"].fillna("").astype(str)

    # Replace NaT with None
    df = df.replace({pd.NaT: None})

    df["sentiment_score"] = df["sentiment_score"].astype(float)

    # Add sentiment column based on Nuvi sentiment score, if score is greater than 0, sentiment is positive, else if score is less than 0, sentiment is negative else neutral
    df["sentiment"] = np.where(
        df["sentiment_score"] > 0,
        "positive",
        np.where(df["sentiment_score"] < 0, "negative", "neutral"),
    )

    required_columns.append("sentiment")

    return df[required_columns].rename(
        columns={
            "author_username": "author",
            "post_created_at": "timestamp",
            "raw_body_text": "text",
            "sentiment_score": "sentiment_score_nuvi",
            "sentiment": "sentiment_nuvi",
        }
    )


if __name__ == "__main__":
    filepath = "data/twitter_sample.csv"
    df = process_csv(filepath)
    print(df)
