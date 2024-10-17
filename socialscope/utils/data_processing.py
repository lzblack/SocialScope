import re
import pandas as pd
import numpy as np
from datetime import datetime


def remove_ordinal_indicator(date_string):
    """"""
    return re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_string)


def process_csv(file):
    df = pd.read_csv(file)
    required_columns = ["author_username", "post_created_at", "raw_body_text"]

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

    return df[required_columns].rename(
        columns={
            "author_username": "author",
            "post_created_at": "timestamp",
            "raw_body_text": "text",
        }
    )


if __name__ == "__main__":
    filepath = "data/twitter_sample.csv"
    df = process_csv(filepath)
    print(df)
