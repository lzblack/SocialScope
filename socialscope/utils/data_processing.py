import pandas as pd
import numpy as np
from datetime import datetime


def process_csv(file):
    df = pd.read_csv(file)
    required_columns = ["author_username", "post_created_at", "raw_body_text"]

    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV文件缺少必要的列")

    def parse_date(date_string):
        try:
            return pd.to_datetime(date_string, format="%B %d %Y, %H:%M %Z")
        except ValueError:
            return pd.NaT

    df["post_created_at"] = df["post_created_at"].apply(parse_date)

    # 将 'raw_body_text' 列转换为字符串，并用空字符串替换 NaN 值
    df["raw_body_text"] = df["raw_body_text"].fillna("").astype(str)

    # 替换 NaT 为 None
    df = df.replace({pd.NaT: None})

    return df[required_columns].rename(
        columns={
            "author_username": "author",
            "post_created_at": "timestamp",
            "raw_body_text": "text",
        }
    )
