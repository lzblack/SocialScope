from fastapi import APIRouter, UploadFile, File, HTTPException
from socialscope.services.sentiment_analysis import analyze_dataframe_textblob
from socialscope.utils.data_processing import process_csv
import pandas as pd
import numpy as np
from datetime import datetime

router = APIRouter()


@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file containing tweets and perform sentiment analysis on the text data.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a CSV file."
        )

    try:
        df = process_csv(file.file)
        print(df.head())
        print(df.dtypes)
        print(df["text"].isnull().sum())
        print(df["text"].apply(type).value_counts())

        df = analyze_dataframe(df)

        results = df.where(pd.notnull(df), None).to_dict("records")

        return {
            "message": "CSV processed successfully",
            "total_rows": len(df),
            "results": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@router.get("/metrics")
async def get_metrics(start_time: datetime, end_time: datetime):
    """
    Get metrics for the sentiment analysis service.
    """
    pass
