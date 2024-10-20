from datetime import datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from socialscope.services.sentiment_analysis import analyze_dataframe
from socialscope.utils.data_processing import process_csv

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
    except ValueError as e:
        return JSONResponse(status_code=400, content={"detail": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {str(e)}"},
        )


@router.get("/metrics")
async def get_metrics(start_time: datetime, end_time: datetime):
    """
    Get metrics for the sentiment analysis service.
    """
    pass
