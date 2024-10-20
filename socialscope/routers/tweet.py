from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
import pandas as pd

from socialscope.services.sentiment_analysis import analyze_dataframe
from socialscope.utils.data_processing import process_csv

router = APIRouter()


class TextInput(BaseModel):
    text: str


@router.post("/analyze-sentiment")
async def analyze_sentiment(text_input: TextInput):
    """
    Analyze sentiment of a single text input.
    """
    try:
        # Create a single-row DataFrame
        df = pd.DataFrame([{"text": text_input.text}])

        # Perform sentiment analysis
        df = analyze_dataframe(df)

        # Convert the result to a dictionary
        result = df.where(pd.notnull(df), None).to_dict("records")[0]

        return {
            "message": "Text analyzed successfully",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing text: {str(e)}")
