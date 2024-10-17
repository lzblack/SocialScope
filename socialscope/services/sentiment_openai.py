import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def analyze_sentiment_openai(text: str) -> dict:
    """"""
    if not isinstance(text, str) or not text.strip():
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.0,
        }

    prompt = f"""Analyze the sentiment of the following text and provide a response in JSON format with the following keys:
    - sentiment: either "positive", "negative", or "neutral"
    - confidence: a float value between 0 and 1 indicating the confidence in the sentiment classification

    Text to analyze: "{text}"

    JSON response:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis AI. Respond only with the requested JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )

        result = eval(response.choices[0].message.content)

        return {
            "text": text,
            "sentiment_openai": result["sentiment"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.0,
        }


if __name__ == "__main__":
    result = analyze_sentiment_openai("I love this product!")
     
    print(result)
    print(
        analyze_sentiment_openai(
            "Model S plaid is faster, quicker, more range, and far safer than the Taycan. But hey, it's your money."
        )
    )
