import os
import pprint

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
            "reason": "Text is empty or not a string",
        }

    prompt = f"""Analyze the sentiment of the following text and provide a response in JSON format with the following keys:
    - sentiment: either "positive", "negative", or "neutral"
    - confidence: a float value between 0 and 1 indicating the confidence in the sentiment classification
    - reason: a string that briefly explains why the text was classified as the given sentiment


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
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.2,
        )

        result = eval(response.choices[0].message.content)

        return {
            "text": text,
            "sentiment_openai": result["sentiment"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        }
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.0,
            "reason": "An error occurred while analyzing the text",
        }


def main():
    texts = [
        "I really love this amazing product but the service was terrible",
        "The product quality is excellent but customer service was disappointing",
        "This is a neutral statement without much emotion",
        "Covid cases are increasing fast!",
        "😊🤣",
        "@charles_pare @28delayslater @Model3Owners Model S plaid is faster, quicker, more range, and far safer than the Taycan. But hey, it's your money.",
    ]

    for i, text in enumerate(texts, 1):
        print(f"\nAnalyzing Text {i}:")
        print(f"Text: {text}")
        pprint.pprint(analyze_sentiment_openai(text))


if __name__ == "__main__":
    main()
