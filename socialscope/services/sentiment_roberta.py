import os
from pathlib import Path
import numpy as np
from scipy.special import softmax
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import threading


class SentimentAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
                    cls._instance.tokenizer = None
                    cls._instance.model = None
                    cls._instance.config = None
        return cls._instance

    def load_model(self, model_name: str, cache_dir: str):
        if self.tokenizer is None or self.model is None or self.config is None:
            local_model_path = Path(cache_dir) / model_name.split("/")[-1]

            if local_model_path.exists():
                print(f"Use local model: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    str(local_model_path)
                )
                self.config = AutoConfig.from_pretrained(str(local_model_path))
            else:
                print(f"Download from HuggingFace: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self.config = AutoConfig.from_pretrained(model_name)

                local_model_path.mkdir(parents=True, exist_ok=True)
                self.tokenizer.save_pretrained(str(local_model_path))
                self.model.save_pretrained(str(local_model_path))
                self.config.save_pretrained(str(local_model_path))
                print(f"Model is saved to: {local_model_path}")

    def analyze_sentiment(self, text: str) -> dict:
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        max_index = np.argmax(scores)
        max_label = self.config.id2label[max_index]
        max_score = scores[max_index]
        return {"label": max_label, "score": np.round(float(max_score), 4)}


def analyze_sentiment_roberta(text: str, custom_cache_dir: str = None) -> dict:
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    cache_dir = custom_cache_dir or os.path.join(os.getcwd(), ".cache")

    analyzer = SentimentAnalyzer()
    analyzer.load_model(MODEL, cache_dir)

    try:
        return analyzer.analyze_sentiment(text)
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    result = analyze_sentiment_roberta("I love this product!")
    print(result)
    print(
        analyze_sentiment_roberta(
            "@charles_pare @28delayslater @Model3Owners Model S plaid is faster, quicker, more range, and far safer than the Taycan. But hey, it's your money."
        )
    )
