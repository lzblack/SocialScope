import numpy as np
from scipy.special import softmax
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import os
import logging
from pathlib import Path
import shutil
import requests

logging.basicConfig(level=logging.INFO)
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


def preprocess(text):
    """"""
    new_text = []
    for t in text.split(" "):
        # t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "" if t.startswith("@") and len(t) > 1 else t
        t = "" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def download_file(url, filename):
    """"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    with open(filename, "wb") as file, logging.StreamHandler() as handler:
        for data in response.iter_content(block_size):
            size = file.write(data)
            handler.stream.write(
                f"\rDownloading {filename}: {size/total_size*100:.2f}%"
            )
            handler.stream.flush()
    print()


def load_or_download_model(model_name: str, cache_dir: str):
    """"""
    local_model_path = Path(cache_dir) / model_name.split("/")[-1]

    if local_model_path.exists():
        logger.info(f"Use local model: {local_model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
            model = AutoModelForSequenceClassification.from_pretrained(
                str(local_model_path)
            )
            config = AutoConfig.from_pretrained(str(local_model_path))
            return tokenizer, model, config
        except Exception as e:
            logger.warning(f"Failed to load local model: {e}")
            logger.info("Try to download the model again.")
            shutil.rmtree(local_model_path, ignore_errors=True)

    logger.info(f"Download from HuggingFace: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)

        local_model_path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(local_model_path))
        model.save_pretrained(str(local_model_path))
        config.save_pretrained(str(local_model_path))
        logger.info(f"Model is saved to: {local_model_path}")

        return tokenizer, model, config
    except Exception as e:
        logger.error(f"Fail to download: {e}")
        raise


def analyze_sentiment_roberta(text: str, custom_cache_dir: str = None) -> dict:
    """
    Perform sentiment analysis on the given text using RoBERTa.
    参考: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    """
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    cache_dir = custom_cache_dir or os.path.join(os.getcwd(), ".cache")

    try:
        tokenizer, model, config = load_or_download_model(MODEL, cache_dir)

        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors="pt")
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        max_index = np.argmax(scores)
        max_label = config.id2label[max_index]
        max_score = scores[max_index]
        result = {"label": max_label, "score": np.round(float(max_score), 4)}
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    result = analyze_sentiment_roberta("I love this product!")
    print(result)
    print(
        analyze_sentiment_roberta(
            "@charles_pare @28delayslater @Model3Owners Model S plaid is faster, quicker, more range, and far safer than the Taycan. But hey, it's your money."
        )
    )
