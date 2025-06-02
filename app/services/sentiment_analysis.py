from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

vader = SentimentIntensityAnalyzer()
bert_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)


def analyze_sentiment_bert(text: str) -> str:
    result = bert_pipeline(text[:512])[0]["label"]
    return result.lower()  # returns "positive" or "negative"


def analyze_sentiment_textblob(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    return (
        "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"
    )


def analyze_sentiment_vader(text: str) -> str:
    score = vader.polarity_scores(text)["compound"]
    return "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral"
