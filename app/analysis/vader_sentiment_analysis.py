import argparse
import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Input CSV with 'review' column")
parser.add_argument("file_output", type=str, help="Output CSV with 'sentiment' column")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.file_input)
if "review" not in df.columns:
    raise ValueError("Input CSV must contain a 'review' column")

# Initialize VADER
sid = SentimentIntensityAnalyzer()


# Predict sentiment
def get_sentiment(text):
    score = sid.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["review"].astype(str).apply(get_sentiment)

# Save output
output_folder = os.path.dirname(args.file_output)
os.makedirs(output_folder, exist_ok=True)
df.to_csv(args.file_output, index=False)

print(f"✅ VADER sentiment analysis saved to {args.file_output}")
