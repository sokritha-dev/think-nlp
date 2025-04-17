import argparse
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
import matplotlib.pyplot as plt

nltk.download("vader_lexicon")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="Input CSV with 'review' and 'dominant_topic'"
)
parser.add_argument(
    "file_output", type=str, help="Output CSV with sentiment distribution"
)
args = parser.parse_args()

# Load data
df = pd.read_csv(args.file_input)
if "review" not in df.columns or "dominant_topic" not in df.columns:
    raise ValueError("Input CSV must contain 'review' and 'dominant_topic' columns")

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment
df["sentiment_score"] = df["review"].apply(
    lambda x: analyzer.polarity_scores(str(x))["compound"]
)
df["sentiment"] = df["sentiment_score"].apply(
    lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"
)

# Group by topic and sentiment
sentiment_dist = pd.pivot_table(
    df, index="dominant_topic", columns="sentiment", aggfunc="size", fill_value=0
).reset_index()
sentiment_dist.columns = ["topic_id", "negative", "neutral", "positive"]

# Add proportions
sentiment_dist["review_count"] = sentiment_dist[
    ["negative", "neutral", "positive"]
].sum(axis=1)
sentiment_dist["neg_prop"] = sentiment_dist["negative"] / sentiment_dist["review_count"]
sentiment_dist["neu_prop"] = sentiment_dist["neutral"] / sentiment_dist["review_count"]
sentiment_dist["pos_prop"] = sentiment_dist["positive"] / sentiment_dist["review_count"]

# Add average sentiment score
avg_sentiment = df.groupby("dominant_topic")["sentiment_score"].mean().reset_index()
sentiment_dist = sentiment_dist.merge(
    avg_sentiment, left_on="topic_id", right_on="dominant_topic"
).drop("dominant_topic", axis=1)
sentiment_dist.rename(columns={"sentiment_score": "avg_sentiment"}, inplace=True)

# Merge with topic labels
topic_labels = pd.read_csv("app/data/topics/lda_topics_labeled.csv")
sentiment_dist = sentiment_dist.merge(
    topic_labels[["topic", "label"]], left_on="topic_id", right_on="topic", how="left"
)
sentiment_dist.drop(columns=["topic"], inplace=True)

# Reorder columns
sentiment_dist = sentiment_dist[
    [
        "topic_id",
        "label",
        "positive",
        "negative",
        "neutral",
        "pos_prop",
        "neg_prop",
        "neu_prop",
        "avg_sentiment",
        "review_count",
    ]
]

# Save results
output_dir = os.path.dirname(args.file_output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
sentiment_dist.to_csv(args.file_output, index=False)
print(f"✅ Sentiment analysis saved to {args.file_output}")
print(sentiment_dist)

# Plot sentiment distribution
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.25
index = range(len(sentiment_dist))

ax.bar(index, sentiment_dist["pos_prop"], bar_width, label="Positive", color="green")
ax.bar(
    [i + bar_width for i in index],
    sentiment_dist["neg_prop"],
    bar_width,
    label="Negative",
    color="red",
)
ax.bar(
    [i + 2 * bar_width for i in index],
    sentiment_dist["neu_prop"],
    bar_width,
    label="Neutral",
    color="gold",
)

ax.set_xlabel("Topic")
ax.set_ylabel("Proportion of Sentiment")
ax.set_title("Sentiment Distribution by Topic")
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(sentiment_dist["label"], rotation=30, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig("app/data/topics/sentiment_distribution_vader.png")
print("✅ Sentiment distribution plot saved to 'sentiment_distribution.png'")
