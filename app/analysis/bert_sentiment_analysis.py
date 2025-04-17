import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

# -------------------------
# 1. Parse arguments
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="Input CSV with 'review' and 'dominant_topic'"
)
parser.add_argument(
    "file_output", type=str, help="Output CSV with sentiment distribution"
)
args = parser.parse_args()

# -------------------------
# 2. Load data
# -------------------------
df = pd.read_csv(args.file_input)
if "review" not in df.columns or "dominant_topic" not in df.columns:
    raise ValueError("Input CSV must contain 'review' and 'dominant_topic' columns")

# -------------------------
# 3. Load DistilBERT model
# -------------------------
device = 0 if torch.cuda.is_available() else -1
sentiment_model = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    truncation=True,
)


# -------------------------
# 4. Predict sentiment in batches
# -------------------------
def run_in_batches(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = sentiment_model(batch)
        results.extend(batch_results)
    return results


texts = df["review"].astype(str).tolist()
bert_results = run_in_batches(texts)


# -------------------------
# 5. Parse sentiment label
# -------------------------
def parse_result(res):
    label = res["label"].capitalize()  # 'POSITIVE' → 'Positive'
    score = res["score"]
    if score < 0.6:
        return "Neutral", score
    return label, score


df["sentiment"], df["sentiment_score"] = zip(*[parse_result(r) for r in bert_results])

# -------------------------
# 6. Sentiment distribution
# -------------------------
sentiment_dist = pd.pivot_table(
    df, index="dominant_topic", columns="sentiment", aggfunc="size", fill_value=0
).reset_index()

# Fill in missing sentiment classes if needed
for col in ["Positive", "Negative", "Neutral"]:
    if col not in sentiment_dist.columns:
        sentiment_dist[col] = 0

sentiment_dist.columns = ["topic_id"] + [c.lower() for c in sentiment_dist.columns[1:]]

# Add proportions
sentiment_dist["review_count"] = sentiment_dist[
    ["positive", "negative", "neutral"]
].sum(axis=1)
sentiment_dist["pos_prop"] = sentiment_dist["positive"] / sentiment_dist["review_count"]
sentiment_dist["neg_prop"] = sentiment_dist["negative"] / sentiment_dist["review_count"]
sentiment_dist["neu_prop"] = sentiment_dist["neutral"] / sentiment_dist["review_count"]

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
).drop(columns=["topic"])

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

# -------------------------
# 7. Save CSV
# -------------------------
output_dir = os.path.dirname(args.file_output)
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

sentiment_dist.to_csv(args.file_output, index=False)
print(f"✅ DistilBERT sentiment analysis saved to {args.file_output}")
print(sentiment_dist)

# -------------------------
# 8. Plot bar chart
# -------------------------
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
ax.set_title("Sentiment Distribution by Topic (DistilBERT)")
ax.set_xticks([i + bar_width for i in index])
ax.set_xticklabels(sentiment_dist["label"], rotation=30, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig("app/data/topics/sentiment_distribution_distilbert.png")
print("✅ Sentiment distribution plot saved to 'sentiment_distribution_distilbert.png'")
