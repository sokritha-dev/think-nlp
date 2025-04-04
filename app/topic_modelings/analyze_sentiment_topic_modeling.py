import argparse
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="CSV with 'review' and 'dominant_topic'"
)
parser.add_argument(
    "file_output", type=str, help="Path to save sentiment by topic analysis (CSV)"
)
parser.add_argument(
    "vectorizer_input", type=str, help="Path to TF-IDF vectorizer (pkl)"
)
parser.add_argument(
    "model_input", type=str, help="Path to trained sentiment model (pkl)"
)
args = parser.parse_args()

# Load reviews + topics
df = pd.read_csv(args.file_input)

# Load model + vectorizer
with open(
    "app/data/train_models/tfidf_logistic_regression_sentiment_model.pkl", "rb"
) as f:
    model = pickle.load(f)

with open("app/data/features/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict sentiment
X = vectorizer.transform(df["review"].astype(str))
df["sentiment"] = model.predict(X)

# Clean + prepare
df = df.dropna(subset=["review", "sentiment", "dominant_topic"])
df["sentiment"] = df["sentiment"].astype(str).str.strip().str.capitalize()
df["dominant_topic"] = (
    pd.to_numeric(df["dominant_topic"], errors="coerce").fillna(-1).astype(int)
)
df = df[df["dominant_topic"] >= 0]

# Count sentiment per topic
sentiment_counts = (
    df.groupby(["dominant_topic", "sentiment"]).size().reset_index(name="count")
)
total_per_topic = df.groupby("dominant_topic").size().reset_index(name="total")
sentiment_counts = sentiment_counts.merge(total_per_topic, on="dominant_topic")
sentiment_counts["percentage"] = (
    sentiment_counts["count"] / sentiment_counts["total"] * 100
)

# Save CSV
output_folder = os.path.dirname(args.file_output)
os.makedirs(output_folder, exist_ok=True)
sentiment_counts.to_csv(args.file_output, index=False)
print(f"✅ Sentiment by topic analysis saved to {args.file_output}")

# Save plot
plt.figure(figsize=(10, 6))
sns.barplot(data=sentiment_counts, x="dominant_topic", y="percentage", hue="sentiment")
plt.title("Sentiment Distribution by Topic")
plt.ylabel("Percentage")
plt.xlabel("Topic")
plt.legend(title="Sentiment")
plt.tight_layout()

plot_path = args.file_output.replace(".csv", ".png")
plt.savefig(plot_path)
print(f"📊 Sentiment plot saved to {plot_path}")
