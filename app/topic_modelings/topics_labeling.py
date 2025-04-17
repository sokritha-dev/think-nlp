import argparse
import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="Path to the input CSV file containing reviews"
)
parser.add_argument(
    "file_output", type=str, help="Path to save the topic modeling output CSV"
)
parser.add_argument(
    "candidate_labels", type=str, help="JSON string of candidate labels"
)

args = parser.parse_args()
print(args.candidate_labels)
candidate_labels = json.loads(args.candidate_labels)

# Load your LDA topic keywords
lda_df = pd.read_csv(args.file_input)

# Load Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode candidate labels
label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)

# Generate labels for each topic
assigned_labels = []
for index, row in lda_df.iterrows():
    topic_keywords = row["keywords"].replace(",", " ")
    topic_embedding = model.encode(topic_keywords, convert_to_tensor=True)
    scores = util.cos_sim(topic_embedding, label_embeddings)
    best_label_idx = scores.argmax().item()
    best_label = candidate_labels[best_label_idx]
    assigned_labels.append(best_label)

# Add labels to the DataFrame
lda_df["label"] = assigned_labels

# Save the labeled topics
lda_df.to_csv("app/data/topics/lda_topics_labeled.csv", index=False)
print("✅ Topic labels generated and saved.")
