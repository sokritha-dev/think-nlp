import argparse
import pandas as pd
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import os
import pyLDAvis
import pyLDAvis.gensim_models

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="Path to the input CSV file containing reviews"
)
parser.add_argument(
    "file_output", type=str, help="Path to save the topic modeling output CSV"
)
parser.add_argument("num_topics", type=int, default=5, help="Number of LDA topics")
args = parser.parse_args()

# Load data
df = pd.read_csv(args.file_input)
if "review" not in df.columns:
    raise ValueError("Input file must contain a 'review' column")

# Optional: keep rating for downstream sentiment analysis
if "rating" not in df.columns:
    df["rating"] = None  # Fallback if rating is missing


# Tokenization for LDA
def preprocess_text(text):
    return str(text).lower().split()


df["tokens"] = df["review"].astype(str).apply(preprocess_text)

# Create bigrams and trigrams
bigram = Phrases(df["tokens"], min_count=2, threshold=5)
bigram_mod = Phraser(bigram)
trigram = Phrases(bigram_mod[df["tokens"]], threshold=5)
trigram_mod = Phraser(trigram)
df["tokens"] = df["tokens"].apply(lambda x: trigram_mod[bigram_mod[x]])

# Create dictionary and corpus
id2word = corpora.Dictionary(df["tokens"])
corpus = [id2word.doc2bow(text) for text in df["tokens"]]

# Build LDA model
lda_model = models.LdaModel(
    corpus=corpus,
    id2word=id2word,
    num_topics=args.num_topics,
    random_state=42,
    passes=10,
    iterations=100,
    per_word_topics=True,
)

# Extract topic keywords
topic_keywords = []
for i in range(args.num_topics):
    topic_terms = lda_model.show_topic(i, topn=20)
    keywords = ", ".join([term for term, _ in topic_terms])
    topic_keywords.append({"topic": i, "keywords": keywords})

# Assign dominant topic to each review
dominant_topics = []
for bow in corpus:
    topic_probs = lda_model.get_document_topics(bow)
    dominant_topic = (
        sorted(topic_probs, key=lambda x: -x[1])[0][0] if topic_probs else -1
    )
    dominant_topics.append(dominant_topic)

df["dominant_topic"] = dominant_topics

# Save outputs
output_folder = os.path.dirname(args.file_output)
os.makedirs(output_folder, exist_ok=True)

# ✅ Save topic keywords
topics_df = pd.DataFrame(topic_keywords)
topics_df.to_csv(args.file_output.replace(".csv", "_topics.csv"), index=False)

# ✅ Save only review and dominant_topic
df[["review", "dominant_topic"]].to_csv(args.file_output, index=False)


# ✅ Save LDA visualization
vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis_data, os.path.join(output_folder, "lda_visualization.html"))

print(f"✅ LDA topic modeling completed. Results saved to {args.file_output}")
