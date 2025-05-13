from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")


class EDA:
    def __init__(self, df: pd.DataFrame, file_id: str):
        self.df = df
        self.file_id = file_id
        self.text_column = "lemmatized_tokens"

    def run_eda(self) -> dict:
        # Ensure column is string list
        docs = self.df[self.text_column].dropna().astype(str)

        # Word cloud data (top 100 words)
        all_words = " ".join(docs).split()
        word_freq = Counter(all_words)
        word_cloud = [{"text": w, "value": c} for w, c in word_freq.most_common(100)]

        # Text length distribution
        length_series = docs.apply(lambda x: len(x.split()))
        length_distribution = length_series.value_counts().sort_index().reset_index()
        length_distribution.columns = ["length", "count"]
        length_distribution_data = length_distribution.to_dict("records")

        # Most common words (excluding stopwords)
        stop_words = set(stopwords.words("english"))
        tokenized = docs.str.split().explode().str.lower()
        tokenized = tokenized[tokenized.str.isalpha() & ~tokenized.isin(stop_words)]

        # Bigrams and Trigrams
        bigrams = self._extract_ngrams(docs.tolist(), n=2, label="bigram")
        trigrams = self._extract_ngrams(docs.tolist(), n=3, label="trigram")

        return {
            "word_cloud": word_cloud,
            "length_distribution": length_distribution_data,
            "bigrams": bigrams,
            "trigrams": trigrams,
        }

    def _extract_ngrams(self, docs, n=2, top_k=20, label="ngram"):
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(n, n),
            token_pattern=r"\b\w+\b",
        )
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        freq_data = (
            pd.Series(freqs, index=vocab).sort_values(ascending=False).head(top_k)
        )
        return [{label: k, "count": v} for k, v in freq_data.items()]
