from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from app.services.s3_uploader import upload_file_to_s3

nltk.download("stopwords")


class EDA:
    def __init__(self, df: pd.DataFrame, file_id: str):
        self.df = df
        self.file_id = file_id
        self.text_column = "lemmatized_tokens"
        self.image_urls = {}

    def _upload_plot(self, fig, key_prefix: str) -> str:
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        s3_key = f"eda/{self.file_id}/{key_prefix}.png"
        return upload_file_to_s3(buf, s3_key, content_type="image/png")

    def _plot_ngrams(self, n: int, top_k: int = 20):
        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(n, n),
            token_pattern=r"\b\w+\b",  # Keep punctuation removed
        )
        docs = self.df[self.text_column].astype(str)
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        freqs = X.sum(axis=0).A1
        ngram_freq = (
            pd.Series(freqs, index=vocab).sort_values(ascending=False).head(top_k)
        )

        fig = plt.figure(figsize=(10, 5))
        sns.barplot(x=ngram_freq.values, y=ngram_freq.index)
        plt.title(f"Top {top_k} {n}-grams")
        plt.xlabel("Frequency")
        plt.tight_layout()

        key = f"{n}gram"
        self.image_urls[key] = self._upload_plot(fig, key)
        plt.close(fig)

    def run_eda(self):
        # Text length distribution
        self.df["text_length"] = (
            self.df[self.text_column].astype(str).apply(lambda x: len(x.split()))
        )
        fig_len = plt.figure(figsize=(8, 5))
        sns.histplot(self.df["text_length"], bins=30, kde=True)
        plt.title("Distribution of Review Text Lengths")
        self.image_urls["length_distribution"] = self._upload_plot(
            fig_len, "length_distribution"
        )
        plt.close(fig_len)

        # Word cloud
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            " ".join(self.df[self.text_column].astype(str))
        )
        fig_wc = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        self.image_urls["word_cloud"] = self._upload_plot(fig_wc, "word_cloud")
        plt.close(fig_wc)

        # Word frequency
        stop_words = set(stopwords.words("english"))
        words = self.df[self.text_column].astype(str).str.lower().str.split().explode()
        words = words[~words.isin(stop_words) & words.str.isalpha()]
        word_counts = words.value_counts().head(20)
        fig_freq = plt.figure(figsize=(8, 5))
        sns.barplot(x=word_counts.index, y=word_counts.values)
        plt.xticks(rotation=45)
        plt.title("Top 20 Most Common Words")
        self.image_urls["common_words"] = self._upload_plot(fig_freq, "word_frequency")
        plt.close(fig_freq)

        # Bigram and Trigram plots
        self._plot_ngrams(n=2, top_k=20)
        self._plot_ngrams(n=3, top_k=20)

        return self.image_urls
