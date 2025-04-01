import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string

nltk.download("stopwords")


class EDA:
    def __init__(self, filepath):
        """Initialize EDA with the dataset"""
        self.df = pd.read_csv(filepath)
        self.text_column = "review"
        self.rating_column = "rating"  # Ratings 1-5

        # Assign sentiment based on rating
        self.df["sentiment"] = self.df[self.rating_column].apply(self.assign_sentiment)

    @staticmethod
    def assign_sentiment(rating):
        """Convert numerical rating into categorical sentiment."""
        if rating >= 4:
            return "Positive"
        elif rating == 3:
            return "Neutral"
        else:
            return "Negative"

    def check_data_quality(self):
        """Check missing values and duplicate rows"""
        print("Missing Values:\n", self.df.isnull().sum())
        print("\nDuplicate Rows:", self.df.duplicated().sum())

    def plot_sentiment_distribution(self):
        """Plot distribution of sentiment labels"""
        plt.figure(figsize=(6, 4))
        sns.countplot(x=self.df["sentiment"])
        plt.title("Sentiment Distribution")
        plt.savefig("app/reports/sentiment_distribution.png")
        plt.show()

    def plot_text_length_distribution(self):
        """Plot histogram of review text lengths"""
        self.df["text_length"] = (
            self.df[self.text_column].astype(str).apply(lambda x: len(x.split()))
        )
        plt.figure(figsize=(8, 5))
        sns.histplot(self.df["text_length"], bins=30, kde=True)
        plt.title("Distribution of Review Text Lengths")
        plt.savefig("app/reports/text_length_distribution.png")
        plt.show()

    def generate_word_cloud(self):
        """Generate and plot word cloud from review text"""
        text = " ".join(review for review in self.df[self.text_column].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Reviews")
        plt.savefig("app/reports/word_cloud.png")
        plt.show()

    def get_common_words(self, sentiment, num_words=10):
        """Find the most common words for a given sentiment"""
        text = " ".join(
            self.df[self.df["sentiment"] == sentiment][self.text_column].astype(str)
        )
        words = text.split()
        return Counter(words).most_common(num_words)

    def plot_most_common_words(self, num_words=20):
        """Optimized: Plot the most common words in the dataset"""
        stop_words = set(stopwords.words("english"))  # Faster lookup with set()

        # Extract words efficiently
        words = (
            self.df[self.text_column]
            .astype(str)
            .str.lower()
            .str.split()
            .explode()  # Turns lists into rows for faster processing
        )

        # Remove stopwords and punctuation efficiently
        words = words[~words.isin(stop_words) & words.str.isalpha()]  # Fast filtering

        # Count most common words
        word_counts = words.value_counts().head(num_words)

        # Plot
        plt.figure(figsize=(8, 5))
        sns.barplot(x=word_counts.index, y=word_counts.values)
        plt.xticks(rotation=45)
        plt.title("Top 20 Most Common Words")
        plt.savefig("app/reports/word_frequency.png")
        plt.show()
