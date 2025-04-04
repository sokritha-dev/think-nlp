import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Argument parser for file input and output
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Path to cleaned dataset CSV file")
parser.add_argument(
    "feature_output", type=str, help="Path to save extracted TF-IDF features"
)
parser.add_argument(
    "vector_output", type=str, help="Path to save extracted TF-IDF features"
)
parser.add_argument("ngram_min", type=str, help="Minimum n-gram size")
parser.add_argument("ngram_max", type=str, help="Maximum n-gram size")

args = parser.parse_args()


class TFIDFExtractor:
    def __init__(self, filepath, max_features=5000, ngram_range=(1, 1)):
        """Initialize TF-IDF extractor"""
        self.df = pd.read_csv(filepath)
        self.text_column = "review"  # Ensure correct column
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=ngram_range
        )
        self.features = None

    def transform_text(self):
        """Convert text reviews into TF-IDF vectors"""
        self.features = self.vectorizer.fit_transform(
            self.df[self.text_column].astype(str)
        )
        return self.features

    # Store important word
    def save_vectorizer(self, filename):
        """Save the TF-IDF vectorizer for later use"""
        with open(filename, "wb") as f:
            pickle.dump(self.vectorizer, f)

    # Convert important word to numeric
    def save_features(self, filename):
        """Save the transformed feature matrix"""
        with open(filename, "wb") as f:
            pickle.dump(self.features, f)


if __name__ == "__main__":
    ngram_range = (int(args.ngram_min), int(args.ngram_max))
    extractor = TFIDFExtractor(args.file_input, ngram_range=ngram_range)
    features = extractor.transform_text()
    extractor.save_vectorizer(filename=args.vector_output)
    print(
        f"✅ TF-IDF vector extraction completed. Features saved at {args.vector_output}!"
    )
    extractor.save_features(args.feature_output)
    print(
        f"✅ TF-IDF feature extraction completed. Features saved at {args.vector_output}!"
    )
