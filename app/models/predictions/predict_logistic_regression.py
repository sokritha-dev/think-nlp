import pickle
import argparse
import pandas as pd

# Argument parser to accept input file
parser = argparse.ArgumentParser()
parser.add_argument(
    "file_input", type=str, help="Path to the CSV file containing reviews"
)
parser.add_argument("model", type=str, help="Path to the trained model file")
parser.add_argument("vectorizer", type=str, help="Path to the TF-IDF vectorizer file")
parser.add_argument("file_output", type=str, help="Path to save predictions")
args = parser.parse_args()

# Load the trained TF-IDF vectorizer
with open(args.vectorizer, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Load the trained sentiment model
with open(args.model, "rb") as f:
    model = pickle.load(f)

# Load the input reviews file
reviews_df = pd.read_csv(args.file_input)

if "review" not in reviews_df.columns:
    raise ValueError("The input CSV file must contain a 'review' column.")

# Transform the reviews using the TF-IDF vectorizer
reviews_transformed = tfidf_vectorizer.transform(reviews_df["review"].astype(str))

# Predict sentiment for each review
predictions = model.predict(reviews_transformed)

# Save predictions to the file
reviews_df["predicted_sentiment"] = predictions
reviews_df.to_csv(args.file_output, index=False)

print(f"✅ Predictions saved successfully at {args.file_output}")
