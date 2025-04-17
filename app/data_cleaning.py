import pandas as pd
import argparse
import json
from app.services.preprocess import clean_text, clean_to_sentence

# Argument parser to accept file input and output
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Path to input CSV file")
parser.add_argument("file_output", type=str, help="Path to save cleaned CSV file")
parser.add_argument(
    "split_mode",
    type=str,
    default="token",
    choices=["token", "sentence"],
    help="Path to save cleaned CSV file",
)
parser.add_argument("custom_stopwords", type=str, help="JSON string of stopwords")

args = parser.parse_args()
custom_stopwords = json.loads(args.custom_stopwords)

# Load CSV file
df = pd.read_csv(args.file_input)

if args.split_mode == "sentence":
    new_rows = []
    for _, row in df.iterrows():
        sentences = clean_to_sentence(row["review"], custom_stopwords=custom_stopwords)

        for sentence in sentences:
            new_rows.append({"review": sentence})
    cleaned_df = pd.DataFrame(new_rows)
else:
    # Apply text preprocessing
    df["review"] = (
        df["review"]
        .astype(str)
        .apply(lambda t: clean_text(t, custom_stopwords=custom_stopwords))
    )
    cleaned_df = df

# Save the cleaned dataset
cleaned_df.to_csv(args.file_output, index=False)
print(f"✅ Cleaned dataset saved successfully at {args.file_output}!")
