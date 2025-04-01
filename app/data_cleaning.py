import pandas as pd
import argparse
from app.services.preprocess import clean_text

# Argument parser to accept file input and output
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Path to input CSV file")
parser.add_argument("file_output", type=str, help="Path to save cleaned CSV file")
args = parser.parse_args()

# Load CSV file
df = pd.read_csv(args.file_input)

# Apply text preprocessing
df["review"] = df["review"].astype(str).apply(clean_text)

# Save the cleaned dataset
df.to_csv(args.file_output, index=False)
print(f"✅ Cleaned dataset saved successfully at {args.file_output}!")
