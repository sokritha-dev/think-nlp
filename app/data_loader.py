import pandas as pd
from app.services.preprocess import clean_text

# Load CSV file
df = pd.read_csv("app/data/tripadvisor.csv")

# Apply text preprocessing to relevant columns
df["title"] = df["title"].astype(str).apply(clean_text)
df["content"] = df["content"].astype(str).apply(clean_text)
df["tip"] = df["tip"].astype(str).apply(clean_text)
df["response"] = df["response"].astype(str).apply(clean_text)

# Show the cleaned dataset
print(df.head())

# Save the cleaned dataset
df.to_csv("app/data/tripadvisor_cleaned.csv", index=False)

print("Cleaned dataset saved successfully!")
