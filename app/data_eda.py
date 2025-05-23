import argparse
from app.eda.eda_analysis import EDA

# Argument parser for file input
parser = argparse.ArgumentParser()
parser.add_argument("file_input", type=str, help="Path to cleaned dataset CSV file")
args = parser.parse_args()

# Initialize EDA object
eda = EDA(args.file_input)

# Run EDA functions
print("\n🔍 Checking Data Quality...")
eda.check_data_quality()

print("\n📝 Plotting Text Length Distribution...")
eda.plot_text_length_distribution()

print("\n🌥 Generating Word Cloud...")
eda.generate_word_cloud()

print("\n📌 Most Common Words in Positive Reviews:", eda.get_common_words("Positive"))
print("\n📌 Most Common Words in Negative Reviews:", eda.get_common_words("Negative"))

print("\n📈 Plotting Most Common Words...")
eda.plot_most_common_words()

print("\n📈 Plotting Most bigram Common Words...")
eda.plot_top_ngrams(ngram_range=(2, 2), num_ngrams=20)

print("\n📈 Plotting Most trigram Common Words...")
eda.plot_top_ngrams(ngram_range=(3, 3), num_ngrams=20)

print(f"✅ EDA completed successfully using dataset: {args.file_input}")
