import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

# Download necessary resources once
nltk.download("stopwords")
nltk.download("wordnet")

# Load stopwords as a **set** (faster lookup)
# Default + custom stopwords
custom_stopwords = [
    "nt",
    "got",
    "like",
    "just",
    "thing",
    "told",
    "said",
    "asked",
    "really",
    "went",
    "come",
    "back",
    "make",
    "one",
    "would",
    "could",
    "also",
    "hotel",
]
stop_words = set(stopwords.words("english")).union(set(custom_stopwords))

# Initialize lemmatizer once (instead of per function call)
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Optimized text preprocessing:
    1. Removes special characters and numbers.
    2. Tokenizes text faster.
    3. Removes stopwords efficiently.
    4. Applies lemmatization in a batch.
    """
    # 1. Remove special characters & numbers (faster regex)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 2. Tokenize using `wordpunct_tokenize()` (faster than `word_tokenize`)
    tokens = wordpunct_tokenize(text.lower())

    # 3. Remove stopwords (set lookup is faster)
    tokens = [word for word in tokens if word not in stop_words]

    # 4. Apply lemmatization in a batch
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
