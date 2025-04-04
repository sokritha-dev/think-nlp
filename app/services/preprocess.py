import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize


# Download necessary resources once
nltk.download("stopwords")
nltk.download("wordnet")

# Load stopwords as a **set** (faster lookup)
# Default + custom stopwords
custom_stopwords = [
    "th",
    "hotel",
    "nt",
    "got",
    "like",
    "just",
    "thing",
    "told",
    "said",
    "asked",
    "went",
    "come",
    "back",
    "make",
    "one",
    "would",
    "could",
    "also",
]
stop_words = set(stopwords.words("english")).union(set(custom_stopwords))

# Initialize lemmatizer once (instead of per function call)
lemmatizer = WordNetLemmatizer()


def fix_broken_contractions(text):
    """Manually fix common incorrectly spaced contractions"""
    broken_map = {
        "ca n't": "can't",
        "wo n't": "won't",
        "did n't": "didn't",
        "does n't": "doesn't",
        "is n't": "isn't",
        "was n't": "wasn't",
        "were n't": "weren't",
        "should n't": "shouldn't",
        "would n't": "wouldn't",
        "could n't": "couldn't",
        "do n't": "don't",
        "has n't": "hasn't",
        "have n't": "haven't",
        "had n't": "hadn't",
        "must n't": "mustn't",
        "might n't": "mightn't",
    }

    for broken, fixed in broken_map.items():
        text = text.replace(broken, fixed)
    return text


def clean_text(text):
    """
    Optimized text preprocessing:
    1. Apply Normalization (can't -> cannot)
    2. Removes special characters and numbers.
    3. Tokenizes text faster.
    4. Removes stopwords efficiently.
    5. Applies lemmatization in a batch.
    """
    # 1. Apply Normatilzation
    text = fix_broken_contractions(text)
    text = contractions.fix(text)

    # 2. Remove special characters & numbers (faster regex)
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenize using `wordpunct_tokenize()` (faster than `word_tokenize`)
    tokens = wordpunct_tokenize(text.lower())

    # 4. Remove stopwords (set lookup is faster)
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Apply lemmatization in a batch
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def clean_to_token(text):
    return clean_text(text)


def clean_to_sentence(text):
    # First try normal spaCy sentence splitting
    sentences = sent_tokenize(text)
    if len(sentences) <= 1 and "," in text:
        sentences = [s.strip() for s in text.split(",") if s.strip()]

    # Fallback: if it's still just one long sentence, try splitting by comma
    if len(sentences) <= 1 and "," in text:
        raw_chunks = text.split(",")
        sentences = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    # Clean each sentence after splitting
    return [clean_text(s) for s in sentences]
