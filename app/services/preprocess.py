import re
from typing import Dict, Optional
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize, sent_tokenize
import unicodedata
import emoji


# Download necessary resources once
nltk.download("stopwords")
nltk.download("wordnet")

# Load stopwords as a **set** (faster lookup)
# Default + custom stopwords
# custom_stopwords = [
#     "th",
#     "nt",
#     "got",
#     "like",
#     "just",  # General noise
#     "went",
#     "come",
#     "back",
#     "make",
#     "one",  # Action words with low specificity
#     "would",
#     "could",
#     "also",  # Modals
#     "great",
#     "good",
#     "nice",
#     "really",
#     "thing",  # Generic adjectives
#     "time",
#     "day",
#     "night",
#     "people",  # Overly frequent in reviews
#     "hotel",
#     "stay",
#     "stayed",
#     "place",  # Domain-specific noise
# ]
# stop_words = set(stopwords.words("english")).union(set(custom_stopwords))

# Initialize lemmatizer once (instead of per function call)
lemmatizer = WordNetLemmatizer()


def fix_broken_contractions(
    text: str, broken_map: Optional[Dict[str, str]] = None
) -> str:
    """Fix common or user-defined broken contractions."""
    default_map = {
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

    final_map = broken_map if broken_map else default_map
    for broken, fixed in final_map.items():
        text = text.replace(broken, fixed)
    return text


def clean_text(text, custom_stopwords):
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
    stop_words = set(stopwords.words("english")).union(set(custom_stopwords))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Apply lemmatization in a batch
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


def clean_to_token(text):
    return clean_text(text)


def clean_to_sentence(text, custom_stopwords):
    # First try normal spaCy sentence splitting
    sentences = sent_tokenize(text)
    if len(sentences) <= 1 and "," in text:
        sentences = [s.strip() for s in text.split(",") if s.strip()]

    # Fallback: if it's still just one long sentence, try splitting by comma
    if len(sentences) <= 1 and "," in text:
        raw_chunks = text.split(",")
        sentences = [chunk.strip() for chunk in raw_chunks if chunk.strip()]

    # Clean each sentence after splitting
    return [clean_text(s, custom_stopwords) for s in sentences]


def normalize_text(text: str, broken_map: Optional[Dict[str, str]] = None) -> str:
    """
    Normalize text by:
    - Fixing broken contractions (customizable)
    - Expanding contractions
    - Unicode normalization
    - Lowercasing
    - Removing extra whitespace
    """
    text = unicodedata.normalize("NFKC", text)
    text = fix_broken_contractions(text, broken_map)
    text = contractions.fix(text)
    text = text.lower().strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


def get_removed_characters(original: str, cleaned: str) -> list[str]:
    return sorted(list(set(original) - set(cleaned)))


def remove_special_characters(
    text: str,
    remove_special: bool = True,
    remove_numbers: bool = True,
    remove_emoji: bool = True,
) -> tuple[str, list[str]]:
    original = text

    # Remove emoji first (before pattern matching)
    if remove_emoji:
        text = emoji.replace_emoji(text, replace="")

    # Build pattern
    if remove_special and remove_numbers:
        pattern = r"[^a-zA-Z\s]"
    elif remove_special and not remove_numbers:
        pattern = r"[^a-zA-Z0-9\s]"
    elif not remove_special and remove_numbers:
        pattern = r"[0-9]"
    else:
        return text, get_removed_characters(original, text)

    cleaned = re.sub(pattern, "", text)
    removed_chars = get_removed_characters(original, cleaned)
    return cleaned, removed_chars


def tokenize_text(text: str) -> list[str]:
    return wordpunct_tokenize(text)


def remove_stopwords_from_tokens(
    tokens: list[str],
    custom_stopwords: list[str] = [],
    exclude_stopwords: list[str] = [],
) -> dict:
    stop_words = set(stopwords.words("english")).union(set(custom_stopwords))
    stop_words -= set(exclude_stopwords)

    filtered = [t for t in tokens if t.lower() not in stop_words]
    removed = [t for t in tokens if t.lower() in stop_words]

    return {
        "original_tokens": tokens,
        "cleaned_tokens": filtered,
        "removed_stopwords": removed,
    }


def lemmatize_tokens(tokens: list[str]) -> dict:
    lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
    changes = [(o, l) for o, l in zip(tokens, lemmatized) if o != l]
    return {
        "original_tokens": tokens,
        "lemmatized_tokens": lemmatized,
        "changes": changes,
    }
