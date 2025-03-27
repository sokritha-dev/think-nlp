import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text):
    """
    Preprocess text by:
    1. Removing special characters, numbers, and punctuation.
    2. Tokenizing words.
    3. Removing stopwords.
    4. Applying lemmatization.
    """
    # 1. Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # 2. Tokenize words (lowercase, tokenize)
    tokens = word_tokenize(text.lower())

    # 3. Removing stopwords
    tokens = [word for word in tokens if word not in stopwords.words("english")]

    # 4. Applying lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Return cleaned text as a single string
    return " ".join(tokens)
