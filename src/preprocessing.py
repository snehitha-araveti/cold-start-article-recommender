"""
Text preprocessing functions.

This module cleans article text before feature extraction.
"""

import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean article text for TF-IDF.

    Parameters
    ----------
    text : str
        Raw article text.

    Returns
    -------
    str
        Cleaned article text.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text)

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)