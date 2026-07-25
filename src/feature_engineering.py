"""
Feature engineering module.

Creates TF-IDF representations of article text.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def create_tfidf_matrix(
    texts,
    max_features: int = 10000,
    ngram_range=(1, 2)
):
    """
    Create TF-IDF feature matrix.

    Parameters
    ----------
    texts : pd.Series
        Cleaned article text.

    max_features : int
        Maximum vocabulary size.

    ngram_range : tuple
        N-gram range.

    Returns
    -------
    vectorizer
        Trained TF-IDF vectorizer.

    tfidf_matrix
        Sparse TF-IDF matrix.
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=ngram_range
    )

    texts = texts.fillna("").astype(str)

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix