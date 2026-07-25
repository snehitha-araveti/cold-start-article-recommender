"""
Utility functions for the recommendation system.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def get_shared_keywords(
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    article_index: int,
    recommended_index: int,
    top_n: int = 5
):
    """
    Return the top shared TF-IDF keywords
    between two articles.
    """

    feature_names = vectorizer.get_feature_names_out()

    article_vector = tfidf_matrix[article_index].toarray().flatten()
    recommendation_vector = tfidf_matrix[recommended_index].toarray().flatten()

    shared_scores = article_vector * recommendation_vector

    top_indices = shared_scores.argsort()[::-1][:top_n]

    keywords = [
        feature_names[i]
        for i in top_indices
        if shared_scores[i] > 0
    ]

    return keywords