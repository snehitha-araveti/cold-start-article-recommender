"""
Recommendation Engine

Provides functions to recommend similar articles
using cosine similarity on TF-IDF vectors.
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def recommend_articles(
    article_index: int,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Recommend the top N similar articles.

    Parameters
    ----------
    article_index : int
        Index of selected article.

    df : pd.DataFrame
        Article dataset.

    tfidf_matrix :
        TF-IDF feature matrix.

    top_n : int
        Number of recommendations.

    Returns
    -------
    pd.DataFrame
        Top recommended articles.
    """

    similarity_scores = cosine_similarity(
        tfidf_matrix[article_index],
        tfidf_matrix
    ).flatten()

    similar_indices = similarity_scores.argsort()[::-1]

    similar_indices = similar_indices[
        similar_indices != article_index
    ]

    top_indices = similar_indices[:top_n]

    recommendations = df.loc[
        top_indices,
        ["headline", "category", "clean_text"]
    ].copy()

    recommendations["similarity_score"] = similarity_scores[top_indices]

    return recommendations


def search_articles(
    keyword: str,
    df: pd.DataFrame,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Search article headlines.
    """

    results = df[
        df["headline"].str.contains(
            keyword,
            case=False,
            na=False
        )
    ]

    return results[["headline", "category"]].head(top_n)


def recommend_articles_by_title(
    title: str,
    df: pd.DataFrame,
    tfidf_matrix,
    top_n: int = 5
):
    """
    Recommend articles using a headline search.
    """

    matches = df[
        df["headline"].str.contains(
            title,
            case=False,
            na=False
        )
    ]

    if matches.empty:
        print("No matching article found.")
        return None

    article_index = matches.index[0]

    print("=" * 70)
    print("Selected Article")
    print("=" * 70)

    print(df.loc[article_index, "headline"])
    print("Category:", df.loc[article_index, "category"])

    print("\nTop Recommendations\n")

    return recommend_articles(
        article_index,
        df,
        tfidf_matrix,
        top_n
    )