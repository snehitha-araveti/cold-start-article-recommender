"""
Cold Start Article Recommendation System
Streamlit Application
"""
import joblib
import os
import sys

import pandas as pd
import streamlit as st

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.recommender import recommend_articles
from src.utils import get_shared_keywords

# -------------------------
# Page Configuration
# -------------------------

st.set_page_config(
    page_title="Cold Start Article Recommender",
    page_icon="📰",
    layout="centered"
)

@st.cache_resource
def load_artifacts():

    df = joblib.load("models/clean_articles.pkl")

    vectorizer = joblib.load(
        "models/tfidf_vectorizer.joblib"
    )

    tfidf_matrix = joblib.load(
        "models/tfidf_matrix.joblib"
    )

    return df, vectorizer, tfidf_matrix


df, vectorizer, tfidf_matrix = load_artifacts()

categories = sorted(df["category"].unique())


st.title("📰 Cold Start Article Recommendation System")
st.caption(
    "Find similar news articles using Content-Based Filtering powered by TF-IDF and Cosine Similarity.")
st.divider()
# -------------------------
# Search Box
# -------------------------

search_query = st.text_input(
    "🔍 Search by Headline",
    placeholder="Try: Apple, FIFA World Cup, Artificial Intelligence..."
)
selected_category = st.selectbox(
        "📂 Filter by Category",
        ["All"] + categories
)

st.divider()
# -------------------------
# Search Results
# -------------------------

if search_query:

    search_df = df

    if selected_category != "All":
        search_df = df[
            df["category"] == selected_category
        ]

    matches = search_df[
        search_df["headline"].str.contains(
            search_query,
            case=False,
            na=False
    )
    ].head(20)

    if matches.empty:

        st.warning("No matching articles found.")

    else:

        selected_headline = st.selectbox(
            "Select an article",
            matches["headline"]
        )

        selected_article = matches[
            matches["headline"] == selected_headline
        ].iloc[0]

        with st.container(border=True):

            st.subheader("📰 Selected Article")

            st.markdown(
                f"### {selected_article['headline']}"
            )

            st.write(
                 f"**📂 Category:** {selected_article['category']}"
            )

            if pd.notna(selected_article["short_description"]):

                st.write("**📝 Summary**")

                st.write(selected_article["short_description"])

            if pd.notna(selected_article["date"]):

                st.caption(
                f"📅 Published: {selected_article['date']}"
            )
# -------------------------
# Get Recommendations
# -------------------------

        recommendations = recommend_articles(
            article_index=selected_article.name,
            df=df,
            tfidf_matrix=tfidf_matrix,
            top_n=5
            )

        recommendations["similarity_score"] = (
                recommendations["similarity_score"] * 100
                ).round(2)
        # ---------------------------------
# Add Shared Keywords
# ---------------------------------

        shared_keywords = []

        for rec_index in recommendations.index:
             keywords = get_shared_keywords(
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                article_index=selected_article.name,
                recommended_index=rec_index,
                top_n=5)
             shared_keywords.append(", ".join(keywords))

        recommendations["shared_keywords"] = shared_keywords

        st.subheader("⭐ Top 5 Recommended Articles")

        display_df = recommendations[
            [ "headline","category","similarity_score","shared_keywords"]
        ].copy()

        display_df.rename(
            columns={
                "headline": "Headline",
                "category": "Category",
                "similarity_score": "Similarity (%)",
                "shared_keywords": "Shared Keywords"
            },
            inplace=True
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

st.caption(
    "Built using Python • Streamlit • Scikit-learn • TF-IDF • Cosine Similarity"
)