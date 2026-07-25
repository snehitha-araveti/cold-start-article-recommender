"""
Train and save the recommendation model.
"""

import joblib
import pandas as pd

from feature_engineering import create_tfidf_matrix

print("Loading dataset...")

df = pd.read_csv("data/processed/clean_articles.csv")

df["clean_text"] = df["clean_text"].fillna("")

print("Building TF-IDF model...")

vectorizer, tfidf_matrix = create_tfidf_matrix(
    df["clean_text"]
)

print("Saving artifacts...")

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.joblib"
)

joblib.dump(
    tfidf_matrix,
    "models/tfidf_matrix.joblib"
)

joblib.dump(
    df,
    "models/clean_articles.pkl"
)

print("Model saved successfully!")