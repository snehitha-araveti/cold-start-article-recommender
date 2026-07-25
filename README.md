# 📰 Cold Start Article Recommendation System

> A content-based news article recommendation system that recommends relevant articles without requiring historical user ratings or interaction data.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple?logo=pandas)
![Status](https://img.shields.io/badge/Status-Deployment%20Preparation-yellow)

---

## 📌 Overview

The **Cold Start Article Recommendation System** is a content-based recommendation engine designed to recommend similar news articles even when little or no user interaction history is available.

Traditional collaborative filtering systems generally depend on signals such as:

- User ratings
- Click history
- Reading history
- Likes/dislikes
- User-item interaction matrices

These signals are unavailable for a new user or a new article.

This project addresses that challenge by using the **content of the articles themselves**.

The system analyzes article headlines and descriptions, converts textual information into numerical representations using **TF-IDF**, and calculates similarity between articles using **Cosine Similarity**.

The final system is exposed through an interactive **Streamlit web application**.

---

# 🎯 Problem Statement

Recommendation systems often suffer from the **cold-start problem** when sufficient user-item interaction data is unavailable.

### New User

```text
No reading history
        ↓
No interaction data
        ↓
Collaborative filtering cannot reliably recommend
```

### New Article

```text
New article
     ↓
No users have interacted with it yet
     ↓
Limited collaborative signals
```

This project solves the problem using a **content-based approach**:

```text
Article Content
      ↓
Text Preprocessing
      ↓
TF-IDF Representation
      ↓
Cosine Similarity
      ↓
Similarity Ranking
      ↓
Top-N Recommendations
```

---

# 💡 Solution

Instead of asking:

> "What did similar users read?"

the system asks:

> "Which articles have content most similar to this article?"

For example:

```text
Selected Article
"Apple launches new AI-powered device"

                ↓

TF-IDF + Cosine Similarity

                ↓

Recommended Articles

1. Apple introduces new AI technology
2. New AI devices enter the market
3. Technology companies invest in AI
4. Apple expands artificial intelligence research
5. Latest developments in AI hardware
```

The system therefore works without requiring user profiles or historical interactions.

---

# ✨ Key Features

## 🔍 Article Search

Users can search for articles using keywords from the headline.

Examples:

```text
Apple
Climate
Trump
Technology
AI
Politics
```

---

## 🗂️ Category Filtering

Users can narrow article searches using available news categories.

This helps users find relevant articles more efficiently.

---

## 📰 Article Selection

After searching, users can select an article from the matching results.

The application displays information such as:

- Article headline
- Category
- Summary
- Publication date

---

## 🤖 Content-Based Recommendations

The selected article is compared against the article corpus using TF-IDF representations and cosine similarity.

The system returns the **Top 5 most similar articles**.

---

## 📊 Similarity Score

Each recommendation includes a similarity percentage.

Example:

| Article | Similarity |
|---|---:|
| Article A | 82.41% |
| Article B | 76.92% |
| Article C | 71.53% |

A higher score indicates greater textual similarity.

---

## 🔑 Shared Keywords

The application identifies important terms shared between the selected article and recommended articles.

Example:

```text
Why Recommended

AI, technology, Apple, device, innovation
```

This provides an additional explanation for why an article was recommended.

---

# 🧠 Machine Learning Approach

The project uses a **content-based recommendation architecture**.

## 1. Text Preprocessing

Article text is cleaned before feature extraction.

```text
Raw Text
   ↓
Lowercasing
   ↓
HTML Removal
   ↓
URL Removal
   ↓
Punctuation Removal
   ↓
Number Removal
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Clean Text
```

---

# 📐 TF-IDF Feature Extraction

TF-IDF stands for **Term Frequency – Inverse Document Frequency**.

It represents each article as a numerical vector based on the importance of words within the document collection.

The basic idea is:

```text
Important words → Higher TF-IDF weight
Common words    → Lower TF-IDF weight
```

This transforms textual articles into numerical vectors that can be compared mathematically.

---

# 📏 Cosine Similarity

After TF-IDF transformation, the system calculates similarity between articles using **Cosine Similarity**.

```text
Article A → TF-IDF Vector
Article B → TF-IDF Vector

             ↓

       Cosine Similarity

             ↓

       Similarity Score
```

The similarity score is used to rank candidate articles.

The most similar articles are returned as recommendations.

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   HuffPost Dataset   │
                    │   ~209K Articles     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Data Preprocessing  │
                    │                      │
                    │ Cleaning             │
                    │ Tokenization         │
                    │ Stopword Removal     │
                    │ Lemmatization        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    TF-IDF Model      │
                    │  Feature Extraction  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  TF-IDF Matrix       │
                    │  Precomputed Model   │
                    └──────────┬───────────┘
                               │
                               ▼
User ───────► Streamlit UI
                 │
                 ▼
          Search Article
                 │
                 ▼
          Select Article
                 │
                 ▼
        Cosine Similarity
                 │
                 ▼
       Similarity Ranking
                 │
                 ▼
         Top 5 Articles
                 │
                 ▼
      Similarity + Keywords
```

---

# 📊 Dataset

The project uses the **HuffPost News Category Dataset** containing approximately **209,527 articles**.

Relevant fields include:

- `headline`
- `short_description`
- `category`
- `authors`
- `date`

The project combines textual fields to create article representations for recommendation.

The full dataset is **not stored directly in this GitHub repository** because of its size. A dataset source/download link can be added here when the deployment setup is finalized.

---

# 🧪 Exploratory Data Analysis

The project includes exploratory analysis covering:

- Dataset dimensions
- Missing values
- Duplicate records
- Category distribution
- Article text characteristics
- Word-count distribution
- Word frequency analysis
- Word cloud visualization

The analysis is available in:

```text
notebooks/01_eda.ipynb
```

---

# 🖥️ Application Workflow

```text
1. Open application
        ↓
2. Search article
        ↓
3. Select category
        ↓
4. Select article
        ↓
5. View article information
        ↓
6. Generate recommendations
        ↓
7. View Top 5 similar articles
        ↓
8. Inspect similarity and shared keywords
```

---

# 📸 Application Screenshots

Place screenshots in the repository under:

```text
screenshots/
├── home.png
├── search.png
└── recommendations.png
```

### Home Page

![Home Page](screenshots/home.png)

### Article Search

![Article Search](screenshots/search.png)

### Recommendation Results

![Recommendation Results](screenshots/recommendations.png)

---

# 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical operations |
| NLTK | Natural Language Processing |
| Scikit-learn | TF-IDF and similarity |
| Joblib | Model serialization |
| Streamlit | Web application |
| Matplotlib | Data visualization |
| WordCloud | Text visualization |
| Git | Version control |
| GitHub | Source code hosting |

---

# 📁 Project Structure

```text
cold-start-article-recommender/
│
├── app/
│   └── app.py
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── recommender.py
│   ├── train_model.py
│   └── utils.py
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── models/
│   └── README.md
│
├── data/
│   └── ...
│
├── screenshots/
│   ├── home.png
│   ├── search.png
│   └── recommendations.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/snehitha-araveti/cold-start-article-recommender.git
cd cold-start-article-recommender
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

# 🧠 Training and Model Generation

The project separates model generation from application execution.

The training script is:

```text
src/train_model.py
```

It generates reusable artifacts such as:

```text
TF-IDF Vectorizer
TF-IDF Matrix
Processed Article Dataset
```

These artifacts can then be loaded by the Streamlit application.

This avoids rebuilding the complete TF-IDF model every time the application starts.

---

# ⚡ Performance Considerations

The project works with a large dataset of approximately **209K articles**.

During development, directly calculating a full dense cosine-similarity matrix caused unnecessary memory consumption.

Instead, similarity is calculated only for the selected article:

```text
Selected Article
      ↓
One TF-IDF Vector
      ↓
Compare against TF-IDF matrix
      ↓
Rank similarity scores
      ↓
Return Top N
```

This avoids constructing an unnecessary full article-by-article similarity matrix.

---

# 🔄 Cold-Start Advantage

The main advantage of this architecture is that it does **not require user interaction history**.

### Traditional collaborative approach

```text
User History
     ↓
Interactions
     ↓
User-Item Matrix
     ↓
Recommendations
```

### This project

```text
Article Content
     ↓
TF-IDF
     ↓
Content Similarity
     ↓
Recommendations
```

Therefore, recommendations can be generated even when:

- A user is new
- An article is new
- There is no rating history
- There is no click history
- There is no user profile

---

# 🔎 Explainability

For every recommended article, the application provides:

```text
Recommended Article
        +
Similarity Score
        +
Shared Keywords
```

This allows users to understand the textual relationship between the selected article and its recommendation.

---

# 📈 Example

Suppose the user selects:

```text
"Apple announces new artificial intelligence technology"
```

The system may identify:

```text
Recommended Article:
"Apple expands its AI research"

Similarity:
78.42%

Why Recommended:
Apple, AI, technology, research
```

The recommendation is therefore based on measurable textual similarity rather than random selection.

---

# 🔐 Data & Repository Strategy

Large files are intentionally excluded from GitHub using `.gitignore`.

This includes:

```text
Large datasets
Model artifacts
Virtual environments
Python cache files
```

The repository focuses on:

```text
Source Code
+
ML Pipeline
+
EDA
+
Application
+
Documentation
```

This keeps the Git repository lightweight and maintainable.

---

# 🚀 Deployment

The application is designed to be deployed as a Streamlit web application.

The planned deployment architecture is:

```text
GitHub
   │
   ▼
Streamlit Application
   │
   ├── Application Code
   ├── Dependencies
   └── Model Artifacts
           │
           ▼
      Recommendation Engine
```

Large datasets and serialized ML artifacts should be hosted separately rather than committed directly to GitHub.

> Deployment is currently the next project stage. Once the live application is deployed, a public demo link will be added here.

---

# ⚠️ Limitations

### 1. Content-only recommendations

Recommendations depend heavily on textual similarity.

Two articles may be semantically related but use different vocabulary and therefore receive a lower similarity score.

### 2. No personalized recommendations

The current system does not model:

- User preferences
- Reading history
- Click behavior
- Likes/dislikes

Therefore, recommendations are article-to-article rather than user-personalized.

### 3. TF-IDF limitations

TF-IDF captures word importance but does not fully understand semantic meaning.

For example:

```text
"car"
```

and

```text
"automobile"
```

may be semantically related but are treated as different tokens.

### 4. Large-scale similarity computation

Comparing a query article against a very large corpus can still require computational resources.

---

# 🔮 Future Improvements

## 1. Semantic Embeddings

Replace or complement TF-IDF with:

- Sentence Transformers
- BERT embeddings
- Other transformer-based embeddings

This would improve semantic understanding.

## 2. Hybrid Recommendation System

Combine:

```text
Content-Based Filtering
          +
Collaborative Filtering
          +
User Preferences
```

to create personalized recommendations.

## 3. User Profiles

Allow users to build preference profiles based on:

- Selected categories
- Reading history
- Saved articles
- Likes/dislikes

## 4. Recommendation Feedback

Add feedback mechanisms such as:

```text
👍 Relevant
👎 Not Relevant
```

Collected feedback could later be used to improve ranking.

## 5. Advanced Ranking

A future ranking pipeline could combine:

```text
Content Similarity
+
Recency
+
Category Relevance
+
User Preference
+
Popularity
```

## 6. Vector Database

For larger article collections, the system could use vector search infrastructure such as:

- FAISS
- Milvus
- Pinecone
- Weaviate

This would make similarity retrieval more scalable.

---

# 🎓 Learning Outcomes

This project demonstrates practical experience with:

- Recommendation Systems
- Cold-Start Problem
- Natural Language Processing
- Text Preprocessing
- Feature Engineering
- TF-IDF
- Cosine Similarity
- Exploratory Data Analysis
- Model Serialization
- Streamlit Application Development
- Git & GitHub
- ML Project Architecture
- Deployment Preparation
- Memory and Performance Optimization

---

# 📌 Project Status

```text
Phase 1  – Project Planning             ✅
Phase 2  – Dataset Selection            ✅
Phase 3  – Exploratory Data Analysis    ✅
Phase 4  – Text Preprocessing            ✅
Phase 5  – TF-IDF Feature Engineering   ✅
Phase 6  – Recommendation Engine        ✅
Phase 7  – Streamlit Application        ✅
Phase 8  – UI Improvements               ✅
Phase 9  – Model Serialization           ✅
Phase 10 – GitHub Integration            ✅
Phase 11 – Deployment                    🔄
```

---

# 👩‍💻 Author

**Snehitha Araveti**

B.Tech Computer Science Engineering

Developed as part of the **IIT Jammu Data Science & AI Internship**.

---

# 📄 License

This project is intended for educational and portfolio purposes.
