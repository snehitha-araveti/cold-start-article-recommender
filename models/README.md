# Model Artifacts

The trained model artifacts are not stored directly in this GitHub repository
because they are large binary files.

The project uses:

- TF-IDF Vectorizer
- TF-IDF Matrix
- Preprocessed HuffPost article dataset

These artifacts are generated using:

```bash
python src/train_model.py
```

# Dataset

This project uses the HuffPost News Category Dataset containing approximately
209,527 news articles.

The dataset is not committed to this repository because of its size.

## Dataset Columns

The dataset contains information such as:

- headline
- short_description
- category
- authors
- date

## Reproducing the Dataset

Download the dataset from its original source and place the required data
file inside the appropriate data directory.

The preprocessing and model-building pipeline can then be executed using:

```bash
python src/train_model.py
```
