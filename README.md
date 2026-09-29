# Twitter Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![License](https://img.shields.io/badge/License-GPLv3-blue)

A Natural Language Processing (NLP) project that classifies tweets about brands, companies and video games as **Positive**, **Negative** or **Neutral**. It uses classic text preprocessing (tokenization, stopword removal, stemming), **TF-IDF** feature extraction, and two machine learning classifiers: **Bernoulli Naive Bayes** and a tuned **Logistic Regression**.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [1. Data Loading and Cleaning](#1-data-loading-and-cleaning)
  - [2. Exploratory Data Analysis](#2-exploratory-data-analysis)
  - [3. Text Preprocessing](#3-text-preprocessing)
  - [4. Feature Extraction (TF-IDF)](#4-feature-extraction-tf-idf)
  - [5. Model Training](#5-model-training)
  - [6. Evaluation](#6-evaluation)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

Social media is one of the richest sources of public opinion. Companies track the sentiment of tweets about their products to measure brand perception, spot customer complaints early, and understand how people react to launches and announcements.

This project builds an end-to-end sentiment classification pipeline:

```
Raw tweets ──► Cleaning ──► Tokenize / Stopwords / Stem ──► TF-IDF ──► Classifier ──► Sentiment
```

The pipeline runs **two experiments**, one that keeps stopwords and one that removes them, so you can see how that preprocessing choice affects model performance.

---

## Features

- Cleans the raw data: removes empty tweets and duplicates, and merges the `Irrelevant` label into `Neutral`
- Exploratory data analysis: class distribution pie chart, tweet length histograms, and summary statistics
- **Word clouds** showing the most frequent words for each sentiment
- Text preprocessing with NLTK: lowercasing, tokenization, removing non-alphanumeric tokens, optional stopword removal, and Porter stemming
- TF-IDF vectorization with the 3,000 most informative terms
- Two classifiers:
  - Bernoulli Naive Bayes (fast baseline)
  - Logistic Regression with hyperparameters tuned by `GridSearchCV`
- Evaluation on a held-out test split **and** on a separate validation dataset
- Detailed per-class precision, recall and F1-score reports
- Plots are optional (`--plots` flag), so the script also runs headless

---

## Dataset

The project uses the **Twitter Entity Sentiment Analysis** dataset. Each tweet is about a specific *entity* (a brand, company or video game) and is labelled with the author's sentiment towards that entity.

### Files

| File | Rows | Description |
|------|-----:|-------------|
| `data/twitter_training.csv.zip` | 74,682 | Training data (zipped CSV, read directly by pandas) |
| `data/twitter_validation.csv` | 1,000 | Separate validation data |

### Columns

The CSV files have no header row. The columns are:

| Column | Description | Example |
|--------|-------------|---------|
| `ID` | Tweet / thread identifier | `352` |
| `Entity` | The brand or game the tweet is about | `Amazon` |
| `Sentiment` | Label: `Positive`, `Negative`, `Neutral` or `Irrelevant` | `Neutral` |
| `Content` | The tweet text | `BBC News - Amazon boss Jeff Bezos rejects claims…` |

### Label Distribution

| Sentiment | Training (raw) | Training (after cleaning) | Validation (after cleaning) |
|-----------|---------------:|--------------------------:|----------------------------:|
| Negative | 22,542 | ~21,650 | 266 |
| Positive | 20,832 | ~19,680 | 277 |
| Neutral | 18,318 | ~30,150 * | 457 * |
| Irrelevant | 12,990 | merged into Neutral | merged into Neutral |
| **Total** | **74,682** | **~71,500** | **1,000** |

\* Includes tweets originally labelled `Irrelevant`.

### Entities (32)

<details>
<summary>Click to expand the full list</summary>

Amazon, ApexLegends, AssassinsCreed, Battlefield, Borderlands, CS-GO, CallOfDuty, CallOfDutyBlackopsColdWar, Cyberpunk2077, Dota2, FIFA, Facebook, Fortnite, Google, GrandTheftAuto(GTA), Hearthstone, HomeDepot, johnson&johnson, LeagueOfLegends, MaddenNFL, Microsoft, NBA2K, Nvidia, Overwatch, PlayStation5(PS5), PlayerUnknownsBattlegrounds(PUBG), RedDeadRedemption(RDR), TomClancysGhostRecon, TomClancysRainbowSix, Verizon, WorldOfCraft, Xbox(Xseries)

</details>

---

## Project Structure

```
Twitter-Sentiment-Analysis/
├── data/
│   ├── twitter_training.csv.zip   # Training dataset (zipped)
│   └── twitter_validation.csv     # Validation dataset
├── sentiment_analysis.py          # Full pipeline: cleaning, EDA, preprocessing, training, evaluation
├── requirements.txt               # Python dependencies
├── LICENSE                        # GNU GPL v3
└── README.md
```

### Main Functions in `sentiment_analysis.py`

| Function | Purpose |
|----------|---------|
| `download_nltk_resources()` | Downloads the NLTK tokenizer and stopword data if missing |
| `load_dataset(path)` | Reads a CSV, drops empty and duplicate rows, and maps `Irrelevant` to `Neutral` |
| `transform_text(text, stop_words)` | Lowercases, tokenizes, filters and stems a tweet |
| `plot_eda(df)` | Shows the class distribution and tweet length plots |
| `plot_word_clouds(df)` | Shows one word cloud per sentiment |
| `run_experiment(...)` | Vectorizes the text, trains both models and prints evaluation reports |
| `main()` | Parses command-line arguments and runs both experiments |

---

## How It Works

### 1. Data Loading and Cleaning

- Loads both datasets with the column names `ID, Entity, Sentiment, Content`.
- Drops rows with missing tweet text.
- Relabels `Irrelevant` tweets as `Neutral`, since neither expresses a clear opinion about the entity. This turns the task into a 3-class problem.
- Removes exact duplicate rows.

### 2. Exploratory Data Analysis

Available when you run with `--plots`:

- **Pie chart** of the sentiment class distribution. After merging, Neutral is the largest class, so the data is moderately imbalanced.
- **Summary statistics** for the number of characters, words and sentences per tweet.
- **Histograms** of tweet length (characters and words), split by sentiment.
- **Word clouds** of the most frequent stemmed words for each sentiment.

### 3. Text Preprocessing

Every tweet goes through `transform_text`:

| Step | Example |
|------|---------|
| Original | `I LOVE the new update!! Playing all night 😍` |
| Lowercase | `i love the new update!! playing all night 😍` |
| Tokenize (NLTK) | `['i', 'love', 'the', 'new', 'update', '!', '!', 'playing', 'all', 'night', '😍']` |
| Keep alphanumeric tokens only | `['i', 'love', 'the', 'new', 'update', 'playing', 'all', 'night']` |
| Remove stopwords *(experiment 2 only)* | `['love', 'new', 'update', 'playing', 'night']` |
| Porter stemming | `love new updat play night` |

### 4. Feature Extraction (TF-IDF)

The cleaned text is converted into numeric vectors with `TfidfVectorizer(max_features=3000)`:

- **TF (term frequency):** how often a word appears in a tweet.
- **IDF (inverse document frequency):** down-weights words that appear in many tweets.
- Only the 3,000 most frequent terms are kept, which keeps the model fast and reduces noise.

The vectorizer is fitted **only on the training split** and then applied to the test split and the validation set. This prevents information from the evaluation data leaking into training.

### 5. Model Training

The cleaned training data is split **80% train / 20% test** (`random_state=2` for reproducibility).

| Model | Details |
|-------|---------|
| **Bernoulli Naive Bayes** | Probabilistic baseline that models whether each term is present or absent. Very fast to train. |
| **Logistic Regression** | Linear classifier (`liblinear` solver). The regularization strength `C` is tuned over `{1, 5, 10}` with 3-fold cross-validated `GridSearchCV`, using accuracy as the metric. |

### 6. Evaluation

Each experiment reports:

- **Bernoulli NB:** a classification report on the test split.
- **Logistic Regression:** the best `C`, then accuracy and a classification report on the test split **and** on the separate validation dataset.

The classification reports include **precision**, **recall**, **F1-score** and **support** for each class, plus macro and weighted averages.

---

## Installation

### Prerequisites

- Python **3.9 or newer**
- `pip`
- About 2 GB of free RAM for vectorization and training

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Sanjanithrao06/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

# 2. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

The required NLTK data (`punkt`, `punkt_tab`, `stopwords`) downloads automatically the first time you run the script.

---

## Usage

### Train and evaluate the models

```bash
python sentiment_analysis.py
```

### Also show EDA plots and word clouds

```bash
python sentiment_analysis.py --plots
```

Each plot opens in a window. Close the windows to let the script continue.

### Show help

```bash
python sentiment_analysis.py --help
```

> **Running time:** Preprocessing about 71,000 tweets with NLTK takes a few minutes, and it runs twice (once per experiment). Training itself is fast.

---

## Sample Output

The script prints output in this structure (the numbers depend on your environment and library versions):

```
Training samples: 71xxx | Validation samples: 1000
Sentiment
Neutral     30xxx
Negative    21xxx
Positive    19xxx

============================================================
WITHOUT stopword removal
============================================================

--- Bernoulli Naive Bayes (test split) ---
              precision    recall  f1-score   support
    Negative       ...
     Neutral       ...
    Positive       ...

Fitting 3 folds for each of 3 candidates, totalling 9 fits

--- Logistic Regression (best params: {'C': ...}) ---
Test split accuracy: 0.xxx
...
Validation set accuracy: 0.xxx
...

============================================================
WITH stopword removal
============================================================
...
```

---

## Tech Stack

| Library | Used For |
|---------|----------|
| [pandas](https://pandas.pydata.org/) | Loading and cleaning the data |
| [NLTK](https://www.nltk.org/) | Tokenization, stopwords and Porter stemming |
| [scikit-learn](https://scikit-learn.org/) | TF-IDF, Naive Bayes, Logistic Regression, GridSearchCV and metrics |
| [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) | Charts and histograms |
| [wordcloud](https://github.com/amueller/word_cloud) | Word cloud visualizations |

---

## Future Improvements

- Handle class imbalance with `class_weight="balanced"` or resampling
- Add n-grams (`ngram_range=(1, 2)`) and try lemmatization instead of stemming
- Compare more models: Linear SVM, Random Forest, XGBoost
- Use transformer models such as BERT, RoBERTa or `cardiffnlp/twitter-roberta-base-sentiment`
- Save the trained model and vectorizer with `joblib` for reuse
- Add a simple web demo (Streamlit or Flask) that predicts the sentiment of any text
- Use the `Entity` column for per-brand sentiment analysis

---

## License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
