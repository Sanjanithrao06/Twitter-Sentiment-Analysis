# Twitter Sentiment Analysis

Sentiment classification of tweets (Positive / Negative / Neutral) using NLP preprocessing, TF-IDF features, and classic machine learning models.

## Overview

The pipeline:

1. **Load & clean** the training and validation datasets (drop empty tweets and duplicates, merge the `Irrelevant` label into `Neutral`).
2. **Explore** the data: sentiment distribution, character / word / sentence counts, and word clouds per sentiment (optional).
3. **Preprocess text**: lowercase, tokenize, keep alphanumeric tokens, optionally remove English stopwords, and apply Porter stemming.
4. **Vectorize** with TF-IDF (top 3,000 features).
5. **Train & evaluate**:
   - Bernoulli Naive Bayes
   - Logistic Regression tuned with `GridSearchCV` over `C ∈ {1, 5, 10}`

The experiment runs twice, **with** and **without** stopword removal, so you can compare the two. Models are evaluated on a 20% held-out split of the training data and on the separate validation set.

## Dataset

Twitter entity-level sentiment data, stored in `data/`:

| File | Description |
|------|-------------|
| `twitter_training.csv.zip` | Training set (~74k tweets, zipped) |
| `twitter_validation.csv` | Validation set (1k tweets) |

Each row has the columns `ID, Entity, Sentiment, Content`.

## Project Structure

```
.
├── data/
│   ├── twitter_training.csv.zip
│   └── twitter_validation.csv
├── sentiment_analysis.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

Requires Python 3.9+.

```bash
git clone https://github.com/Sanjanithrao06/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Train and evaluate the models
python sentiment_analysis.py

# Also show EDA plots and word clouds
python sentiment_analysis.py --plots
```

Required NLTK data (`punkt`, `stopwords`) is downloaded automatically on first run.

## License

This project is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
