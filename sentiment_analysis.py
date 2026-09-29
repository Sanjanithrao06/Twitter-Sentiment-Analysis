"""Twitter sentiment analysis with TF-IDF features and classic ML classifiers.

Trains Bernoulli Naive Bayes and Logistic Regression models on the Twitter
entity-sentiment dataset, once keeping stopwords and once removing them, and
reports results on a held-out split and on the validation set.

Usage:
    python sentiment_analysis.py            # train and evaluate
    python sentiment_analysis.py --plots    # also show EDA plots and word clouds
"""

import argparse
import string
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import BernoulliNB

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "twitter_training.csv.zip"
VALIDATION_PATH = DATA_DIR / "twitter_validation.csv"
COLUMNS = ["ID", "Entity", "Sentiment", "Content"]
SENTIMENT_COLORS = {"Negative": "red", "Neutral": "gold", "Positive": "green"}
RANDOM_STATE = 2

stemmer = PorterStemmer()


def download_nltk_resources():
    for resource in ("punkt", "punkt_tab", "stopwords"):
        nltk.download(resource, quiet=True)


def load_dataset(path):
    df = pd.read_csv(path, names=COLUMNS)
    df = df.dropna(subset=["Content"])
    df["Sentiment"] = df["Sentiment"].replace("Irrelevant", "Neutral")
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def transform_text(text, stop_words=None):
    """Lowercase, tokenize, keep alphanumeric tokens, optionally drop stopwords, and stem."""
    tokens = [t for t in nltk.word_tokenize(text.lower()) if t.isalnum()]
    if stop_words is not None:
        tokens = [t for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(stemmer.stem(t) for t in tokens)


def plot_eda(df):
    counts = df["Sentiment"].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(
        counts,
        labels=counts.index,
        autopct="%0.2f",
        colors=[SENTIMENT_COLORS[s] for s in counts.index],
    )
    plt.title("Sentiment distribution")

    stats = pd.DataFrame({
        "Sentiment": df["Sentiment"],
        "num_char": df["Content"].str.len(),
        "num_words": df["Content"].apply(lambda x: len(nltk.word_tokenize(x))),
        "num_sentences": df["Content"].apply(lambda x: len(nltk.sent_tokenize(x))),
    })
    print(stats.describe())

    for column in ("num_char", "num_words"):
        plt.figure(figsize=(6, 4))
        for sentiment, color in SENTIMENT_COLORS.items():
            sns.histplot(stats.loc[stats["Sentiment"] == sentiment, column], color=color, label=sentiment)
        plt.legend()
        plt.title(f"{column} by sentiment")

    plt.show()


def plot_word_clouds(df):
    from wordcloud import WordCloud

    for sentiment in SENTIMENT_COLORS:
        text = df.loc[df["Sentiment"] == sentiment, "transformed_text"].str.cat(sep=" ")
        plt.figure(figsize=(8, 8))
        plt.imshow(WordCloud(width=1000, height=1000, min_font_size=10).generate(text))
        plt.axis("off")
        plt.title(f"{sentiment} tweets")
    plt.show()


def run_experiment(train, validation, remove_stopwords, show_plots=False):
    title = "WITH stopword removal" if remove_stopwords else "WITHOUT stopword removal"
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")

    stop_words = set(stopwords.words("english")) if remove_stopwords else None
    train = train.assign(transformed_text=train["Content"].apply(transform_text, stop_words=stop_words))
    validation_text = validation["Content"].apply(transform_text, stop_words=stop_words)

    if show_plots:
        plot_word_clouds(train)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        train["transformed_text"], train["Sentiment"], test_size=0.2, random_state=RANDOM_STATE
    )

    tfidf = TfidfVectorizer(max_features=3000)
    X_train = tfidf.fit_transform(X_train_text)
    X_test = tfidf.transform(X_test_text)
    X_validation = tfidf.transform(validation_text)

    bnb = BernoulliNB().fit(X_train, y_train)
    print("\n--- Bernoulli Naive Bayes (test split) ---")
    print(classification_report(y_test, bnb.predict(X_test)))

    grid = GridSearchCV(
        LogisticRegression(solver="liblinear"),
        param_grid={"C": [1, 5, 10]},
        cv=3,
        scoring="accuracy",
        verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"\n--- Logistic Regression (best params: {grid.best_params_}) ---")
    test_pred = grid.predict(X_test)
    print(f"Test split accuracy: {accuracy_score(y_test, test_pred):.3f}")
    print(classification_report(y_test, test_pred))

    validation_pred = grid.predict(X_validation)
    print(f"Validation set accuracy: {accuracy_score(validation['Sentiment'], validation_pred):.3f}")
    print(classification_report(validation["Sentiment"], validation_pred))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots", action="store_true", help="show EDA plots and word clouds")
    args = parser.parse_args()

    download_nltk_resources()

    train = load_dataset(TRAIN_PATH)
    validation = load_dataset(VALIDATION_PATH)
    print(f"Training samples: {len(train)} | Validation samples: {len(validation)}")
    print(train["Sentiment"].value_counts())

    if args.plots:
        plot_eda(train)

    for remove_stopwords in (False, True):
        run_experiment(train, validation, remove_stopwords, show_plots=args.plots)


if __name__ == "__main__":
    main()
