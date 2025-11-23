import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-1 EDA on financial news dataset")
    parser.add_argument("--csv", required=True, help="Path to news CSV with columns: headline,url,publisher,date,stock")
    parser.add_argument("--output-dir", default="outputs/eda", help="Directory to store EDA outputs")
    parser.add_argument("--topics", type=int, default=10, help="Number of topics for NMF topic modeling")
    parser.add_argument("--max-features", type=int, default=5000, help="Max vocabulary size for TF-IDF")
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"headline", "url", "publisher", "date", "stock"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in CSV: {missing}")

    # Parse date as datetime; errors='coerce' to drop bad rows
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "headline"])

    # Basic derived columns
    df["headline_len"] = df["headline"].astype(str).str.len()
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    df["day_of_week"] = df["date"].dt.day_name()
    return df


def descriptive_stats(df: pd.DataFrame, out_dir: Path) -> None:
    stats = df["headline_len"].describe()
    stats.to_csv(out_dir / "basic_stats.csv")

    plt.figure(figsize=(8, 4))
    sns.histplot(df["headline_len"], bins=50, kde=True)
    plt.title("Headline Length Distribution")
    plt.xlabel("Characters")
    plt.tight_layout()
    plt.savefig(out_dir / "length_hist.png")
    plt.close()

    publishers = df["publisher"].value_counts().rename_axis("publisher").reset_index(name="article_count")
    publishers.to_csv(out_dir / "top_publishers.csv", index=False)


def time_series_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    per_day = df.groupby("date_only").size().rename("article_count").reset_index()
    per_day.to_csv(out_dir / "articles_per_day.csv", index=False)

    plt.figure(figsize=(10, 4))
    sns.lineplot(data=per_day, x="date_only", y="article_count")
    plt.title("Articles per Day")
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "articles_per_day.png")
    plt.close()

    by_hour = df.groupby("hour").size().rename("article_count").reset_index()
    by_hour.to_csv(out_dir / "articles_by_hour.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=by_hour, x="hour", y="article_count")
    plt.title("Articles by Hour of Day")
    plt.tight_layout()
    plt.savefig(out_dir / "articles_by_hour.png")
    plt.close()

    by_dow = df.groupby("day_of_week").size().rename("article_count").reset_index()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow["day_of_week"] = pd.Categorical(by_dow["day_of_week"], categories=order, ordered=True)
    by_dow = by_dow.sort_values("day_of_week")
    by_dow.to_csv(out_dir / "articles_by_dow.csv", index=False)

    plt.figure(figsize=(8, 4))
    sns.barplot(data=by_dow, x="day_of_week", y="article_count")
    plt.title("Articles by Day of Week")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "articles_by_dow.png")
    plt.close()


def publisher_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    # If publishers look like emails, extract domain
    publishers = df["publisher"].astype(str)
    domains = publishers.str.extract(r"@(?P<domain>[^>\s]+)")
    df["publisher_domain"] = domains["domain"]

    domain_counts = (
        df.dropna(subset=["publisher_domain"])
        .groupby("publisher_domain")
        .size()
        .sort_values(ascending=False)
        .rename("article_count")
        .reset_index()
    )
    domain_counts.to_csv(out_dir / "publisher_domains.csv", index=False)


def text_ngrams(df: pd.DataFrame, out_dir: Path, max_features: int = 5000) -> None:
    headlines = df["headline"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        min_df=5,
    )
    X = vectorizer.fit_transform(headlines)
    vocab = vectorizer.get_feature_names_out()

    # Compute mean TF-IDF score per term as a rough importance ranking
    import numpy as np

    mean_scores = X.mean(axis=0).A1
    top_idx = mean_scores.argsort()[::-1][:100]
    top_terms = [(vocab[i], float(mean_scores[i])) for i in top_idx]
    pd.DataFrame(top_terms, columns=["term", "mean_tfidf"]).to_csv(out_dir / "top_ngrams.csv", index=False)


def topic_modeling(df: pd.DataFrame, out_dir: Path, n_topics: int, max_features: int) -> None:
    headlines = df["headline"].astype(str).tolist()
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        stop_words="english",
        min_df=5,
    )
    X = vectorizer.fit_transform(headlines)

    nmf = NMF(n_components=n_topics, random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    vocab = vectorizer.get_feature_names_out()

    top_terms_per_topic: list[list[str]] = []
    for topic_idx, topic_weights in enumerate(H):
        top_indices = topic_weights.argsort()[::-1][:15]
        top_terms = [vocab[i] for i in top_indices]
        top_terms_per_topic.append(top_terms)

    rows = []
    for i, terms in enumerate(top_terms_per_topic):
        rows.append({"topic": i, "top_terms": ", ".join(terms)})
    pd.DataFrame(rows).to_csv(out_dir / "topics_top_terms.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)

    df = load_data(args.csv)
    descriptive_stats(df, out_dir)
    time_series_analysis(df, out_dir)
    publisher_analysis(df, out_dir)
    text_ngrams(df, out_dir, max_features=args.max_features)
    topic_modeling(df, out_dir, n_topics=args.topics, max_features=args.max_features)


if __name__ == "__main__":
    main()
