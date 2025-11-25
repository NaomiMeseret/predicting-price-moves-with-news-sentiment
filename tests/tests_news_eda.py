from pathlib import Path

import pandas as pd

from src.news_eda import NewsEDA


def test_news_eda_run_creates_expected_outputs(tmp_path):
    """End-to-end smoke test for the NewsEDA wrapper.

    Uses a tiny synthetic dataset to ensure the pipeline runs and
    produces the key CSV artifacts without raising.
    """

    csv_path = tmp_path / "news.csv"
    df = pd.DataFrame(
        {
            "headline": ["Stock soars on earnings", "Shares fall after downgrade"],
            "url": ["http://example.com/a", "http://example.com/b"],
            "publisher": ["news@example.com", "research@example.com"],
            "date": ["2024-01-01T10:00:00", "2024-01-02T15:30:00"],
            "stock": ["AAPL", "AAPL"],
        }
    )
    df.to_csv(csv_path, index=False)

    out_dir = tmp_path / "eda"
    eda = NewsEDA(csv_path=csv_path, output_dir=out_dir, topics=2, max_features=100)
    result_df = eda.run()

    # Basic sanity checks on the returned frame
    assert not result_df.empty
    assert "headline_len" in result_df.columns

    # Key output artifacts should exist
    expected_files = [
        "basic_stats.csv",
        "length_hist.png",
        "top_publishers.csv",
        "articles_per_day.csv",
        "articles_by_hour.csv",
        "articles_by_dow.csv",
        "top_ngrams.csv",
        "topics_top_terms.csv",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), f"missing EDA artifact: {name}"
