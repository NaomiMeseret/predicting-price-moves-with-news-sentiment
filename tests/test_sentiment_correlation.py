from pathlib import Path

import pandas as pd

from src.sentiment_correlation import SentimentCorrelationAnalysis


def test_sentiment_correlation_end_to_end(monkeypatch, tmp_path):
    """End-to-end style test with a stubbed price downloader.

    Ensures that daily sentiment is computed and correlation artifacts
    are written for at least one ticker without calling external APIs.
    """

    from src import sentiment_correlation as sc

    news_csv = tmp_path / "news.csv"
    news_df = pd.DataFrame(
        {
            "headline": ["Great results", "Terrible outlook"],
            "url": ["u1", "u2"],
            "publisher": ["p1", "p2"],
            "date": ["2024-01-01T10:00:00", "2024-01-02T10:00:00"],
            "stock": ["AAPL", "AAPL"],
        }
    )
    news_df.to_csv(news_csv, index=False)

    # Two days of synthetic returns for AAPL
    price_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-02").date()],
            "Close": [100.0, 101.0],
            "return_1d": [float("nan"), 0.01],
        }
    )

    def fake_download_price_data(ticker: str, period: str) -> pd.DataFrame:
        return price_df.copy()

    monkeypatch.setattr(
        "src.sentiment_correlation._rc.download_price_data", fake_download_price_data
    )

    out_dir = tmp_path / "corr"
    analysis = SentimentCorrelationAnalysis(
        news_csv=news_csv,
        tickers=["AAPL"],
        period="1y",
        output_dir=out_dir,
        lag_days=0,
    )
    daily_sentiment = analysis.run()

    assert not daily_sentiment.empty
    assert set(daily_sentiment["ticker"]) == {"AAPL"}

    # Check that correlation summary and merged data were written
    assert (out_dir / "AAPL_correlation_summary.csv").exists()
    assert (out_dir / "AAPL_sentiment_returns.csv").exists()
