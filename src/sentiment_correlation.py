"""OO facade for Task-3 sentiment and correlation analysis.

`SentimentCorrelationAnalysis` builds on the functions in
``scripts.run_correlation`` and provides a small, testable entry point
for running the full pipeline from Python code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from scripts import run_correlation as _rc


@dataclass
class SentimentCorrelationAnalysis:
    """Run sentiment scoring and return correlation for one or more tickers.

    Parameters
    ----------
    news_csv:
        Path to the raw news CSV with columns
        ``headline, url, publisher, date, stock``.
    tickers:
        Iterable of ticker symbols to analyse.
    period:
        History window for price data when calling ``yfinance``.
    output_dir:
        Directory where daily sentiment, merged data and plots are stored.
    lag_days:
        How many days ahead to look when correlating sentiment with
        returns.  ``0`` means same-day returns, ``1`` means next-day
        returns, and so on.
    """

    news_csv: Path
    tickers: Iterable[str]
    period: str = "1y"
    output_dir: Path = Path("outputs/correlation")
    lag_days: int = 0

    def run(self) -> pd.DataFrame:
        """Execute the full Task-3 pipeline and return daily sentiment.

        The returned frame contains ticker-level daily sentiment and
        article counts; correlation summaries and scatter plots are
        written to ``output_dir``.
        """

        out_dir = _rc.ensure_output_dir(self.output_dir)

        news = _rc.load_news(self.news_csv)
        news_with_sentiment = _rc.compute_headline_sentiment(news)
        daily_sentiment = _rc.aggregate_daily_sentiment(news_with_sentiment)

        for raw in self.tickers:
            ticker = raw.strip().upper()
            if not ticker:
                continue

            _rc.correlate_for_ticker(
                ticker=ticker,
                sentiment_daily=daily_sentiment,
                period=self.period,
                out_dir=out_dir,
                lag_days=self.lag_days,
            )

        return daily_sentiment
