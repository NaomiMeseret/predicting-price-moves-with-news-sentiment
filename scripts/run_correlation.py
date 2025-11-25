import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from textblob import TextBlob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task-3: sentiment analysis on news headlines and correlation with daily stock returns",
    )
    parser.add_argument(
        "--news-csv",
        required=True,
        help="Path to news CSV with columns: headline, url, publisher, date, stock",
    )
    parser.add_argument(
        "--tickers",
        required=True,
        help="Comma-separated list of ticker symbols, e.g. AAPL,MSFT",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="History window for yfinance (e.g. 6mo, 1y, 2y)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/correlation",
        help="Directory to save sentiment, merged data, and plots",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=0,
        help=(
            "Number of days to shift returns relative to sentiment (0 = same day, "
            "1 = next-day return, etc.)"
        ),
    )
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_news(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"headline", "url", "publisher", "date", "stock"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"News CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["headline"] = df["headline"].astype(str).fillna("")
    df["stock"] = df["stock"].astype(str).str.upper().str.strip()

    # Parse timezone-aware datetimes when possible; dataset uses UTC-4 in source.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df.dropna(subset=["date"])  # drop rows with unparseable dates
    df["news_date"] = df["date"].dt.date
    return df


def compute_headline_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    def polarity(text: str) -> float:
        try:
            return float(TextBlob(text).sentiment.polarity)
        except Exception:
            return 0.0

    out = df.copy()
    out["sentiment"] = out["headline"].astype(str).apply(polarity)
    return out


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["stock", "news_date"], as_index=False)["sentiment"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped = grouped.rename(
        columns={
            "stock": "ticker",
            "news_date": "date",
            "mean": "sentiment_mean",
            "count": "n_articles",
        }
    )
    return grouped


def download_price_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No price data returned for ticker {ticker}")

    df = df.rename(columns=str.title)
    df["return_1d"] = df["Close"].pct_change()
    df["date"] = df.index.date
    return df[["date", "Close", "return_1d"]].reset_index(drop=True)


def correlate_for_ticker(
    ticker: str,
    sentiment_daily: pd.DataFrame,
    period: str,
    out_dir: Path,
    lag_days: int = 0,
) -> None:
    ticker = ticker.upper().strip()
    sent = sentiment_daily.loc[sentiment_daily["ticker"] == ticker].copy()
    if sent.empty:
        # No sentiment for this ticker; nothing to do.
        return

    prices = download_price_data(ticker, period=period)

    merged = prices.merge(sent, on="date", how="left")

    # Optionally look at next-day (or further) returns relative to today's sentiment.
    if lag_days != 0:
        merged["return_shifted"] = merged["return_1d"].shift(-lag_days)
        ret_col = "return_shifted"
    else:
        ret_col = "return_1d"

    valid = merged.dropna(subset=["sentiment_mean", ret_col])
    if not valid.empty:
        corr_value = float(valid["sentiment_mean"].corr(valid[ret_col]))
        n_obs = int(len(valid))
    else:
        corr_value = float("nan")
        n_obs = 0

    sent.to_csv(out_dir / f"{ticker}_sentiment_by_day.csv", index=False)
    merged.to_csv(out_dir / f"{ticker}_sentiment_returns.csv", index=False)

    summary = pd.DataFrame(
        {
            "ticker": [ticker],
            "lag_days": [lag_days],
            "n_obs": [n_obs],
            "pearson_correlation": [corr_value],
        }
    )
    summary.to_csv(out_dir / f"{ticker}_correlation_summary.csv", index=False)

    if not valid.empty:
        plot_sentiment_vs_returns(ticker, valid, ret_col, corr_value, out_dir)


def plot_sentiment_vs_returns(
    ticker: str,
    df: pd.DataFrame,
    ret_col: str,
    corr_value: float,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["sentiment_mean"], df[ret_col], alpha=0.6)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Average daily sentiment (polarity)")
    ax.set_ylabel("Daily return")
    ax.set_title(f"{ticker}: sentiment vs daily return (r = {corr_value:.2f})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{ticker}_sentiment_vs_return.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)

    news = load_news(args.news_csv)
    news_with_sentiment = compute_headline_sentiment(news)
    daily_sentiment = aggregate_daily_sentiment(news_with_sentiment)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    for ticker in tickers:
        correlate_for_ticker(
            ticker=ticker,
            sentiment_daily=daily_sentiment,
            period=args.period,
            out_dir=out_dir,
            lag_days=args.lag_days,
        )


if __name__ == "__main__":
    main()
