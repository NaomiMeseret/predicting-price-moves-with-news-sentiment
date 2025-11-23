import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
try:
    import pynance as pn
except Exception:  # pragma: no cover - optional dependency
    pn = None

try:
    import talib
except Exception:  # pragma: no cover - optional dependency
    talib = None

import pandas_ta as ta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-2 technical indicators using TA-Lib (if available) and pandas-ta fallback")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of ticker symbols, e.g. AAPL,MSFT")
    parser.add_argument("--period", default="1y", help="History window for yfinance (e.g. 6mo, 1y, 2y)")
    parser.add_argument("--interval", default="1d", help="Bar interval for yfinance (e.g. 1d, 1h)")
    parser.add_argument("--output-dir", default="outputs/indicators", help="Directory to save indicator CSVs and plots")
    return parser.parse_args()


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def download_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df = df.rename(columns=str.title)  # Ensure Open, High, Low, Close, Volume
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    if talib is not None:
        df["SMA_20"] = talib.SMA(close, timeperiod=20)
        df["RSI_14"] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df["MACD"] = macd
        df["MACD_signal"] = macd_signal
        df["MACD_hist"] = macd_hist
    else:
        # Fallback to pandas-ta
        df["SMA_20"] = ta.sma(close, length=20)
        df["RSI_14"] = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]

    # Daily returns for potential correlation with future work
    df["return_1d"] = df["Close"].pct_change()

    # Financial metric using PyNance when available: Sharpe ratio of daily returns
    returns = df["return_1d"].dropna()
    if pn is not None and hasattr(pn, "sharpe_ratio") and not returns.empty:
        sharpe_value = float(pn.sharpe_ratio(returns, risk_free_rate=0.0))
        df["sharpe_ratio"] = sharpe_value
    elif not returns.empty:
        # Manual Sharpe ratio fallback: mean / std of daily returns (risk-free assumed 0)
        sharpe_value = returns.mean() / returns.std(ddof=1)
        df["sharpe_ratio"] = float(sharpe_value)
    else:
        df["sharpe_ratio"] = pd.NA

    return df


def plot_indicators(ticker: str, df: pd.DataFrame, out_dir: Path) -> None:
    fig, (ax_price, ax_rsi, ax_macd) = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})

    ax_price.plot(df.index, df["Close"], label="Close", color="black")
    ax_price.plot(df.index, df["SMA_20"], label="SMA 20", color="blue")
    ax_price.set_title(f"{ticker} Close & SMA20")
    ax_price.legend(loc="upper left")

    ax_rsi.plot(df.index, df["RSI_14"], label="RSI 14", color="purple")
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax_rsi.set_ylabel("RSI")

    ax_macd.plot(df.index, df["MACD"], label="MACD", color="blue")
    ax_macd.plot(df.index, df["MACD_signal"], label="Signal", color="orange")
    ax_macd.bar(df.index, df["MACD_hist"], label="Hist", color="grey", alpha=0.5)
    ax_macd.set_ylabel("MACD")
    ax_macd.legend(loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / f"{ticker}_indicators.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_output_dir(args.output_dir)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    for ticker in tickers:
        df = download_price_data(ticker, period=args.period, interval=args.interval)
        df = compute_indicators(df)
        df.to_csv(out_dir / f"{ticker}_indicators.csv")
        plot_indicators(ticker, df, out_dir)


if __name__ == "__main__":
    main()
