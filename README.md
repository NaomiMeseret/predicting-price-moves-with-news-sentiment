# Predicting Price Moves with News Sentiment

A reproducible Python project to analyze how financial news headlines relate to stock price movements. Includes Task 1 (Git/GitHub + EDA) and Task 2 (technical indicators with TA-Lib/PyNance alternative and yfinance).

## Folder Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   └── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
└── scripts/
    ├── __init__.py
    ├── README.md
    ├── run_eda.py
    └── run_indicators.py
```

## Prerequisites

- macOS with Homebrew.
- Python 3.11+
- Git and a GitHub account.

TA-Lib system library (optional, only if you want native TA-Lib instead of pandas-ta fallback):

```
brew install ta-lib
```

## Quickstart

1. Create venv and install dependencies

```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Initialize Git and first commit

```
git init
git add .
git commit -m "feat: initial scaffold for tasks 1 and 2"
```

3. Create GitHub repo and push

- Create an empty repo on GitHub (no README/license).
- Add remote and push:

```
# Replace <YOUR-REPO-URL>
git branch -M main
git remote add origin <YOUR-REPO-URL>
git push -u origin main
```

4. Branch for Task 1

```
git checkout -b task-1
```

5. Add your dataset at `data/news.csv` with columns: `headline`, `url`, `publisher`, `date`, `stock` (timezone in source is UTC-4).

6. Run Task 1 (EDA)

```
python scripts/run_eda.py --csv data/news.csv --output-dir outputs/eda --topics 10
```

Artifacts:

- outputs/eda/basic_stats.csv
- outputs/eda/length_hist.png
- outputs/eda/top_publishers.csv
- outputs/eda/articles_per_day.csv
- outputs/eda/articles_by_hour.csv
- outputs/eda/articles_by_dow.csv
- outputs/eda/top_ngrams.csv
- outputs/eda/topics_top_terms.csv

7. Commit frequently (at least 3/day)

```
git add -A
git commit -m "feat(eda): add topic modeling and publisher analysis"
```

8. Open PR to merge task-1 → main

- Push your branch: `git push -u origin task-1`
- On GitHub, open a Pull Request from `task-1` into `main` and merge after review.

9. Start Task 2 on a new branch

```
git checkout main
git pull
git checkout -b task-2
```

10. Run Task 2 (technical indicators)

```
# TA-Lib is optional if you installed it via Homebrew; otherwise the script uses pandas-ta fallback
python scripts/run_indicators.py --tickers AAPL,MSFT --period 2y --output-dir outputs/indicators
```

Artifacts per ticker:

- outputs/indicators/<TICKER>\_indicators.csv
- outputs/indicators/<TICKER>\_indicators.png

11. Commit and open PR for task-2

```
git add -A
git commit -m "feat(indicators): SMA, RSI, MACD plots and csv"
git push -u origin task-2
```

## CI (GitHub Actions)

- Workflow: `.github/workflows/unittests.yml` runs on pushes/PRs.
- Installs dependencies and runs `pytest` smoke test.
- TA-Lib is not installed on CI by default; the code falls back to pandas-ta.

## Notes

- Dates are parsed and handled as timezone-aware when present. If your `date` is naive but represents UTC-4, you can use `--assume-tz UTC-4` in the EDA script (see `--help`).
- Topic modeling uses scikit-learn (NMF on TF-IDF) to extract top terms.
- For stock prices, we use `yfinance` for convenience.

## Tasks and Contributions

### Task 1 – Git, Environment, and Exploratory Data Analysis

- Set up a reproducible Python environment with `requirements.txt`, virtual environment instructions, and a standard data-science folder structure (`src/`, `scripts/`, `notebooks/`, `tests/`).
- Configured Git and GitHub usage pattern with separate feature branches (`task-1`, `task-2`), frequent descriptive commits, and Pull Requests into `main`.
- Implemented `scripts/run_eda.py` to perform EDA on the financial news dataset (`data/news.csv`) with required columns: `headline`, `url`, `publisher`, `date`, `stock`.
  - **Descriptive statistics:**
    - Computes `headline_len` and saves summary statistics to `outputs/eda/basic_stats.csv`.
    - Plots headline-length distribution as `outputs/eda/length_hist.png`.
  - **Publisher analysis:**
    - Counts articles per `publisher` in `outputs/eda/top_publishers.csv`.
    - If publishers are email-like, extracts domains and aggregates by `publisher_domain` in `outputs/eda/publisher_domains.csv`.
  - **Time series analysis:**
    - Articles per day (`articles_per_day.csv` + `.png`).
    - Articles per hour of day (`articles_by_hour.csv` + `.png`).
    - Articles per day of week (`articles_by_dow.csv` + `.png`).
  - **Text analysis / topic modeling:**
    - Uses TF-IDF (unigrams and bigrams) and NMF to extract topics.
    - Saves top n-grams to `outputs/eda/top_ngrams.csv` and top terms per topic to `outputs/eda/topics_top_terms.csv`.

These outputs are intended to support:

- Identifying the most active publishers and domains.
- Understanding when news volume spikes across calendar time and intraday.
- Discovering common themes such as earnings, guidance changes, downgrades, or regulatory events.

### Task 2 – Quantitative Analysis with TA-Lib, pandas-ta, PyNance, and yfinance

- Implemented `scripts/run_indicators.py` to download OHLCV data for one or more tickers using `yfinance` and compute standard technical indicators.
  - **Data preparation:**
    - Downloads `Open`, `High`, `Low`, `Close`, `Volume` for each ticker using the chosen `period` and `interval`.
  - **Technical indicators:**
    - If TA-Lib is available: computes `SMA_20`, `RSI_14`, and MACD (12, 26, 9) via TA-Lib.
    - If TA-Lib is not available: falls back to `pandas-ta` to compute the same indicators.
  - **Returns and financial metric (PyNance):**
    - Computes daily percentage returns `return_1d`.
    - Uses `pynance` (when available) to compute a Sharpe ratio of `return_1d` (risk-free rate assumed 0).
    - If PyNance is unavailable, falls back to a manual Sharpe ratio (`mean / std` of daily returns).
  - **Outputs and visualizations:**
    - For each ticker, saves a CSV with prices, indicators, returns, and Sharpe ratio as `outputs/indicators/<TICKER>_indicators.csv`.
    - Produces a plot per ticker `outputs/indicators/<TICKER>_indicators.png` with:
      - Close price and 20-day SMA.
      - RSI (14) with overbought/oversold levels.
      - MACD, signal line, and histogram.

These components satisfy the quantitative analysis requirements by:

- Integrating additional market data (OHLCV) with the news-based analysis.
- Computing standard technical indicators used in trading workflows (MA, RSI, MACD).
- Providing a risk-adjusted performance metric (Sharpe ratio) via PyNance or a manual fallback.

## How to Reproduce and Extend

1. Place the financial news dataset at `data/news.csv` and run:

   ```bash
   python scripts/run_eda.py --csv data/news.csv --output-dir outputs/eda --topics 10
   ```

2. For selected tickers (e.g. `AAPL,MSFT`), run:

   ```bash
   python scripts/run_indicators.py --tickers AAPL,MSFT --period 2y --output-dir outputs/indicators
   ```

3. Use the generated CSVs and plots as inputs to further analysis, such as:
   - Correlating news sentiment (derived from headlines) with subsequent daily returns.
   - Studying how technical indicators interact with sentiment around major news events.
