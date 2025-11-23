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
