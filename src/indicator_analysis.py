"""Object-oriented helper around the Task-2 indicator analysis.

`IndicatorAnalysis` wraps the functions in ``scripts.run_indicators`` so
that indicators can be computed and plotted from Python code (tests,
notebooks, or other pipelines) without shelling out to the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from scripts import run_indicators as _ri


@dataclass
class IndicatorAnalysis:
    """Compute technical indicators for a list of tickers.

    Parameters
    ----------
    tickers:
        Iterable of ticker symbols (e.g. ``["AAPL", "MSFT"]``).
    period:
        History window passed through to ``yfinance`` (for example ``"1y"``
        or ``"2y"``).
    interval:
        Bar interval used by ``yfinance`` (for example ``"1d"`` or ``"1h"``).
    output_dir:
        Directory where CSVs and plots will be written.
    """

    tickers: List[str]
    period: str = "1y"
    interval: str = "1d"
    output_dir: Path = Path("outputs/indicators")

    def run(self) -> Dict[str, pd.DataFrame]:
        """Download data, compute indicators, and write artifacts.

        Returns a dictionary mapping ticker → enriched price/indicator
        DataFrame.  The implementation is intentionally thin and defers
        to the battle-tested functions in ``scripts.run_indicators``.
        """

        out_dir = _ri.ensure_output_dir(self.output_dir)
        results: Dict[str, pd.DataFrame] = {}

        for raw in self.tickers:
            ticker = raw.strip().upper()
            if not ticker:
                continue

            df = _ri.download_price_data(ticker, period=self.period, interval=self.interval)
            df = _ri.compute_indicators(df)

            df.to_csv(out_dir / f"{ticker}_indicators.csv")
            _ri.plot_indicators(ticker, df, out_dir)

            results[ticker] = df

        return results
