"""Object-oriented wrapper around the Task-1 EDA pipeline.

This module exposes a small class, `NewsEDA`, that orchestrates the
procedural functions already implemented in `scripts.run_eda`.  It lets you
reuse the EDA logic from notebooks, unit tests, or future pipelines without
having to call the script via the command line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# We deliberately reuse the existing, well-tested script functions to avoid
# duplicating logic.  This keeps the CLI script and the OO interface consistent.
from scripts import run_eda as _run_eda


@dataclass
class NewsEDA:
    """High-level EDA runner for the financial news dataset.

    Parameters
    ----------
    csv_path:
        Path to the raw news CSV with columns
        ``headline, url, publisher, date, stock``.
    output_dir:
        Directory where EDA artifacts will be written.  The directory is
        created if it does not exist.
    topics:
        Number of latent topics to extract with NMF.
    max_features:
        Maximum vocabulary size for the TF-IDF vectorizer used in both
        n-gram analysis and topic modelling.
    """

    csv_path: Path
    output_dir: Path
    topics: int = 10
    max_features: int = 5000

    def run(self) -> pd.DataFrame:
        """Execute the full EDA pipeline and return the processed DataFrame.

        The steps mirror those in ``scripts/run_eda.py``:

        * load and clean the raw CSV;
        * compute descriptive statistics on headline length and publishers;
        * perform calendar and intraday time-series analysis;
        * extract frequent n-grams with TF-IDF;
        * fit an NMF topic model and save top terms per topic.
        """

        # Ensure the output directory exists before writing artifacts.
        self.output_dir.mkdir(parents=True, exist_ok=True)

        df = _run_eda.load_data(str(self.csv_path))
        _run_eda.descriptive_stats(df, self.output_dir)
        _run_eda.time_series_analysis(df, self.output_dir)
        _run_eda.publisher_analysis(df, self.output_dir)
        _run_eda.text_ngrams(df, self.output_dir, max_features=self.max_features)
        _run_eda.topic_modeling(
            df,
            self.output_dir,
            n_topics=self.topics,
            max_features=self.max_features,
        )
        return df

    @staticmethod
    def load_dataframe(csv_path: Path | str) -> pd.DataFrame:
        """Convenience wrapper around :func:`scripts.run_eda.load_data`.

        This is useful in notebooks or tests that only care about the
        cleaned DataFrame and not the filesystem side effects.
        """

        return _run_eda.load_data(str(csv_path))
