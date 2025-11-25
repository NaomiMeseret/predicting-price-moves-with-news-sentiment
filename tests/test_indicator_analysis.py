from pathlib import Path

import numpy as np
import pandas as pd

from src.indicator_analysis import IndicatorAnalysis


def test_indicator_analysis_uses_compute_indicators(monkeypatch, tmp_path):
    """Smoke test IndicatorAnalysis with a patched price downloader.

    The real `download_price_data` would hit the network; here we
    replace it with a deterministic stub to keep the test fast and
    reliable.
    """

    from src import indicator_analysis as ia

    base_index = pd.date_range("2024-01-01", periods=50, freq="D")
    base_prices = pd.Series(np.linspace(100, 120, len(base_index)), index=base_index)
    fake_df = pd.DataFrame({"Close": base_prices})

    def fake_download_price_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
        return fake_df.copy()

    def fake_plot_indicators(ticker, df, out_dir):  # pragma: no cover - trivial
        # No-op: we only care that the function is called without error.
        return None

    monkeypatch.setattr(
        "src.indicator_analysis._ri.download_price_data", fake_download_price_data
    )
    monkeypatch.setattr("src.indicator_analysis._ri.plot_indicators", fake_plot_indicators)

    out_dir = tmp_path / "indicators"
    analysis = IndicatorAnalysis(tickers=["AAPL"], period="1y", interval="1d", output_dir=out_dir)
    results = analysis.run()

    assert "AAPL" in results
    df = results["AAPL"]
    # Columns added by compute_indicators
    for col in ["SMA_20", "RSI_14", "MACD", "MACD_signal", "MACD_hist", "return_1d"]:
        assert col in df.columns

    # CSV artifact should be written for the ticker
    assert (out_dir / "AAPL_indicators.csv").exists()
