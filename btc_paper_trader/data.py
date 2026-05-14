from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketData:
    frame: pd.DataFrame
    source: str


def download_btc_usd(start: str, end: str | None = None, cache_path: Path | None = None) -> MarketData:
    """Download BTC-USD daily candles and optionally cache them to disk."""

    if cache_path and cache_path.exists():
        preview = cache_path.read_text().splitlines()[:2]
        if any("Ticker" in line for line in preview):
            frame = pd.read_csv(cache_path, header=[0, 1], index_col=0)
            frame = frame.drop(index="Date", errors="ignore")
            frame.index = pd.to_datetime(frame.index, errors="coerce")
            frame = frame.loc[frame.index.notna()].copy()
            frame.columns = [col[0] if isinstance(col, tuple) else col for col in frame.columns]
        else:
            frame = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
            if getattr(frame.index, "tz", None) is not None:
                frame.index = frame.index.tz_localize(None)
        return MarketData(frame=frame, source=f"cache:{cache_path}")

    raw = yf.download("BTC-USD", start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError("No BTC-USD data returned from Yahoo Finance.")

    frame = raw.rename_axis("Date").copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.dropna().sort_index()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [col[0] if col[0] != "Ticker" else col[1] for col in frame.columns]
    frame.columns = [str(col) for col in frame.columns]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index_label="Date")

    return MarketData(frame=frame, source="yfinance")
