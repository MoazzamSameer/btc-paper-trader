from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create a modeling table from OHLCV data."""

    data = df.copy()
    close = data["Close"]
    volume = data["Volume"] if "Volume" in data.columns else pd.Series(index=data.index, dtype=float)

    data["return_1d"] = close.pct_change(1)
    data["return_3d"] = close.pct_change(3)
    data["return_5d"] = close.pct_change(5)
    data["return_10d"] = close.pct_change(10)
    data["return_20d"] = close.pct_change(20)
    data["volatility_5d"] = data["return_1d"].rolling(5).std()
    data["volatility_10d"] = data["return_1d"].rolling(10).std()
    data["volatility_20d"] = data["return_1d"].rolling(20).std()

    for window in [5, 10, 20, 50, 100, 200]:
        sma = close.rolling(window).mean()
        data[f"sma_{window}"] = sma
        data[f"close_sma_{window}"] = close / sma - 1

    data["ema_12"] = _ema(close, 12)
    data["ema_26"] = _ema(close, 26)
    data["macd"] = data["ema_12"] - data["ema_26"]
    data["macd_signal"] = _ema(data["macd"], 9)
    data["macd_hist"] = data["macd"] - data["macd_signal"]
    data["rsi_14"] = _rsi(close, 14)

    data["high_low_spread"] = (data["High"] - data["Low"]) / close
    data["open_close_spread"] = (data["Close"] - data["Open"]) / data["Open"]
    data["log_volume"] = np.log1p(volume)
    data["volume_zscore_20d"] = (data["log_volume"] - data["log_volume"].rolling(20).mean()) / data["log_volume"].rolling(20).std()

    data["future_return_1d"] = close.shift(-1) / close - 1
    data["future_return_5d"] = close.shift(-5) / close - 1
    data["future_return_10d"] = close.shift(-10) / close - 1

    data["target"] = (data["future_return_5d"] > 0).astype(int)

    features = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "close_sma_5",
        "close_sma_10",
        "close_sma_20",
        "close_sma_50",
        "close_sma_100",
        "close_sma_200",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi_14",
        "high_low_spread",
        "open_close_spread",
        "log_volume",
        "volume_zscore_20d",
    ]

    keep = [c for c in features + ["future_return_1d", "future_return_5d", "future_return_10d", "target"] if c in data.columns]
    table = data[keep].dropna().copy()
    table.attrs["feature_columns"] = features
    return table

