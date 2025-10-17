# analytics/indicators.py
"""
Reusable technical indicators for the MCP analytics layer.
- SMA, RSI (with pandas_ta fallback already present)
- EMA, MACD, ATR, Bollinger Bands (new)
All functions are vectorized and return pandas objects aligned to input index.

Conventions:
- Prefer pandas_ta when available for speed and battle-tested math.
- Fallbacks implement standard textbook formulas (Wilder where applicable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    _HAS_TA = True
except Exception:
    _HAS_TA = False

# -------------------------
# Moving Averages / Momentum
# -------------------------

def sma(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Simple Moving Average.
    pandas_ta: ta.sma
    fallback: rolling mean with min_periods=length
    """
    if _HAS_TA:
        return ta.sma(series, length=length)
    return series.rolling(window=length, min_periods=length).mean()

def ema(series: pd.Series, length: int = 20) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    pandas_ta: ta.ema
    fallback: EWM(span=length, adjust=False)
    """
    if _HAS_TA:
        return ta.ema(series, length=length)
    # Using span=length matches common EMA convention; adjust=False for recursive form
    return series.ewm(span=length, adjust=False, min_periods=length).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Relative Strength Index (Wilder).
    pandas_ta: ta.rsi
    fallback: Wilder smoothing on gains/losses with EWM alpha=1/length.
    """
    if _HAS_TA:
        return ta.rsi(series, length=length)

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# -------------------------
# Trend / Oscillator Combos
# -------------------------

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.
    Returns DataFrame with columns: macd, macd_signal, macd_hist.
    pandas_ta: ta.macd -> columns MACD_, MACDs_, MACDh_
    fallback: EMA(fast) - EMA(slow), signal = EMA(macd, signal), hist = macd - signal
    """
    if _HAS_TA:
        df = ta.macd(series, fast=fast, slow=slow, signal=signal)
        # pandas_ta column names example: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
        # Normalize to stable names:
        out = pd.DataFrame(index=df.index)
        macd_col = [c for c in df.columns if c.startswith("MACD_")][0]
        sig_col  = [c for c in df.columns if c.startswith("MACDs_")][0]
        hist_col = [c for c in df.columns if c.startswith("MACDh_")][0]
        out["macd"] = df[macd_col]
        out["macd_signal"] = df[sig_col]
        out["macd_hist"] = df[hist_col]
        return out

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
        index=series.index
    )

# -------------------------
# Volatility
# -------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average True Range (Wilder).
    pandas_ta: ta.atr -> 'ATR_' column
    fallback: TR = max(high-low, |high-prev_close|, |low-prev_close|),
              ATR = EWM(alpha=1/length, adjust=False, min_periods=length) of TR
    """
    if _HAS_TA:
        df = ta.atr(high=high, low=low, close=close, length=length)
        # pandas_ta returns a single-column Series/DataFrame; normalize name
        # If DataFrame, it will have a column like ATR_14
        if isinstance(df, pd.DataFrame):
            return df.iloc[:, 0].rename("atr")
        return df.rename("atr")

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    return atr_val.rename("atr")

# -------------------------
# Bands
# -------------------------

def bbands(series: pd.Series, length: int = 20, stdev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with: bb_lower, bb_middle, bb_upper, bb_width, bb_percent
      - bb_middle: SMA(length)
      - bb_upper/lower: middle ± stdev * rolling_std
      - bb_width: (bb_upper - bb_lower) / bb_middle
      - bb_percent (%B): (price - bb_lower) / (bb_upper - bb_lower)
    pandas_ta: ta.bbands -> columns BBL_, BBM_, BBU_, BBB_(bandwidth), BBP_(percent b)
    """
    if _HAS_TA:
        df = ta.bbands(series, length=length, std=stdev)
        out = pd.DataFrame(index=df.index)
        # Example pandas_ta names: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
        bbl = [c for c in df.columns if c.startswith("BBL_")][0]
        bbm = [c for c in df.columns if c.startswith("BBM_")][0]
        bbu = [c for c in df.columns if c.startswith("BBU_")][0]
        bbw = [c for c in df.columns if c.startswith("BBB_")][0]  # bandwidth
        bbp = [c for c in df.columns if c.startswith("BBP_")][0]  # percent b
        out["bb_lower"] = df[bbl]
        out["bb_middle"] = df[bbm]
        out["bb_upper"] = df[bbu]
        out["bb_width"] = df[bbw]
        out["bb_percent"] = df[bbp]
        return out

    mid = sma(series, length)
    # Use population std (ddof=0) as most TA libraries do; min_periods=length aligns with SMA
    std = series.rolling(window=length, min_periods=length).std(ddof=0)
    upper = mid + stdev * std
    lower = mid - stdev * std

    width = (upper - lower) / mid.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)

    return pd.DataFrame(
        {
            "bb_lower": lower,
            "bb_middle": mid,
            "bb_upper": upper,
            "bb_width": width,
            "bb_percent": percent_b,
        },
        index=series.index,
    )
