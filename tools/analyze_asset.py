# tools/analyze_asset.py
"""
Defines the "analyze_asset" MCP tool — it computes high-level analytics for any asset (equity or crypto) over a specified window of historical prices.

It combines:
- Data retrieval (via adapters)
- Technical indicators (SMA, RSI)
- Risk metrics (volatility, Sharpe ratio, drawdown)
- Simple trend regime classification
- Structured output (validated by Pydantic)

Analysis Mode

basic -> SMA(20/50), RSI, vol, Sharpe, MDD, regime

technical -> adds EMA(20/50), MACD(12,26,9), ATR(14), Bollinger(20,2)

risk_plus -> adds Sortino, VaR(95% historical & parametric), CVaR(95%)

full -> both of the above

"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Dict, Any, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from adapters.ccxt_adapter import get_ohlcv as ccxt_ohlcv, ohlcv_to_dataframe as ccxt_df
from adapters.yfinance_adapter import get_ohlcv as yf_ohlcv, ohlcv_to_dataframe as yf_df
from analytics.indicators import sma, ema, rsi, macd, atr, bbands
from analytics.risk import (
    annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown_from_prices, value_at_risk, value_at_risk_parametric, conditional_value_at_risk
)

AnalysisMode = Literal["basic", "technical", "risk_plus", "full"]

_ALLOWED_INTERVALS = {
    "1m","3m","5m","15m","30m",
    "1h","2h","4h","6h","8h","12h",
    "1d","3d","1w","1wk","1M","1mo",
}

class AnalyzeInput(BaseModel):
    symbol: str = Field(..., description='e.g., "BTC/USDT" or "AAPL"')
    interval: str = Field("1d", description='e.g., "1d", "4h", "1wk"')
    limit: int = Field(120, ge=30, le=2000)
    mode: AnalysisMode = Field("basic", description="basic | technical | risk_plus | full")

    @field_validator("interval")
    @classmethod
    def _valid_interval(cls, v: str) -> str:
        if v not in _ALLOWED_INTERVALS:
            raise ValueError(f"interval must be one of {sorted(_ALLOWED_INTERVALS)}")
        return v

class MetadataInfo(BaseModel):
    """Describes contextual and unit metadata for the analysis result."""
    interval: str = Field(..., description="Data interval used (e.g., '1d', '4h', '1wk')")
    limit: int = Field(..., description="Number of bars used for the analysis window")
    units: Dict[str, str] = Field(
        ...,
        description="Unit semantics for numeric fields (e.g., 'percentage', 'fraction', 'unitless')"
    )

class AnalyzeOutput(BaseModel):
    symbol: str
    window_used_bars: int
    trend_regime: str
    window_return: Optional[float] = Field(
        None, description="Fractional return over the window (e.g 0.10 = +10%)"
    )
    volatility_annualized: float = Field(
        ..., description="Fractional annualized volatility (e.g 0.55 = 55%)"
    )
    sharpe: float = Field(..., description="Unitless risk-adjusted return (Sharpe ratio)")
    max_drawdown: float = Field(
        ..., description="Fractional drawdown (e.g., -0.25 = -25%)"
    )
    summary: str
    indicators: Optional[Dict[str, Any]] = None
    risk_extras: Optional[Dict[str, Any]] = None
    source: str
    asof: str
    bars_used: int
    mode: AnalysisMode = "basic"
    not_investment_advice: bool = True
    metadata: MetadataInfo
    checksum: str

def _is_crypto(symbol: str) -> bool:
    return "/" in symbol

def _iso_now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _last(s: pd.Series | None) -> float | None:
    """
    Ensures getting a float or None (never crashes on empty/NaN).
    """
    if s is None:
        return None
    s = s.dropna()
    return float(s.iloc[-1]) if not s.empty else None

def _last_opt(series: pd.Series | None) -> float | None:
    return _last(series) if series is not None else None

def analyze_asset_tool(input: AnalyzeInput) -> AnalyzeOutput:
    """
    Routes automatically to either ccxt (crypto) or yfinance (equity) depending on symbol.
    Asks each adapter for slightly more data than requested (window_days + 10) — a safety margin in case some days are missing or holidays.
    Converts the adapter payload into a DataFrame via ohlcv_to_dataframe.
    """
    # Pull daily bars covering at least window_days; adapters trim limits internally

    if _is_crypto(input.symbol):
        payload = ccxt_ohlcv(input.symbol, timeframe=input.interval, limit=input.limit + 10)
        df = ccxt_df(payload)
    else:
        # yfinance supports only 1d/1h/1wk
        if input.interval not in {"1d","1h","1wk"}:
            raise ValueError(f"Interval '{input.interval}' not supported for equities via yfinance.")
        payload = yf_ohlcv(input.symbol, interval=input.interval, limit=input.limit + 10)
        df = yf_df(payload)

    # Use only last N days with valid closes
    prices = df["close"].dropna().tail(input.limit)
    bars_used = int(prices.shape[0])

    
    # Compute total return over the analyzed window
    window_return = None
    if bars_used >= 2:
        window_return = float(prices.iloc[-1] / prices.iloc[0] - 1.0)

    # Metrics
    returns = prices.pct_change()
    vol = annualized_volatility(returns)
    sr = sharpe_ratio(returns)  # risk_free 0 for MVP
    mdd = max_drawdown_from_prices(prices)  # negative fraction

    # Indicators
    """
    Uses standard short-term (20d) and medium-term (50d) moving averages.
    rsi_last takes the latest non-NaN RSI value (14-day lookback).
    If RSI unavailable (too few bars) -> None.
    """
    sma_fast = sma(prices, 20)
    sma_slow = sma(prices, 50)
    rsi_series = rsi(prices, 14)
    rsi_last = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else None

    # TODO: Could parameterize lengths later for flexibility.

    # Trend regime (simple rules)
    """
    Simple, rule-based logic:
    Uptrend -> short SMA > long SMA and price above short SMA
    Downtrend -> short SMA < long SMA and price below short SMA
    Else -> "sideways"

    The bars_used >= 50 guard ensures enough data to compute the slow SMA reliably.
    """
    regime = "sideways"
    if bars_used >= 50 and sma_fast.iloc[-1] > sma_slow.iloc[-1] and prices.iloc[-1] > sma_fast.iloc[-1]:
        regime = "up"
    elif bars_used >= 50 and sma_fast.iloc[-1] < sma_slow.iloc[-1] and prices.iloc[-1] < sma_fast.iloc[-1]:
        regime = "down"

    indicators_out: Dict[str, Any] | None = {}
    risk_out: Dict[str, Any] = {}

    # --- base metrics already computed: prices, returns, vol, sr, mdd, rsi_last, regime ---

    # ===== Technical indicators (with bar threshold & safe fallbacks) =====
    indicators_out: Optional[Dict[str, Any]] = None
    
    if input.mode in ("technical", "full"):
        indicators_out = {}

    # Helper: only keep last value if the series has enough bars (guarded by _last anyway)
    def _available(series: pd.Series | None, required: int) -> float | None:
        return _last(series) if bars_used >= required else None

    # Need OHLC for ATR/BB
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Precompute series (safe even with short windows; _last handles NaNs)
    ema20_s = ema(prices, 20) if bars_used >= 20 else None
    ema50_s = ema(prices, 50) if bars_used >= 50 else None
    macd_df   = macd(prices) if bars_used >= 35 else None
    atr14_s = atr(h, l, c, 14) if bars_used >= 15 else None # ATR14 needs at least 14+1
    bb = bbands(prices, 20, 2.0) if bars_used >= 20 else None

    # keep SMA values for symmetry (already computed sma_fast/slow above)
    sma20_last = _available(sma_fast, 20)   # sma_fast is SMA(20)
    sma50_last = _available(sma_slow, 50)   # sma_slow is SMA(50)

    # MACD components (line requires ~26; signal/hist typically require ~35)
    macd_line = macd_sig = macd_hist = None
    if isinstance(macd_df, pd.DataFrame):
        macd_line = _last(macd_df["macd"]) if "macd" in macd_df else None
        macd_sig  = _last(macd_df["macd_signal"]) if "macd_signal" in macd_df else None
        macd_hist = _last(macd_df["macd_hist"]) if "macd_hist" in macd_df else None

    if indicators_out is not None:
        indicators_out.update({
            "ema20": _last_opt(ema20_s),
            "ema50": _last_opt(ema50_s),
            "sma20": sma20_last,
            "sma50": sma50_last,
            "rsi14": rsi_last,
            "macd": macd_line,
            "macd_signal": macd_sig,
            "macd_hist": macd_hist,
            "atr14": _last_opt(atr14_s),
            "bb_upper": _last(bb["bb_upper"]) if isinstance(bb, pd.DataFrame) and "bb_upper" in bb else None,
            "bb_middle": _last(bb["bb_middle"]) if isinstance(bb, pd.DataFrame) and "bb_middle" in bb else None,
            "bb_lower": _last(bb["bb_lower"]) if isinstance(bb, pd.DataFrame) and "bb_lower" in bb else None,
            "bb_percent": _last(bb["bb_percent"]) if isinstance(bb, pd.DataFrame) and "bb_percent" in bb else None,
            "bb_width": _last(bb["bb_width"]) if isinstance(bb, pd.DataFrame) and "bb_width" in bb else None,
        })

    # If everything ends up None/empty, normalize to None to keep schema tidy
    if indicators_out is not None:
        indicators_out = {k: v for k, v in indicators_out.items() if v is not None}

        # If everything ended up empty, use None; otherwise keep the dict
        if not indicators_out:
            indicators_out = None

    # ===== Risk extras (unchanged) =====
    if input.mode in ("risk_plus", "full"):
        sortino_val = sortino_ratio(returns)
        var_hist = value_at_risk(returns, confidence=0.95)
        var_param = value_at_risk_parametric(returns, confidence=0.95)
        cvar = conditional_value_at_risk(returns, confidence=0.95)
        risk_out = {
            "sortino": None if not np.isfinite(sortino_val) else round(sortino_val, 6),
            "var_hist_95": round(var_hist, 6),
            "var_param_95": round(var_param, 6),
            "cvar_95": round(cvar, 6),
        }

    # ===== Summary add-ons (only if we actually have indicators/risk) =====
    extra_bits: list[str] = []
    if window_return is not None:
        extra_bits.append(f"Window ret: {window_return:+.2%}")

    if indicators_out:
        mh = indicators_out.get("macd_hist")
        if mh is not None:
            extra_bits.append(f"MACD hist: {mh:+.3f}")
        atr14_v = indicators_out.get("atr14")
        if atr14_v is not None and bars_used > 0 and prices.iloc[-1] != 0:
            extra_bits.append(f"ATR(14)/P: {atr14_v/prices.iloc[-1]:.2%}")
        bp = indicators_out.get("bb_percent")
        if bp is not None:
            extra_bits.append(f"%B: {bp:.2f}")

    if risk_out:
        if risk_out.get("sortino") is not None:
            extra_bits.append(f"Sortino: {risk_out['sortino']:.2f}")
        if risk_out.get("var_hist_95") is not None:
            extra_bits.append(f"VaR95 (hist): {risk_out['var_hist_95']:.2%}")

    # ===== Summary string =====
    if bars_used == 0:
        summary = f"{input.symbol} (0 bars {input.interval}) No usable data in the requested window. Signals are indicative only."
    else:
        summary = (
            f"{input.symbol} ({bars_used} bars {input.interval}) regime: {regime}. "
            f"Ann. vol: {vol:.2%}, Sharpe: {sr:.2f}, Max DD: {mdd:.2%}. "
            + ("RSI(14): " + f"{rsi_last:.1f}. " if rsi_last is not None else "")
            + (" | " + " | ".join(extra_bits) if extra_bits else "")
            + "  Signals are indicative only."
        )

    asof = _iso_now_utc()
    src = payload["source"]

    metadata = {
        "interval": input.interval,
        "limit": input.limit,
        "units": {
            "window_return": "fraction",
            "volatility_annualized": "fraction",
            "sharpe": "unitless",
            "max_drawdown": "fraction",
            "ema20": "price_units",
            "ema50": "price_units",
            "sma20": "price_units",
            "sma50": "price_units",
            "rsi14": "index(0..100)",
            "macd": "price_units",
            "macd_signal": "price_units",
            "macd_hist": "price_units",
            "atr14": "price_units",
            "bb_upper": "price_units",
            "bb_middle": "price_units", 
            "bb_lower": "price_units",
            "bb_percent": "fraction",
            "bb_width": "fraction", 
            "sortino": "unitless",
            "var_hist_95": "fraction",
            "var_param_95": "fraction",
            "cvar_95": "fraction",
        }
    }

    result = {
        "symbol": input.symbol,
        "window_used_bars": bars_used,
        "trend_regime": regime,
        "window_return": None if window_return is None else round(window_return, 6),
        "volatility_annualized": round(vol, 6),
        "sharpe": round(sr, 6),
        "max_drawdown": round(mdd, 6),
        "summary": summary,
        "indicators": indicators_out or None,
        "risk_extras": risk_out or None,
        "source": src,
        "asof": asof,
        "bars_used": bars_used,
        "mode": input.mode,
        "not_investment_advice": True,
        "metadata": metadata
    }
    
    result["checksum"] = _checksum(result)
    return AnalyzeOutput(**result)
