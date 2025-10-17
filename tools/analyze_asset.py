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
from pydantic import BaseModel, Field

from adapters.ccxt_adapter import get_ohlcv as ccxt_ohlcv, ohlcv_to_dataframe as ccxt_df
from adapters.yfinance_adapter import get_ohlcv as yf_ohlcv, ohlcv_to_dataframe as yf_df
from analytics.indicators import sma, ema, rsi, macd, atr, bbands
from analytics.risk import (
    annualized_volatility, sharpe_ratio, sortino_ratio,
    max_drawdown_from_prices, value_at_risk, value_at_risk_parametric, conditional_value_at_risk
)

AnalysisMode = Literal["basic", "technical", "risk_plus", "full"]

class AnalyzeInput(BaseModel):
    symbol: str = Field(..., description='e.g., "BTC/USDT" or "AAPL"')
    window_days: int = Field(90, ge=30, le=365)
    mode: AnalysisMode = Field("basic", description="basic | technical | risk_plus | full")

class AnalyzeOutput(BaseModel):
    symbol: str
    window_used_days: int
    trend_regime: str
    volatility_annualized: float
    sharpe: float
    max_drawdown: float
    rsi_last: float | None
    summary: str
    indicators: Optional[Dict[str, Any]] = None
    risk_extras: Optional[Dict[str, Any]] = None
    source: str
    asof: str
    bars_used: int
    mode: AnalysisMode = "basic"
    not_investment_advice: bool = True
    checksum: str


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol

def _iso_now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def analyze_asset_tool(input: AnalyzeInput) -> AnalyzeOutput:
    """
    Routes automatically to either ccxt (crypto) or yfinance (equity) depending on symbol.
    Asks each adapter for slightly more data than requested (window_days + 10) — a safety margin in case some days are missing or holidays.
    Converts the adapter payload into a DataFrame via ohlcv_to_dataframe.
    """
    # Pull daily bars covering at least window_days; adapters trim limits internally
    if _is_crypto(input.symbol):
        payload = ccxt_ohlcv(input.symbol, timeframe="1d", limit=input.window_days + 10)
        df = ccxt_df(payload)
    else:
        payload = yf_ohlcv(input.symbol, interval="1d", limit=input.window_days + 10)
        df = yf_df(payload)

    # Use only last N days with valid closes
    prices = df["close"].dropna().tail(input.window_days)
    bars_used = int(prices.shape[0])

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

    indicators_out: Dict[str, Any] = {}
    risk_out: Dict[str, Any] = {}

    # --- base metrics already computed: prices, returns, vol, sr, mdd, rsi_last, regime ---

    if input.mode in ("technical", "full"):
        # Need OHLC for ATR/BB; df has ['open','high','low','close','volume']
        ema20 = ema(prices, 20)
        ema50 = ema(prices, 50)
        macd_df = macd(prices)  # macd, macd_signal, macd_hist
        atr14 = atr(df["high"], df["low"], df["close"], 14)
        bb = bbands(prices, 20, 2.0)

        indicators_out = {
            "ema20": float(ema20.dropna().iloc[-1]) if not ema20.dropna().empty else None,
            "ema50": float(ema50.dropna().iloc[-1]) if not ema50.dropna().empty else None,
            "macd": float(macd_df["macd"].dropna().iloc[-1]) if not macd_df["macd"].dropna().empty else None,
            "macd_signal": float(macd_df["macd_signal"].dropna().iloc[-1]) if not macd_df["macd_signal"].dropna().empty else None,
            "macd_hist": float(macd_df["macd_hist"].dropna().iloc[-1]) if not macd_df["macd_hist"].dropna().empty else None,
            "atr14": float(atr14.dropna().iloc[-1]) if not atr14.dropna().empty else None,
            "bb_upper": float(bb["bb_upper"].dropna().iloc[-1]) if not bb["bb_upper"].dropna().empty else None,
            "bb_middle": float(bb["bb_middle"].dropna().iloc[-1]) if not bb["bb_middle"].dropna().empty else None,
            "bb_lower": float(bb["bb_lower"].dropna().iloc[-1]) if not bb["bb_lower"].dropna().empty else None,
            "bb_percent": float(bb["bb_percent"].dropna().iloc[-1]) if not bb["bb_percent"].dropna().empty else None,
            "bb_width": float(bb["bb_width"].dropna().iloc[-1]) if not bb["bb_width"].dropna().empty else None,
        }

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

    # Build summary add-ons (kept compact)
    extra_bits = []
    if indicators_out:
        mh = indicators_out.get("macd_hist")
        if mh is not None:
            extra_bits.append(f"MACD hist: {mh:+.3f}")
        atr14_v = indicators_out.get("atr14")
        if atr14_v is not None and bars_used > 0 and prices.iloc[-1] != 0:
            extra_bits.append(f"ATR(14)/P: {atr14_v/prices.iloc[-1]:.2%}")
        if indicators_out.get("bb_percent") is not None:
            extra_bits.append(f"%B: {indicators_out['bb_percent']:.2f}")
    if risk_out:
        extra_bits.append(f"Sortino: {risk_out['sortino']:.2f}")
        extra_bits.append(f"VaR95 (hist): {risk_out['var_hist_95']:.2%}")

    if bars_used == 0:
        summary = f"{input.symbol} (0d) No usable data in the requested window. Signals are indicative only."
              
    summary = (
        f"{input.symbol} ({bars_used}d) regime: {regime}. "
        f"Ann. vol: {vol:.2%}, Sharpe: {sr:.2f}, Max DD: {mdd:.2%}. "
        + (f"RSI(14): {rsi_last:.1f}. " if rsi_last is not None else "")
        + (" | " + " | ".join(extra_bits) if extra_bits else "")
        + " Signals are indicative only."
    )
    
    asof = _iso_now_utc()
    src = payload["source"]

    result = {
        "symbol": input.symbol,
        "window_used_days": bars_used,
        "trend_regime": regime,
        "volatility_annualized": round(vol, 6),
        "sharpe": round(sr, 6),
        "max_drawdown": round(mdd, 6),
        "rsi_last": rsi_last,
        "summary": summary,
        "indicators": indicators_out or None,
        "risk_extras": risk_out or None,
        "source": src,
        "asof": asof,
        "bars_used": bars_used,
        "mode": input.mode,
        "not_investment_advice": True,
    }
    result["checksum"] = _checksum(result)
    return AnalyzeOutput(**result)
