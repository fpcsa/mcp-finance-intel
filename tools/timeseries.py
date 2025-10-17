# tools/timeseries.py
"""
Defines the timeseries MCP tool, which provides the historical OHLCV data behind analytics.
It's the bridge between your data adapters and any downstream use (analytics, charts, or LLM reasoning).
"""
from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field, field_validator

from adapters.ccxt_adapter import get_ohlcv as ccxt_ohlcv
from adapters.yfinance_adapter import get_ohlcv as yf_ohlcv

_ALLOWED = {
    # minutes
    "1m","3m","5m","15m","30m",
    # hours
    "1h","2h","4h","6h","8h","12h",
    # days/weeks/months
    "1d","3d","1w","1wk","1M","1mo",
}

class TimeseriesInput(BaseModel):
    symbol: str = Field(..., description='e.g., "BTC/USDT" or "AAPL"')
    interval: str = Field("1d", description='Crypto: many (1m..1M). Equities (yfinance): 1d/1h/1wk.')
    limit: int = Field(90, ge=1, le=5000)  # allow larger; ccxt pagination can handle it

    """
    validate_interval ensures that interval is one of the supported strings.
    It uses Pydantic v2's @field_validator decorator.
    """
    @field_validator("interval")
    @classmethod
    def validate_interval(cls, v: str) -> str:
        if v not in _ALLOWED:
            raise ValueError(f"interval must be one of {sorted(_ALLOWED)}")
        return v
    
class TimeseriesOutput(BaseModel):
    symbol: str
    interval: str
    bars: list[Dict[str, Any]]
    source: str
    asof: str
    bars_used: int
    checksum: str
    not_investment_advice: bool = True


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol

def _normalize_for_ccxt(interval: str) -> str:
    # Alias mapping to ccxt’s canonical keys
    if interval == "1wk": return "1w"
    if interval == "1mo": return "1M"
    return interval  # most are already ccxt-native

def _normalize_for_yf(interval: str) -> str:
    # We keep yfinance conservative here
    if interval in {"1w"}: return "1wk"
    if interval in {"1M","1mo"}: return "1mo"  # only if you later add monthly in the adapter
    return interval

def timeseries_tool(input: TimeseriesInput) -> TimeseriesOutput:
    """
    Routes the request to the appropriate adapter:
    For crypto -> uses CCXT's timeframe parameter.
    For equities -> uses yfinance, which internally maps intervals (_INTERVAL_MAP).

    This routing logic makes the tool multi-domain aware (same API for both asset types).
    Extendible later with other asset classes (e.g., FX, ETFs).
    """
    if _is_crypto(input.symbol):
        norm = _normalize_for_ccxt(input.interval)
        data = ccxt_ohlcv(input.symbol, timeframe=norm, limit=input.limit)
    else:
        norm = _normalize_for_yf(input.interval)
        # yfinance adapter will still only honor 1d/1h/1wk unless future extensions
        data = yf_ohlcv(input.symbol, interval=norm, limit=input.limit)

    data["not_investment_advice"] = True
    data["interval"] = input.interval  # echo user's requested label
    # Validate against schema and return
    return TimeseriesOutput(**data)
