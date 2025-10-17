# adapters/yfinance_adapter.py
"""
This module is the equity-market counterpart of ccxt_adapter.py.
It wraps the yfinance API (which pulls Yahoo Finance data) into a standardized, predictable format used by the MCP server.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import Dict, Any

import pandas as pd
import yfinance as yf


def _iso_now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def _checksum(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def get_quote(symbols: list[str]) -> list[dict[str, Any]]:
    """
    Fetch latest equity quotes via yfinance.
    Computes 24h % change using previous close vs last close.
    """
    results: list[dict[str, Any]] = []
    asof = _iso_now_utc()

    for sym in symbols:
        tk = yf.Ticker(sym)

        # Use 2 days of daily bars to compute change vs previous close
        hist = tk.history(period="2d", interval="1d", auto_adjust=False)
        if hist.empty:
            last = None
            change_pct = None
            vol = None
        else:
            last = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last
            change_pct = ((last - prev) / prev * 100.0) if prev else None
            vol = float(hist["Volume"].iloc[-1])

        payload = {
            "symbol": sym,
            "last": last,
            "change_24h_pct": change_pct if change_pct is not None else 0.0,
            "volume_24h": vol if vol is not None else 0.0,
        }
        provenance = {
            "source": "yfinance",
            "asof": asof,
            "bars_used": int(len(hist)) if hasattr(hist, "__len__") else 0,
        }
        result = {**payload, **provenance}
        result["checksum"] = _checksum(result)
        results.append(result)

    return results

"""
MCP standardizes the API (MCP server only exposes "interval"),
_INTERVAL_MAP adapts it to Yahoo's expectations (interval + period)
"""
_INTERVAL_MAP = {
    # MCP interval -> yfinance arguments (interval - granularity of data, period - how far back to fetch)
    "1d": ("1d", "1y"),
    "1h": ("1h", "60d"),
    "1wk": ("1wk", "5y"),
    "1mo": ("1mo", "10y")
}

def get_ohlcv(symbol: str, interval: str = "1d", limit: int = 90) -> Dict[str, Any]:
    """
    Fetch OHLCV via yfinance.
    interval: '1d'|'1h'|'1wk' (mapped to yfinance)
    limit: number of rows to return (trimmed from tail)
    """
    yf_interval, default_period = _INTERVAL_MAP.get(interval, ("1d", "1y"))
    tk = yf.Ticker(symbol)
    hist = tk.history(period=default_period, interval=yf_interval, auto_adjust=False)

    if not hist.empty:
        hist = hist.tail(limit)

    asof = _iso_now_utc()
    bars = []
    for dt_idx, row in hist.iterrows():
        ts = int(pd.Timestamp(dt_idx).tz_convert("UTC").timestamp() * 1000)
        bars.append(
            {
                "timestamp": ts,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row.get("Volume", 0.0)),
            }
        )

    out = {
        "symbol": symbol,
        "interval": interval,
        "bars": bars,
        "source": "yfinance",
        "asof": asof,
        "bars_used": len(bars),
    }
    out["checksum"] = _checksum(out)
    return out

def ohlcv_to_dataframe(payload: Dict[str, Any]) -> pd.DataFrame:
    """Utility: convert get_ohlcv payload to pandas DataFrame indexed by UTC datetime."""
    df = pd.DataFrame(payload["bars"])
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("datetime", inplace=True)
    return df
