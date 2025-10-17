# adapters/ccxt_adapter.py
"""
This module isolates crypto-market data access using the ccxt library.
It standardizes the structure of ticker and OHLCV data returned from Binance, adds provenance metadata, and computes a checksum for reproducibility.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from typing import List, Dict, Any
import ccxt
import pandas as pd

# Default exchange for spot crypto
_EXCHANGE = ccxt.binance()

def _iso_now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()

def _checksum(obj: Any) -> str:
    # deterministic JSON -> sha256
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _ensure_timeframe_supported(exchange, timeframe: str):
    # Many ccxt exchanges publish supported timeframes
    tfs = getattr(exchange, "timeframes", None)
    if isinstance(tfs, dict) and timeframe not in tfs:
        avail = ", ".join(sorted(tfs.keys()))
        raise ValueError(f"timeframe '{timeframe}' not supported by exchange; available: {avail}")

def get_quote(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch latest quote(s) via ccxt (Binance).
    Returns list of dicts with: symbol, last, change_24h_pct, volume_24h, asof, source, bars_used, checksum.
    """
    results = []
    asof = _iso_now_utc()

    for sym in symbols:
        t = _EXCHANGE.fetch_ticker(sym)
        last = float(t.get("last"))
        # ccxt provides percentage change (if available) as 'percentage'
        change_pct = float(t.get("percentage") or 0.0)
        vol = float(t.get("baseVolume") or 0.0)

        payload = {
            "symbol": sym,
            "last": last,
            "change_24h_pct": change_pct,
            "volume_24h": vol,
        }
        provenance = {
            "source": "ccxt:binance",
            "asof": asof,
            "bars_used": 1,
        }
        result = {
            **payload,
            **provenance,
        }
        result["checksum"] = _checksum(result)
        results.append(result)

    return results

def get_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 90) -> Dict[str, Any]:
    """
    Fetch OHLCV via ccxt (Binance).
    Returns { symbol, timeframe, bars, provenance... }
    bars: list[{ts, open, high, low, close, volume}]
    """
    per_call_max = 1500
    asof = _iso_now_utc()
    bars = []

    if limit <= per_call_max:
        raw = _EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        for ts, o, h, l, c, v in raw:
            bars.append({"timestamp": int(ts), "open": float(o), "high": float(h),
                         "low": float(l), "close": float(c), "volume": float(v)})
    else:
        # ---- OPTIONAL PAGINATION ----
        remaining = limit
        since = None  # oldest first; is possible also to anchor to now and page backward
        while remaining > 0:
            take = min(remaining, per_call_max)
            raw = _EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=take, since=since)
            if not raw:
                break
            for row in raw:
                ts, o, h, l, c, v = row
                bars.append({"timestamp": int(ts), "open": float(o), "high": float(h),
                             "low": float(l), "close": float(c), "volume": float(v)})
            remaining -= len(raw)
            # advance 'since' to last+1 to avoid duplicates
            since = raw[-1][0] + 1

    out = {
        "symbol": symbol,
        "interval": timeframe,
        "bars": bars[-limit:],  # keep only requested most-recent 'limit' bars
        "source": "ccxt:binance",
        "asof": asof,
        "bars_used": len(bars[-limit:]),
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
