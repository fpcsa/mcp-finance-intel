# tools/quote.py
"""
It defines the quote MCP tool.
It provides current market quotes (latest price, 24h % change, volume, etc.) for both:

Cryptocurrencies -> via ccxt_adapter

Equities (stocks) -> via yfinance_adapter

It automatically routes each symbol to the right data source, merges results, and returns a consistent structured output.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from adapters.ccxt_adapter import get_quote as ccxt_get_quote
from adapters.yfinance_adapter import get_quote as yf_get_quote


class QuoteInput(BaseModel):
    symbols: List[str] = Field(..., description='List like ["BTC/USDT", "AAPL"]')

class QuoteOutput(BaseModel):
    results: List[Dict[str, Any]]
    not_investment_advice: bool = True


_CRYPTO_PATTERN = re.compile(r".+/.+")

def is_crypto(symbol: str) -> bool:
    # Very simple heuristic: crypto pairs contain a slash (e.g., BTC/USDT)
    return bool(_CRYPTO_PATTERN.match(symbol))

    # TODO: Is possible to extend this to support : or - conventions if needed (e.g., "BTC-USD").

def quote_tool(input: QuoteInput) -> QuoteOutput:
    crypto_syms = [s for s in input.symbols if is_crypto(s)]
    equity_syms = [s for s in input.symbols if not is_crypto(s)]

    results: List[Dict[str, Any]] = []

    # --- Crypto symbols ---
    for sym in crypto_syms:
        try:
            res = ccxt_get_quote([sym])  # always returns a list
            results.extend(res)
        except Exception as e:
            results.append({
                "symbol": sym,
                "source": "ccxt:binance",
                "error": str(e),
                "note": "Failed to fetch crypto quote from ccxt."
            })

    # --- Equity symbols ---
    for sym in equity_syms:
        try:
            res = yf_get_quote([sym])
            results.extend(res)
        except Exception as e:
            results.append({
                "symbol": sym,
                "source": "yfinance",
                "error": str(e),
                "note": "Failed to fetch equity quote from yfinance."
            })

    return QuoteOutput(results=results, not_investment_advice=True)
