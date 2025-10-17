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

    # TODO: In future, is possible to extend this to support : or - conventions if needed (e.g., "BTC-USD").

def quote_tool(input: QuoteInput) -> QuoteOutput:
    crypto_syms = [s for s in input.symbols if is_crypto(s)]
    equity_syms = [s for s in input.symbols if not is_crypto(s)]

    results = []
    if crypto_syms:
        results.extend(ccxt_get_quote(crypto_syms))
    if equity_syms:
        results.extend(yf_get_quote(equity_syms))

    # Add NIA (Not Investment Advice) flag
    return QuoteOutput(results=results, not_investment_advice=True)

"""
TODO:
Possible future enhancements

Error handling / partial results

Wrap adapter calls in try/except to avoid entire failure if one symbol errors.

e.g.:

try:
    results.extend(ccxt_get_quote(crypto_syms))
except Exception as e:
    results.append({"symbol": sym, "error": str(e)})


Dynamic exchange support

Allow selecting different CCXT exchanges (Binance, Coinbase, etc.) in the future.

Optional metadata fields

Add timestamps or quote latency (time taken) if needed for monitoring.
"""