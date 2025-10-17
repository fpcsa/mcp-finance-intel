# server.py
from __future__ import annotations

from fastmcp import FastMCP
from pydantic import BaseModel

from tools.quote import QuoteInput, QuoteOutput, quote_tool
from tools.timeseries import TimeseriesInput, TimeseriesOutput, timeseries_tool
from tools.analyze_asset import AnalyzeInput, AnalyzeOutput, analyze_asset_tool

# mcp = FastMCP("mcp-finance-intel")
mcp = FastMCP("mcp-finance-intel", stateless_http=True)

@mcp.tool
def quote(input: QuoteInput) -> QuoteOutput:
    """
    Get latest quotes for mixed symbols (crypto via ccxt/Binance, equities via yfinance).
    Input: {"symbols": ["BTC/USDT","AAPL"]}
    Output: structured list with last price, 24h % change, volume, provenance.
    """
    return quote_tool(input)

@mcp.tool
def timeseries(input: TimeseriesInput) -> TimeseriesOutput:
    """
    Return OHLCV timeseries for a symbol/interval/limit.
    Input: {"symbol":"BTC/USDT","interval":"1d","limit":90}
    Output: { symbol, interval, bars[], source, asof, bars_used, checksum, not_investment_advice }
    """
    return timeseries_tool(input)

@mcp.tool
def analyze_asset(input: AnalyzeInput) -> AnalyzeOutput:
    """
    Analyze a single asset over a rolling daily window:
    - Trend regime (up/down/sideways) via SMA20/50 + price location
    - Annualized volatility
    - Sharpe ratio (rf=0)
    - Max drawdown
    - RSI(14) last
    Returns structured JSON with provenance & checksum.
    """
    return analyze_asset_tool(input)

if __name__ == "__main__":
    # Run with: python server.py
    # mcp.run()
    mcp.run(transport="http", port=8000)
