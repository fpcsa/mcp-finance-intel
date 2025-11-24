from __future__ import annotations
from fastmcp import FastMCP
from pydantic import BaseModel

from tools.quote import QuoteInput, QuoteOutput, quote_tool
from tools.timeseries import TimeseriesInput, TimeseriesOutput, timeseries_tool
from tools.analyze_asset import AnalyzeInput, AnalyzeOutput, analyze_asset_tool

# Register server
mcp = FastMCP("mcp-finance-intel", stateless_http=True)

@mcp.tool
def quote(input: QuoteInput) -> QuoteOutput:
    """
    Fetch real-time market quotes for cryptocurrencies and equities.

    Retrieves latest prices, 24h percentage change, and traded volume for each symbol.
    - Cryptos via CCXT (Binance)
    - Equities via Yahoo Finance (yfinance)

    Input example:
    {"symbols": ["BTC/USDT", "AAPL"]}

    Returns structured data with provenance and checksum.
    """
    return quote_tool(input)

@mcp.tool
def timeseries(input: TimeseriesInput) -> TimeseriesOutput:
    """
    Get OHLCV timeseries data for a specific asset.

    Provides open, high, low, close, and volume bars for a given interval and limit.
    Supports crypto (CCXT/Binance) and equities (Yahoo Finance).

    Input example:
    {"symbol": "BTC/USDT", "interval": "1h", "limit": 200}

    Returns structured time series with source, timestamps, and checksum.
    """
    return timeseries_tool(input)

@mcp.tool
def analyze_asset(input: AnalyzeInput) -> AnalyzeOutput:
    """
    Perform full technical and risk analysis of an asset.

    Computes trend regime, window return, volatility, Sharpe ratio, drawdown,
    RSI(14), and optionally EMA, MACD, ATR, Bollinger Bands, and Value-at-Risk.

    Input example:
    {"symbol": "BTC/USDT", "interval": "1d", "limit": 120, "mode": "full"}

    Returns structured analytics with detailed indicators, risk metrics, and metadata.

    | Mode | Description | Included Metrics |
    |------|--------------|------------------|
    | `basic` | Simple technical overview. | SMA(20/50), RSI(14), Volatility, Sharpe, MDD |
    | `technical` | Adds deeper technical indicators. | + EMA(20/50), MACD(12,26,9), ATR(14), Bollinger(20,2) |
    | `risk_plus` | Adds advanced risk metrics. | + Sortino, VaR(95%), CVaR(95%) |
    | `full` | Combines both technical + risk metrics. | All the above |

    """
    return analyze_asset_tool(input)

"""
if __name__ == "__main__":
    # For manual HTTP testing
    mcp.run(transport="http", port=8000)
"""

if __name__ == "__main__":
    # For manual HTTP testing / Docker
    mcp.run(
        transport="http",      # alias of streamable-http on recent FastMCP
        host="0.0.0.0",        # IMPORTANT: listen on all interfaces
        port=8000,
        path="/mcp"            # keep endpoint as /mcp
    )