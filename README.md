# 🧠 mcp-finance-intel  
*A Financial Intelligence MCP Server for market analysis and insights*

---

## 🚀 Overview

`mcp-finance-intel` is a **Model Context Protocol (MCP)** server written in **Python**, designed to serve as a **financial intelligence endpoint** for both **LLMs** and **human developers**.

It provides structured, insight-rich analytics for **crypto** (via [`ccxt`](https://github.com/ccxt/ccxt)) and **equities** (via [`yfinance`](https://github.com/ranaroussi/yfinance)), going far beyond simple price data.

---

## 🧩 Features

| Tool | Purpose | Example Usage |
|------|----------|---------------|
| **`quote`** | Get current prices, 24h change %, and volume for crypto or equities. | Latest quote for `BTC/USDT` and `AAPL`. |
| **`timeseries`** | Fetch OHLCV candles for given symbol and interval. | Retrieve 90 days of daily bars. |
| **`analyze_asset`** | Compute advanced analytics — trends, risk, indicators — with multiple analysis modes. | 60-day full analysis of `ETH/USDT`. |

Each tool returns **validated JSON outputs**, **checksums for provenance**, and includes `"not_investment_advice": true`.

---

## 🧠 Analysis Modes (for `analyze_asset`)

| Mode | Description | Included Metrics |
|------|--------------|------------------|
| `basic` | Simple technical overview. | SMA(20/50), RSI(14), Volatility, Sharpe, MDD |
| `technical` | Adds deeper technical indicators. | + EMA(20/50), MACD(12,26,9), ATR(14), Bollinger(20,2) |
| `risk_plus` | Adds advanced risk metrics. | + Sortino, VaR(95%), CVaR(95%) |
| `full` | Combines both technical + risk metrics. | All the above |

---

## 🏗️ Project Structure

```text
mcp-finance-intel/
├─ adapters/
│  ├─ ccxt_adapter.py        # Crypto data (Binance via ccxt)
│  └─ yfinance_adapter.py    # Equity data (Yahoo Finance)
├─ analytics/
│  ├─ indicators.py          # SMA, EMA, RSI, MACD, ATR, BBands
│  └─ risk.py                # Volatility, Sharpe, Sortino, VaR, CVaR
├─ tools/
│  ├─ quote.py               # Implements quote_tool()
│  ├─ timeseries.py          # Implements timeseries_tool()
│  └─ analyze_asset.py       # Implements analyze_asset_tool()
└─ server.py                 # MCP FastMCP entry point and tool registration
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-org/mcp-finance-intel.git
cd mcp-finance-intel
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 🧭 Running the MCP Server
### ▶️ HTTP transport (recommended for testing)
```bash
fastmcp run server:mcp --transport http --port 8000
```
Server will be available at:
http://localhost:8000


### ▶️ STDIO transport (for LLM integration)
```bash
fastmcp run server:mcp --transport stdio
```
Server will be available at:
http://localhost:8000

## Example HTTP Payloads
Each request follows JSON-RPC 2.0 format.
Ensure you set both:
- Content-Type: application/json
- Accept: application/json

### Quote
POST -> http://localhost:8000/mcp  
Body
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "quote",
        "arguments": {
            "input": {
                "symbols": ["BTC/USDT", "AAPL"]
            }
        }
    }
}
```

Response
```json
{
  "result": {
    "structuredContent": {
      "results": [
        {
          "symbol": "BTC/USDT",
          "last": 104512.25,
          "change_24h_pct": -2.35,
          "volume_24h": 38512.7,
          "source": "ccxt:binance",
          "asof": "2025-10-20T11:35:00+00:00",
          "bars_used": 1,
          "checksum": "..."
        },
        {
          "symbol": "AAPL",
          "last": 247.45,
          "change_24h_pct": -0.75,
          "volume_24h": 39698000,
          "source": "yfinance",
          "asof": "2025-10-20T11:35:02+00:00",
          "bars_used": 2,
          "checksum": "..."
        }
      ],
      "not_investment_advice": true
    }
  }
}
```

### Timeseries
POST -> http://localhost:8000/mcp  
Body
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "timeseries",
        "arguments": {
            "input": {
                "symbol": "BTC/USDT",
                "interval": "1h",
                "limit": 120
            }
        }
    }
}
```

Response (truncated)
```json
{
  "result": {
    "structuredContent": {
      "symbol": "BTC/USDT",
      "interval": "1h",
      "bars_used": 120,
      "source": "ccxt:binance",
      "bars": [
        {
          "timestamp": 1734703200000,
          "open": 103412.5,
          "high": 104011.0,
          "low": 103122.2,
          "close": 103982.4,
          "volume": 1845.1
        }
      ],
      "not_investment_advice": true
    }
  }
}
```
### Analyze Asset
POST -> http://localhost:8000/  
Body
```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "analyze_asset",
        "arguments": {
            "input": {
                "symbol": "ETH/USDT",
                "interval": "1d",
                "limit": 60,
                "mode": "full"
            }
        }
    }
}

```
Response (excerpt)
```json
{
  "result": {
    "structuredContent": {
      "symbol": "ETH/USDT",
      "window_used_bars": 60,
      "trend_regime": "down",
      "window_return": -0.16509,
      "volatility_annualized": 0.55352,
      "sharpe": -1.114,
      "max_drawdown": -0.2246,
      "rsi_last": 45.63,
      "summary": "ETH/USDT (60bars 1d) regime: down. Window ret: -16.51%. Ann. vol: 55.35%, Sharpe: -1.11, Max DD: -22.46%. RSI(14): 45.6. | MACD hist: -18.84 | ATR(14)/P: 6.19% | %B: 0.35 | Sortino: -1.42 | VaR95 (hist): -5.65%. Signals are indicative only.",
      "indicators": {
        "ema20": 4114.65,
        "ema50": 4262.51,
        "macd": -108.56,
        "atr14": 249.59,
        "bb_percent": 0.35
      },
      "risk_extras": {
        "sortino": -1.425,
        "var_hist_95": -0.05649,
        "cvar_95": -0.0916
      },
      "source": "ccxt:binance",
      "asof": "2025-10-20T13:07:46+00:00",
      "meta": {
        "interval": "1d",
        "limit": 60,
        "units": {
          "window_return": "fraction",
          "volatility_annualized": "fraction",
          "max_drawdown": "fraction",
          "sharpe": "unitless"
        }
      },
      "not_investment_advice": true
    }
  }
}

```
## Supported Intervals
1m, 3m, 5m, 15m, 30m,
1h, 2h, 4h, 6h, 8h, 12h,
1d, 3d, 1w, 1wk, 1M, 1mo

For equities (yfinance), only 1d, 1h, and 1wk are supported.

## 🔒 Provenance
Every result includes:
- **asof**: ISO-8601 UTC timestamp
- **source**: "ccxt:binance" or "yfinance"
- **checksum**: SHA-256 hash of the serialized result -> ensures integrity and reproducibility.

## ⚠️ Disclaimer
This project is for educational and analytical purposes only.
All responses include ```"not_investment_advice": true``` —
do not use for live trading or financial decision-making.

## Planned Features
- add custom candles for indicators like RSI, MACD, ATR
- `compare_assets`, `portfolio_analyze`, `news_digest`
- Caching, LLM summaries, SQLite persistence 
- full OpenMCP compliance

## 🐳 Optional: Run in Docker
You can run the MCP server in a containerized environment:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["fastmcp", "run", "server:mcp", "--transport", "http", "--port", "8000"]
```

Build & Run
```bash
docker build -t mcp-finance-intel .
docker run -p 8000:8000 mcp-finance-intel
```

## ☕ Support the Project

If you enjoy using **mcp-finance-intel** and want to help me keep building,  
you can support me here:

<a href="https://buymeacoffee.com/fpcsa" target="_blank">
  <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" />
</a>

👉 [Buy Me a Coffee](https://buymeacoffee.com/fpcsa)