# analytics/risk.py
"""
Defines standard quantitative risk measures commonly used in portfolio and asset analysis:
These metrics help translate raw price or return series into risk-aware summaries, which the MCP server exposes through the analyze_asset tool.

| Metric                          | Measures                                   | Used for                                     |
| ------------------------------- | ------------------------------------------ | -------------------------------------------- |
| **Volatility**                  | Dispersion of returns                      | Basic risk quantification                    |
| **Sharpe ratio**                | Return per unit of (total) volatility      | Performance normalization                    |
| **Max drawdown**                | Worst peak-to-trough loss                  | Tail-risk / “pain” metric                    |
| **Sortino ratio**               | Return per unit of **downside** volatility | Penalizes only harmful volatility            |
| **Value-at-Risk (Historical)**  | Quantile loss at (1-confidence)            | Non-parametric tail-risk threshold           |
| **Conditional VaR (CVaR)**      | Mean loss **beyond** VaR                   | Severity of tail events (expected shortfall) |
| **VaR (Parametric / Gaussian)** | μ - z*σ at given confidence                | Fast, assumption-driven tail estimate        |

Sharpe vs Sortino -> differentiates assets with asymmetric volatility.

VaR/CVaR -> quantify tail risk for LLM summaries or dashboards ("5% chance of losing more than 2.5% in a day").

"""
from __future__ import annotations

import numpy as np
import pandas as pd

def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Annualized volatility = standard deviation of returns * √(number of periods per year)
    252 trading days per year is standard for daily data.

    For hourly data, you could use ~6.5 hours/day * 252 = 1638.

    ddof=1 -> (Delta Degrees of Freedom) sample standard deviation, not population — standard in finance.
    """
    
    # TODO: Add an optional window argument in future for rolling volatilities.
    
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))

def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Sharpe ratio = (Expected excess return / Volatility), annualized

    It measures risk-adjusted performance:
    how much return you earn per unit of risk.

    assumes returns are already percent changes, not log returns — consistent with analyze_asset tool.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    excess = r - (risk_free / periods_per_year)
    mu = float(excess.mean() * periods_per_year)
    vol = float(excess.std(ddof=1) * np.sqrt(periods_per_year))
    return float(mu / vol) if vol != 0 else 0.0

def max_drawdown_from_prices(prices: pd.Series) -> float:
    """
    Maximum drawdown = largest observed drop from a historical peak to a subsequent trough.
    It quantifies worst-case loss from peak to valley during a given period — key for downside risk.

    Returns max drawdown (negative number as percentage, e.g., -0.25 for -25%)
    """
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    dd = (prices / cummax) - 1.0
    return float(dd.min())

def sortino_ratio(
    returns: pd.Series,
    risk_free: float = 0.0,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    The Sortino ratio better reflects asymmetric risk (e.g., assets that move up fast but rarely down shouldn't be penalized for upside variance).

    Sortino ratio = (Mean excess return) / (Downside deviation), annualized.
    Unlike Sharpe, only downside volatility (returns < target) is penalized.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0

    excess = r - (risk_free / periods_per_year)
    downside = r[r < target_return]
    if downside.empty:
        return np.inf  # no downside risk → perfect score

    downside_dev = downside.std(ddof=1) * np.sqrt(periods_per_year)
    mu = excess.mean() * periods_per_year
    return float(mu / downside_dev) if downside_dev != 0 else 0.0

def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Historical Value-at-Risk (VaR).
    Returns the quantile loss at (1 - confidence) level.
    Example: confidence=0.95 → 5% worst-case daily loss.
    Output is negative (e.g., -0.025 means -2.5%).
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    return float(np.percentile(r, (1 - confidence) * 100))

def value_at_risk_parametric(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Parametric (Gaussian, normal-approximation) VaR using mean and std of returns.
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    mu, sigma = r.mean(), r.std(ddof=1)
    z = abs(np.quantile(np.random.normal(0,1,1000000), 1 - confidence))
    return float(mu - z * sigma)

def conditional_value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Conditional VaR (Expected Shortfall) = average loss beyond VaR threshold.

    Useful for LLM summaries — it expresses expected loss beyond VaR
    """
    r = returns.dropna()
    if r.empty:
        return 0.0
    var = value_at_risk(r, confidence)
    tail_losses = r[r <= var]
    if tail_losses.empty:
        return var
    return float(tail_losses.mean())
