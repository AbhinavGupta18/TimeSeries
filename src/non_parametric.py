# src/non_parametric.py
"""Volatility estimation utilities (WMA & EWMA)."""
import numpy as np
import pandas as pd
from .utils import save_fig

def wma_volatility(returns, window=30):
    """Weighted Moving Average volatility with linearly decaying weights.
    Returns a pandas Series aligned with ``returns`` index.
    """
    n = len(returns)
    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    vol = pd.Series(np.nan, index=returns.index)
    for i in range(window, n):
        chunk = returns.iloc[i - window:i].values
        weighted_var = np.sum(weights * chunk ** 2) - (np.sum(weights * chunk)) ** 2
        vol.iloc[i] = np.sqrt(max(weighted_var, 0))
    return vol

def ewma_volatility(returns, lam=0.94):
    """Exponentially Weighted Moving Average volatility (RiskMetrics)."""
    n = len(returns)
    var = np.zeros(n)
    var[0] = returns.iloc[0] ** 2
    for i in range(1, n):
        var[i] = lam * var[i - 1] + (1 - lam) * returns.iloc[i - 1] ** 2
    return pd.Series(np.sqrt(var), index=returns.index)

def plot_volatility(returns, wma_vol, ewma_vol):
    """Plot absolute returns together with WMA and EWMA volatilities.
    Saves the figure to ``plots/models/volatility.png``.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 5))
    plt.plot(returns.index, returns.abs(), color='gray', alpha=0.25, label='|Returns|')
    plt.plot(wma_vol.index, wma_vol, label='WMA (30)', color='#2563eb')
    plt.plot(ewma_vol.index, ewma_vol, label='EWMA (λ=0.94)', color='#dc2626')
    plt.title('Volatility Estimation: WMA vs EWMA')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    save_fig('volatility.png', subfolder='models')

# Compatibility wrappers for the pipeline

def compute_wma(returns, window=30):
    """Wrapper that matches the original pipeline name."""
    return wma_volatility(returns, window=window)


def compute_ewma(returns, lam=0.94):
    """Wrapper that matches the original pipeline name."""
    return ewma_volatility(returns, lam=lam)

