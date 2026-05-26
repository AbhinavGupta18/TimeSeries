# src/eda.py
"""Exploratory Data Analysis utilities.

Provides functions for descriptive statistics, stationarity tests, and various EDA plots.
All plots are saved using src.utils.save_fig into the 'plots/eda' subfolder.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp_stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from .utils import save_fig


def descriptive_stats(returns: pd.Series) -> dict:
    """Compute basic descriptive statistics and normality tests.
    Returns a dictionary suitable for JSON serialization.
    """
    desc = {
        "Count": int(returns.count()),
        "Mean": float(returns.mean()),
        "Std Dev": float(returns.std()),
        "Skewness": float(returns.skew()),
        "Kurtosis": float(returns.kurtosis()),
        "Min": float(returns.min()),
        "Max": float(returns.max()),
    }
    jb_stat, jb_p = sp_stats.jarque_bera(returns)
    desc["Jarque-Bera Stat"] = float(jb_stat)
    desc["Jarque-Bera p-value"] = float(jb_p)
    return desc


def stationarity_tests(returns: pd.Series) -> dict:
    """Run Augmented Dickey‑Fuller and KPSS tests.
    Returns a dictionary with test statistics and p‑values.
    """
    adf_stat, adf_p, _, _, _, _ = adfuller(returns, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(returns, regression="c", nlags="auto")
    return {
        "ADF Statistic": float(adf_stat),
        "ADF p-value": float(adf_p),
        "KPSS Statistic": float(kpss_stat),
        "KPSS p-value": float(kpss_p),
    }


def plot_eda(prices: pd.Series, returns: pd.Series):
    """Create the suite of EDA plots and save them.
    All figures are stored in 'plots/eda/'.
    """
    # Price series
    plt.figure()
    plt.plot(prices.index, prices, color="#2563eb")
    plt.title("S&P 500 Closing Prices (2015–2025)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    save_fig("prices.png", subfolder="eda")

    # Return series
    plt.figure()
    plt.plot(returns.index, returns, color="#1e293b", linewidth=0.5)
    plt.title("S&P 500 Daily Log‑Returns (%)")
    plt.xlabel("Date")
    plt.ylabel("Log‑Return (%)")
    save_fig("returns.png", subfolder="eda")

    # Histogram with normal fit
    plt.figure()
    plt.hist(returns, bins=100, density=True, alpha=0.7, color="#6366f1", edgecolor="white")
    x = np.linspace(returns.min(), returns.max(), 500)
    plt.plot(x, sp_stats.norm.pdf(x, returns.mean(), returns.std()), "r-", lw=2, label="Normal fit")
    plt.title("Histogram of Log‑Returns with Normal Fit")
    plt.xlabel("Return (%)")
    plt.ylabel("Density")
    plt.legend()
    save_fig("histogram.png", subfolder="eda")

    # Q‑Q plot
    plt.figure()
    sp_stats.probplot(returns.values, dist="norm", plot=plt)
    plt.title("Normal Q‑Q Plot of Log‑Returns")
    save_fig("qq.png", subfolder="eda")

    # ACF / PACF of returns
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(returns, lags=40, ax=axes[0])
    axes[0].set_title("ACF of Returns")
    plot_pacf(returns, lags=40, ax=axes[1])
    axes[1].set_title("PACF of Returns")
    save_fig("acf_pacf_returns.png", subfolder="eda")

    # ACF / PACF of squared returns
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(returns ** 2, lags=40, ax=axes[0])
    axes[0].set_title("ACF of Squared Returns")
    plot_pacf(returns ** 2, lags=40, ax=axes[1])
    axes[1].set_title("PACF of Squared Returns")
    save_fig("acf_pacf_squared.png", subfolder="eda")

    # Rolling 21‑day volatility
    rolling_vol = returns.rolling(21).std()
    plt.figure()
    plt.plot(rolling_vol.index, rolling_vol, color="#dc2626")
    plt.title("21‑Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility (%)")
    save_fig("rolling_vol.png", subfolder="eda")

    # Save stats JSON for later reference
    stats = {
        "descriptive": descriptive_stats(returns),
        "stationarity": stationarity_tests(returns),
    }
    os.makedirs("plots/eda", exist_ok=True)
    with open(os.path.join("plots", "eda", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
def generate_eda(prices: pd.Series, returns: pd.Series):
    """Convenience wrapper used by the pipeline to run the full EDA suite."""
    plot_eda(prices, returns)
