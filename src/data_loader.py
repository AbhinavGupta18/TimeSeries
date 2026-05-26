# src/data_loader.py
"""Data loading and caching utilities for the TimeSeries project.

- If cached CSV exists, load from it.
- Otherwise download from Yahoo Finance and cache.
"""
import os
import pandas as pd
import yfinance as yf
import numpy as np

CACHE_PATH = os.path.join('data', 'sp500_data.csv')

def fetch_data():
    """Fetch S&P 500 price data and compute log returns.
    Returns
    -------
    prices: pd.Series
        Closing prices.
    log_ret: pd.Series
        Percentage log returns.
    """
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    if os.path.isfile(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=['Date'], index_col='Date')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        prices = df['Close']
        print('[1/2] Loaded cached S&P 500 data.')
    else:
        print('[1/2] Downloading S&P 500 data ...')
        df = yf.download('^GSPC', start='2015-01-01', end='2025-01-01', progress=False)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.reset_index().to_csv(CACHE_PATH, index=False)
        prices = df['Close']
        print('[1/2] Download complete and cached.')
    # Compute log returns in percentage
    log_ret = np.log(prices / prices.shift(1)).dropna() * 100
    return prices, log_ret
