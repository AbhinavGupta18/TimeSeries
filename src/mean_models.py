# src/mean_models.py
"""Mean model utilities: AR, MA, ARMA fitting and ARMA grid search.
The grid search returns the best (lowest AIC) model, which is plotted and saved.
"""
import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from .utils import save_fig


def fit_ar(returns: pd.Series, order: int = 1):
    """Fit an AR(p) model (p default=1) and return the results object."""
    model = ARIMA(returns, order=(order, 0, 0))
    return model.fit()


def fit_ma(returns: pd.Series, order: int = 1):
    """Fit an MA(q) model (q default=1)."""
    model = ARIMA(returns, order=(0, 0, order))
    return model.fit()


def fit_arma(returns: pd.Series, p: int, q: int):
    """Fit an ARMA(p,q) model."""
    model = ARIMA(returns, order=(p, 0, q))
    return model.fit()


def grid_search_arma(returns: pd.Series, max_p: int = 4, max_q: int = 4):
    """Exhaustively search ARMA(p,q) over 0..max_p and 0..max_q.
    Returns the best model (by AIC) and the full results DataFrame.
    """
    results = []
    aic_matrix = np.full((max_p + 1, max_q + 1), np.nan)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                res = fit_arma(returns, p, q)
                aic_matrix[p, q] = res.aic
                results.append({"p": p, "q": q, "AIC": res.aic, "BIC": res.bic})
            except Exception:
                continue
    df = pd.DataFrame(results)
    # Save heatmap data for later visualisation
    os.makedirs('plots/grid_search', exist_ok=True)
    df.to_csv('plots/grid_search/arma_grid.csv', index=False)
    # Identify best by AIC
    best_row = df.loc[df['AIC'].idxmin()]
    best_model = fit_arma(returns, int(best_row['p']), int(best_row['q']))
    return best_model, df, (int(best_row['p']), int(best_row['q']))


def plot_best_arma(returns: pd.Series, best_model, p: int, q: int):
    """Plot the fitted ARMA series against actual returns and save.
    Saved to ``plots/models/best_arma_fit.png``.
    """
    plt.figure(figsize=(14, 5))
    plt.plot(returns.index, returns, color='gray', alpha=0.4, label='Actual Returns')
    plt.plot(best_model.fittedvalues.index, best_model.fittedvalues, color='#2563eb', label=f'ARMA({p},{q}) Fit')
    plt.title(f'Best ARMA({p},{q}) Fit vs Returns')
    plt.xlabel('Date')
    plt.ylabel('Return (%)')
    plt.legend()
    save_fig('best_arma_fit.png', subfolder='models')


def save_summary(best_model, p: int, q: int):
    """Save a concise summary of the best ARMA model to JSON for reporting."""
    summary = {
        "model": f"ARMA({p},{q})",
        "AIC": float(best_model.aic),
        "BIC": float(best_model.bic),
        "loglikelihood": float(best_model.llf)
    }
    os.makedirs('plots/models', exist_ok=True)
    with open('plots/models/best_arma_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
