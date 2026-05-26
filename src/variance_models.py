# src/variance_models.py
"""Variance (conditional volatility) model utilities.
Provides functions to fit ARCH, GARCH, GARCH-t models, perform a grid
search over (p,q) for the GARCH-t family, and plot the best model.
"""
import json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from .utils import save_fig


def fit_arch(returns: pd.Series, p: int = 1):
    """Fit an ARCH(p) model with constant mean."""
    am = arch_model(returns, mean='Constant', vol='ARCH', p=p)
    return am.fit(disp='off')


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = 'normal'):
    """Fit a GARCH(p,q) model.
    ``dist`` can be ``'normal'`` or ``'t'`` for Student's-t innovations.
    """
    am = arch_model(returns, mean='Constant', vol='Garch', p=p, q=q, dist=dist)
    return am.fit(disp='off')

def fit_garch_t(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit a GARCH(p,q) model with Student's-t errors.

    Wrapper used by the pipeline; forwards to ``fit_garch`` with ``dist='t'``.
    """
    return fit_garch(returns, p=p, q=q, dist='t')



def grid_search_garch_t(returns: pd.Series, max_p: int = 2, max_q: int = 2):
    """Search over GARCH(p,q)-t models and return the best (lowest AIC).
    Returns the best fitted model, the full results DataFrame and the (p,q) tuple.
    """
    results = []
    for p in range(1, max_p + 1):
        for q in range(1, max_q + 1):
            try:
                res = fit_garch(returns, p=p, q=q, dist='t')
                results.append({"p": p, "q": q, "AIC": res.aic, "BIC": res.bic})
            except Exception:
                continue
    df = pd.DataFrame(results)
    os.makedirs('plots/grid_search', exist_ok=True)
    df.to_csv('plots/grid_search/garch_grid.csv', index=False)
    best_row = df.loc[df['AIC'].idxmin()]
    best_model = fit_garch(returns, int(best_row['p']), int(best_row['q']), dist='t')
    return best_model, df, (int(best_row['p']), int(best_row['q']))


def plot_best_garch(returns: pd.Series, best_model, p: int, q: int):
    """Plot conditional volatility of the best GARCH-t model against absolute returns.
    Saved to ``plots/models/best_garch_fit.png``.
    """
    vol = best_model.conditional_volatility
    plt.figure(figsize=(14, 5))
    plt.plot(returns.index, returns.abs(), color='gray', alpha=0.3, label='|Returns|')
    plt.plot(vol.index, vol, color='#dc2626', label=f'GARCH({p},{q})‑t Vol')
    plt.title(f'Best GARCH({p},{q})‑t Conditional Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    save_fig('best_garch_fit.png', subfolder='models')


def save_summary(best_model, p: int, q: int):
    """Save a concise JSON summary of the best GARCH-t model."""
    summary = {
        "model": f"GARCH({p},{q})-t",
        "AIC": float(best_model.aic),
        "BIC": float(best_model.bic),
        "loglikelihood": float(best_model.loglikelihood),
        "nu": float(best_model.params.get('nu', np.nan)),
    }
    os.makedirs('plots/models', exist_ok=True)
    with open('plots/models/best_garch_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
