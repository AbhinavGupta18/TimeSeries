# src/var_backtest.py
"""Value-at-Risk backtesting utilities.

Provides VaR calculation and Kupiec POF likelihood ratio test for given GARCH models.
"""
import json
import os
import numpy as np
import scipy.stats as sp_stats
from src.utils import save_fig

def kupiec_pof_test(violations: int, total_obs: int, alpha: float):
    """Calculate Kupiec POF test statistic and p-value.

    Parameters
    ----------
    violations: int
        Number of observed VaR violations.
    total_obs: int
        Total number of observations.
    alpha: float
        VaR confidence level (e.g., 0.01 for 99%).
    """
    p_observed = violations / total_obs
    p_expected = alpha
    if violations == 0:
        lr_stat = -2 * total_obs * np.log(1 - p_expected)
    elif violations == total_obs:
        lr_stat = -2 * total_obs * np.log(p_expected)
    else:
        null_log_lik = (total_obs - violations) * np.log(1 - p_expected) + violations * np.log(p_expected)
        alt_log_lik = (total_obs - violations) * np.log(1 - p_observed) + violations * np.log(p_observed)
        lr_stat = -2 * (null_log_lik - alt_log_lik)
    p_val = sp_stats.chi2.sf(lr_stat, df=1)
    return lr_stat, p_val

def var_backtest(returns, **model_dict):
    """Run VaR back‑testing for a set of GARCH‑type models.

    Parameters
    ----------
    returns: pd.Series
        Log‑return series.
    **model_dict: dict
        Mapping of model name to fitted arch_model result objects passed as keyword arguments.
    """
    n_obs = len(returns)
    results = {}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.plot(returns.index, returns, color='gray', linewidth=0.3, label='Returns', alpha=0.5)
    for name, res in model_dict.items():
        vol = res.conditional_volatility
        mu = res.params.get('mu', 0)
        results[name] = {}
        for alpha in (0.05, 0.01):
            if getattr(res.model, 'distribution', None) and getattr(res.model.distribution, 'name', '').lower() in ('normal', 'norm'):
                z = sp_stats.norm.ppf(alpha)
            elif 'nu' in res.params:
                nu = res.params['nu']
                z = sp_stats.t.ppf(alpha, df=nu) * np.sqrt((nu - 2) / nu)
            else:
                # fallback to normal if distribution info missing
                z = sp_stats.norm.ppf(alpha)
            var_series = mu + z * vol
            violations = (returns < var_series).sum()
            lr, p = kupiec_pof_test(int(violations), n_obs, alpha)
            results[name][f'{int((1-alpha)*100)}%'] = {
                'violations': int(violations),
                'violation_rate_%': float(violations / n_obs * 100),
                'expected_%': alpha * 100,
                'lr_stat': float(lr),
                'p_value': float(p),
                'pass': bool(p > 0.05),
            }
            if alpha == 0.01:
                ax.plot(var_series.index, var_series, linewidth=1, label=f"{name} 99% VaR (viol={violations})")
    ax.set_title('VaR Back‑test 99% Confidence')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.legend(fontsize=9)
    save_fig('var_backtest.png', subfolder='var')
    out_path = os.path.join('plots', 'var', 'var_backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    return results

