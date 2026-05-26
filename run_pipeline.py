# run_pipeline.py
"""Main entry point for the TimeSeries modular pipeline.

This script orchestrates the workflow by importing utilities from the
`src/` package. It loads cached data (or downloads if missing), runs the
exploratory data analysis, estimates volatility, fits conditional mean and
variance models, performs grid searches, plots the best ARMA and GARCH
models, and finally runs VaR back‑testing.

Running the script:
    python run_pipeline.py

All plots are saved under the `plots/` directory with appropriate
sub‑folders.
"""

import os
import sys

# Ensure the repository root is in sys.path for relative imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import ensure_dir, save_fig
from src.data_loader import fetch_data
from src.eda import generate_eda
from src.non_parametric import compute_wma, compute_ewma, plot_volatility
from src.mean_models import fit_ar, fit_ma, fit_arma, grid_search_arma, plot_best_arma
from src.variance_models import (
    fit_arch,
    fit_garch,
    fit_garch_t,
    grid_search_garch_t,
    plot_best_garch,
)
from src.var_backtest import var_backtest


def main():
    # 1. Load data (cached if available)
    prices, returns = fetch_data()

    # 2. Exploratory Data Analysis
    generate_eda(prices, returns)

    # 3. Volatility estimation (WMA & EWMA)
    wma_vol = compute_wma(returns, window=30)
    ewma_vol = compute_ewma(returns, lam=0.94)
    plot_volatility(returns, wma_vol, ewma_vol)

    # 4. Conditional Mean Models
    ar_res = fit_ar(returns, order=1)
    ma_res = fit_ma(returns, order=1)
    arma_res = fit_arma(returns, p=1, q=1)
    # Grid search for best ARMA(p,q)
    best_arma_model, arma_grid_df, (best_p, best_q) = grid_search_arma(returns, max_p=4, max_q=4)
    # Plot the best ARMA model (choose AIC as criterion)
    plot_best_arma(returns, best_arma_model, best_p, best_q)

    # 5. Conditional Variance Models
    arch_res = fit_arch(returns, p=1)
    garch_res = fit_garch(returns, p=1, q=1)
    garch_t_res = fit_garch_t(returns, p=1, q=1)
    # Grid search for best GARCH(p,q) under t‑distribution
    best_garch_model, garch_grid_df, (best_p, best_q) = grid_search_garch_t(returns, max_p=2, max_q=2)
    # Plot the best GARCH‑t model using the results from grid search
    plot_best_garch(returns, best_garch_model, best_p, best_q)

    # 6. VaR Back‑testing using both GARCH models
    var_models = {
        "GARCH": garch_res,
        "GARCH-t": garch_t_res,
    }
    var_backtest(returns, **var_models)

    print("\n[DONE] Pipeline completed – all figures saved under ./plots/.")


if __name__ == "__main__":
    main()
