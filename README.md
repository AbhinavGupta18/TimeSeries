# TimeSeries — S&P 500 Quantitative Risk Pipeline

A modular Python pipeline for fitting time-series models to S&P 500 daily returns and back-testing Value-at-Risk (VaR) forecasts.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
TimeSeries/
├── run_pipeline.py          # Entry point — orchestrates all steps
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Download and cache S&P 500 prices
│   ├── eda.py               # Exploratory data analysis and diagnostics
│   ├── mean_models.py       # ARMA fitting and grid search
│   ├── variance_models.py   # GARCH and GARCH-t fitting and grid search
│   ├── var_backtest.py      # VaR computation and back-testing
│   └── utils.py             # Plotting helpers, warning filters
│
├── data/
│   └── sp500_data.csv       # Cached price series (auto-generated)
│
└── plots/
    ├── eda/                 # Descriptive statistics, Q-Q, ACF/PACF
    ├── grid_search/         # AIC/BIC tables for ARMA and GARCH grids
    ├── models/              # Fitted volatility and return plots
    └── var/                 # VaR back-test results and exception plots
```

---

## Running the Pipeline

```bash
python run_pipeline.py
```

The script executes the following steps in order:

1. Load (or download and cache) S&P 500 closing prices.
2. Compute log-returns and run exploratory analysis.
3. Fit ARMA mean models via grid search.
4. Fit GARCH and GARCH-t variance models via grid search.
5. Back-test VaR at the 1% and 5% confidence levels.
6. Save all figures and JSON summaries under `plots/`.

---

## Mathematical Background

### Log-Returns

Given a price series $\{P_t\}$, the continuously compounded return is

$$r_t = \ln P_t - \ln P_{t-1}$$

### ARMA(p, q) — Mean Model

The conditional mean follows an AutoRegressive Moving-Average process:

$$r_t = \mu + \sum_{i=1}^{p} \phi_i\, r_{t-i} + \varepsilon_t + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j}, \qquad \varepsilon_t \sim \mathcal{N}(0,\, \sigma_t^2)$$

Model order $(p, q)$ is selected by minimising the Akaike Information Criterion:

$$\mathrm{AIC} = 2k - 2\ln\hat{L}$$

where $k$ is the number of estimated parameters and $\hat{L}$ is the maximised likelihood.

### GARCH(p, q) — Variance Model

The conditional variance $\sigma_t^2$ follows a Generalised AutoRegressive Conditional Heteroskedasticity model:

$$\sigma_t^2 = \omega + \sum_{i=1}^{p} \alpha_i\, \varepsilon_{t-i}^2 + \sum_{j=1}^{q} \beta_j\, \sigma_{t-j}^2$$

with the stationarity constraint $\sum_{i}\alpha_i + \sum_{j}\beta_j < 1$.

### GARCH-t — Heavy-Tailed Innovation

Replace the Gaussian innovation with a standardised Student-$t$ distribution:

$$\varepsilon_t = \sigma_t\, z_t, \qquad z_t \sim t_\nu(0,\,1)$$

The degree-of-freedom parameter $\nu > 2$ is estimated jointly with $(\omega, \alpha, \beta)$ by maximum likelihood. Heavier tails ($\nu$ small) better capture the leptokurtosis observed in financial returns.

### Value-at-Risk (VaR)

The one-step-ahead VaR at confidence level $\alpha$ is the left quantile of the conditional return distribution:

$$\mathrm{VaR}_{t+1}(\alpha) = \mu_{t+1} + \sigma_{t+1}\, F^{-1}(\alpha)$$

where $F^{-1}(\alpha)$ is the quantile function of the assumed innovation distribution ($\Phi^{-1}$ for Gaussian, $t_\nu^{-1}$ for Student-$t$).

### Kupiec Back-Test

The proportion-of-failures (POF) test checks whether the observed exception rate matches $\alpha$. Let $T$ be the number of observations and $n$ the number of exceptions ($r_t < -\mathrm{VaR}_t$). The likelihood-ratio statistic is

$$\mathrm{LR}_\text{POF} = -2\ln\!\left[\frac{\alpha^n(1-\alpha)^{T-n}}{(\hat{p})^n(1-\hat{p})^{T-n}}\right] \xrightarrow{d} \chi^2_1$$

where $\hat{p} = n/T$. Rejection of the null ($\mathrm{LR}_\text{POF} > \chi^2_{1,\,1-\alpha_\text{test}}$) indicates a mis-specified model.

---

## Version Control

```bash
git add -A
git commit -m "Your message"
git push
```

---

## License

This project is provided under the MIT License.
