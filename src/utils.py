# src/utils.py
"""Utility module for the TimeSeries project.
Provides global configuration for warnings, plot styling, and helper functions
for saving figures into appropriate subdirectories.
"""
import os
import warnings
import matplotlib.pyplot as plt

# Suppress specific statsmodels warnings while keeping other warnings visible
try:
    from statsmodels.tools.sm_exceptions import ValueWarning, InterpolationWarning
    warnings.filterwarnings("ignore", category=ValueWarning)
    warnings.filterwarnings("ignore", category=InterpolationWarning)
except ImportError:
    pass

# Suppress specific glyph warnings from matplotlib
warnings.filterwarnings("ignore", message="Glyph 8209*", category=UserWarning)
# Suppress statsmodels convergence and parameter warnings
try:
    from statsmodels.base.model import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", message=".*Non-stationary starting autoregressive parameters.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Non-invertible starting MA parameters.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Check mle_retvals.*", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.figsize': (12, 5),
})

def ensure_dir(path: str):
    """Create a directory if it does not exist (including parents)."""
    os.makedirs(path, exist_ok=True)

def save_fig(fig_name: str, subfolder: str = ""):
    """Save the current matplotlib figure.

    Parameters
    ----------
    fig_name: str
        File name (including extension, e.g., 'my_plot.png').
    subfolder: str, optional
        Subdirectory under the global 'plots' folder. If empty, saved directly
        under 'plots/'.
    """
    base_dir = os.path.join('plots', subfolder)
    ensure_dir(base_dir)
    full_path = os.path.join(base_dir, fig_name)
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight')
    plt.close()
