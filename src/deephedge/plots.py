"""Create publication-friendly plots from evaluation CSVs."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def create_plots(results_dir="results"):
    root = Path(results_dir)
    metrics = pd.read_csv(root / "metrics_by_regime.csv")
    paths = pd.read_csv(root / "path_results.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in metrics.groupby("strategy"):
        ax.plot(group["regime"], group["pnl_std"], marker="o", label=strategy)
    ax.set(title="Hedging P&L volatility by market regime", xlabel="Volatility regime", ylabel="P&L standard deviation")
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(root / "pnl_risk_by_regime.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for strategy, group in paths[paths["regime"] == "normal"].groupby("strategy"):
        ax.hist(group["pnl"], bins=60, alpha=.45, density=True, label=strategy)
    ax.set(title="Terminal hedging P&L distribution — normal volatility", xlabel="Terminal P&L", ylabel="Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "pnl_distribution_normal.png", dpi=180)
    plt.close(fig)
