"""Benchmark hedging policies and common evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from deephedge.market import bs_call_price_delta


def evaluate_delta_hedge(
    paths: np.ndarray, strike: float, maturity: float, rate: float,
    volatility: float, transaction_cost: float,
) -> pd.DataFrame:
    return _evaluate_rule(paths, strike, maturity, rate, volatility, transaction_cost, "delta")


def evaluate_no_hedge(
    paths: np.ndarray, strike: float, maturity: float, rate: float,
    volatility: float, transaction_cost: float = 0.0,
) -> pd.DataFrame:
    return _evaluate_rule(paths, strike, maturity, rate, volatility, transaction_cost, "none")


def evaluate_policy(
    model, paths: np.ndarray, strike: float, maturity: float, rate: float,
    volatility: float, transaction_cost: float,
) -> pd.DataFrame:
    """Evaluate PPO using batched predictions across all paths."""
    n_paths, n_columns = paths.shape
    n_steps = n_columns - 1

    hedge = np.zeros(n_paths, dtype=float)
    total_cost = np.zeros(n_paths, dtype=float)
    hedge_pnl = np.zeros(n_paths, dtype=float)

    premium, _ = bs_call_price_delta(
        paths[:, 0], strike, maturity, rate, volatility
    )

    for t in range(n_steps):
        tau = maturity * (1 - t / n_steps)

        _, delta = bs_call_price_delta(
            paths[:, t], strike, tau, rate, volatility
        )

        obs = np.column_stack((
            paths[:, t] / strike,
            np.full(n_paths, tau / maturity),
            np.full(n_paths, volatility),
            delta,
            hedge,
        )).astype(np.float32)

        actions, _ = model.predict(obs, deterministic=True)
        raw_actions = np.clip(
            np.asarray(actions).reshape(-1), -5.0, 5.0
        )
        target = 1.0 / (1.0 + np.exp(-raw_actions))

        total_cost += transaction_cost * paths[:, t] * np.abs(target - hedge)
        hedge = target
        hedge_pnl += hedge * (paths[:, t + 1] - paths[:, t])

    payoff = np.maximum(paths[:, -1] - strike, 0.0)
    pnl = premium + hedge_pnl - total_cost - payoff

    return pd.DataFrame({
        "pnl": pnl,
        "transaction_cost": total_cost,
        "payoff": payoff,
    })

def _evaluate_rule(paths, strike, maturity, rate, volatility, transaction_cost, rule):
    rows = []
    n_steps = paths.shape[1] - 1
    for path in paths:
        hedge = 0.0
        total_cost = 0.0
        hedge_pnl = 0.0
        premium = float(bs_call_price_delta(path[0], strike, maturity, rate, volatility)[0])
        for t in range(n_steps):
            if rule == "delta":
                tau = maturity * (1 - t / n_steps)
                _, delta = bs_call_price_delta(path[t], strike, tau, rate, volatility)
                target = float(delta)
            else:
                target = 0.0
            total_cost += transaction_cost * path[t] * abs(target - hedge)
            hedge = target
            hedge_pnl += hedge * (path[t + 1] - path[t])
        payoff = max(path[-1] - strike, 0.0)
        pnl = premium + hedge_pnl - total_cost - payoff
        rows.append({"pnl": pnl, "transaction_cost": total_cost, "payoff": payoff})
    return pd.DataFrame(rows)


def summarize(results: dict[str, pd.DataFrame], regime: str) -> pd.DataFrame:
    rows = []
    for strategy, frame in results.items():
        pnl = frame["pnl"].to_numpy()
        rows.append({
            "regime": regime,
            "strategy": strategy,
            "mean_pnl": pnl.mean(),
            "pnl_std": pnl.std(ddof=1),
            "pnl_variance": pnl.var(ddof=1),
            "mean_abs_pnl": np.abs(pnl).mean(),
            "mean_transaction_cost": frame["transaction_cost"].mean(),
            "p05_pnl": np.quantile(pnl, 0.05),
            "p95_pnl": np.quantile(pnl, 0.95),
        })
    return pd.DataFrame(rows)
