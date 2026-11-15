"""Risk-neutral GBM path simulation with configurable constant volatility regimes."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def simulate_gbm(
    n_paths: int = 2_000,
    n_steps: int = 30,
    spot: float = 100.0,
    strike: float = 100.0,
    maturity: float = 30 / 252,
    rate: float = 0.02,
    volatility: float = 0.20,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate risk-neutral geometric Brownian motion paths, shape (paths, steps+1)."""
    if n_paths < 1 or n_steps < 1:
        raise ValueError("n_paths and n_steps must be positive")
    if spot <= 0 or strike <= 0 or maturity <= 0 or volatility <= 0:
        raise ValueError("spot, strike, maturity and volatility must be positive")
    rng = np.random.default_rng(seed)
    dt = maturity / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = np.concatenate(
        [np.full((n_paths, 1), np.log(spot)), np.log(spot) + log_paths], axis=1
    )
    return np.exp(paths)


def bs_call_price_delta(
    spot: float | np.ndarray,
    strike: float,
    tau: float | np.ndarray,
    rate: float,
    volatility: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Black-Scholes European call price and delta; handles expiry explicitly."""
    s, t, vol = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(tau, dtype=float),
        np.asarray(volatility, dtype=float),
    )
    t_safe = np.maximum(t, 1e-12)
    vol_safe = np.maximum(vol, 1e-12)
    d1 = (np.log(s / strike) + (rate + 0.5 * vol_safe**2) * t_safe) / (
        vol_safe * np.sqrt(t_safe)
    )
    d2 = d1 - vol_safe * np.sqrt(t_safe)
    price = s * norm.cdf(d1) - strike * np.exp(-rate * t_safe) * norm.cdf(d2)
    delta = norm.cdf(d1)
    expired = t <= 0
    price = np.where(expired, np.maximum(s - strike, 0.0), price)
    delta = np.where(expired, (s > strike).astype(float), delta)
    return price, delta
