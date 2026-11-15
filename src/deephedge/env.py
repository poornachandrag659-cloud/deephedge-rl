"""Gymnasium environment for hedging one short European call."""
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from deephedge.market import bs_call_price_delta, simulate_gbm


class ShortCallHedgeEnv(gym.Env):
    """Agent selects a target stock hedge ratio in [0, 1] at each hedge date.

    Episode reward is incremental marked-to-market hedge P&L less transaction
    costs; at maturity it includes the short-call liability and a terminal
    quadratic penalty. Evaluation reports the full terminal P&L distribution.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_steps: int = 30,
        spot: float = 100.0,
        strike: float = 100.0,
        maturity: float = 30 / 252,
        rate: float = 0.02,
        volatility: float = 0.20,
        transaction_cost: float = 0.001,
        risk_aversion: float = 100.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.spot0 = spot
        self.strike = strike
        self.maturity = maturity
        self.rate = rate
        self.volatility = volatility
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        self._seed = seed
        self.action_space = spaces.Box(
            low=np.array([-5.0], dtype=np.float32),
            high=np.array([5.0], dtype=np.float32),
            dtype=np.float32,
        )
                # spot / strike, tau / maturity, vol, BS delta, previous hedge
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5.0, 1.0, 3.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.path = None
        self.t = 0
        self.hedge = 0.0
        self.cash_premium = 0.0
        self.total_cost = 0.0
        self.hedge_pnl = 0.0
        self.last_spot = spot
        self.rng = np.random.default_rng(seed)

    def _obs(self):
        s = float(self.path[self.t])
        tau = max(self.maturity * (1 - self.t / self.n_steps), 0.0)
        _, delta = bs_call_price_delta(s, self.strike, tau, self.rate, self.volatility)
        return np.array(
            [s / self.strike, tau / self.maturity, self.volatility, float(delta), self.hedge],
            dtype=np.float32,
        )




    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        path_seed = int(
            self.np_random.integers(0, 2**32 - 1)
        )

        self.path = simulate_gbm(
            n_paths=1,
            n_steps=self.n_steps,
            spot=self.spot0,
            strike=self.strike,
            maturity=self.maturity,
            rate=self.rate,
            volatility=self.volatility,
            seed=path_seed,
        )[0]

        self.t = 0
        self.hedge = 0.0

        self.cash_premium = float(
            bs_call_price_delta(
                self.spot0,
                self.strike,
                self.maturity,
                self.rate,
                self.volatility,
            )[0]
        )

        self.total_cost = 0.0
        self.hedge_pnl = 0.0
        self.last_spot = float(self.path[0])

        return self._obs(), {}

    def step(self, action):
        raw_action = float(
            np.clip(np.asarray(action).reshape(-1)[0], -5.0, 5.0)
        )
        target = float(1.0 / (1.0 + np.exp(-raw_action)))
        spot_now = float(self.path[self.t])
        trade = target - self.hedge
        cost = self.transaction_cost * spot_now * abs(trade)
        self.total_cost += cost
        self.hedge = target

        next_spot = float(self.path[self.t + 1])
        step_pnl = self.hedge * (next_spot - spot_now)
        self.hedge_pnl += step_pnl
        self.t += 1

        terminated = self.t >= self.n_steps
        reward = 0.0

        info = {
            "transaction_cost": cost,
            "step_pnl": step_pnl,
        }

        if terminated:
            payoff = max(next_spot - self.strike, 0.0)
            terminal_pnl = (
                self.cash_premium
                + self.hedge_pnl
                - self.total_cost
                - payoff
            )

            reward += -self.risk_aversion * (
                terminal_pnl / self.spot0
            ) ** 2

            info.update(
                terminal_pnl=terminal_pnl,
                option_payoff=payoff,
                premium=self.cash_premium,
                total_cost=self.total_cost,
                hedge_pnl=self.hedge_pnl,
            )

        return self._obs(), float(reward), terminated, False, info
