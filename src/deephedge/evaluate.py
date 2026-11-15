"""Evaluate PPO, Black-Scholes delta and unhedged strategies across volatility regimes."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from deephedge.market import simulate_gbm
from deephedge.baselines import evaluate_delta_hedge, evaluate_no_hedge, evaluate_policy, summarize

REGIMES = {"low": 0.10, "normal": 0.20, "high": 0.40}


def run_evaluation(model_path="models/ppo_short_call", n_paths=5000, seed=2026, output="results"):
    model = PPO.load(model_path)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_frames = []
    for idx, (regime, vol) in enumerate(REGIMES.items()):
        paths = simulate_gbm(
            n_paths=n_paths, n_steps=30, spot=100, strike=100,
            maturity=30/252, rate=0.02, volatility=vol, seed=seed + idx,
        )
        delta = evaluate_delta_hedge(paths, 100, 30/252, 0.02, vol, 0.001)
        nohedge = evaluate_no_hedge(paths, 100, 30/252, 0.02, vol)
        rl = evaluate_policy(model, paths, 100, 30/252, 0.02, vol, 0.001)
        summary = summarize({"PPO": rl, "Black-Scholes delta": delta, "No hedge": nohedge}, regime)
        summaries.append(summary)
        for name, frame in {"PPO": rl, "Black-Scholes delta": delta, "No hedge": nohedge}.items():
            copy = frame.copy()
            copy["strategy"] = name
            copy["regime"] = regime
            all_frames.append(copy)
    summary_df = pd.concat(summaries, ignore_index=True)
    detail_df = pd.concat(all_frames, ignore_index=True)
    summary_df.to_csv(output_dir / "metrics_by_regime.csv", index=False)
    detail_df.to_csv(output_dir / "path_results.csv", index=False)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.5f}"))
    return summary_df, detail_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_short_call")
    parser.add_argument("--paths", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run_evaluation(args.model, args.paths, args.seed, args.output)


if __name__ == "__main__":
    main()
