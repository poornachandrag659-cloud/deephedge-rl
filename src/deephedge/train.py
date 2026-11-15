"""Train a PPO hedge policy."""
from __future__ import annotations

import argparse
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from deephedge.env import ShortCallHedgeEnv


def train(total_timesteps=150_000, volatility=0.20, seed=42, output="models/ppo_short_call"):
    def make_env():
        return Monitor(ShortCallHedgeEnv(volatility=volatility, seed=seed))
    env = DummyVecEnv([make_env])
    model = PPO(
        "MlpPolicy", env, verbose=1, seed=seed,
        learning_rate=3e-4, n_steps=1024, batch_size=64,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, policy_kwargs={"net_arch": [128, 128]},
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    env.close()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--volatility", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/ppo_short_call")
    args = parser.parse_args()
    train(args.timesteps, args.volatility, args.seed, args.output)


if __name__ == "__main__":
    main()
