import numpy as np
from deephedge.env import ShortCallHedgeEnv


def test_environment_contract():
    env = ShortCallHedgeEnv(n_steps=5, seed=123)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(np.array([0.5], dtype=np.float32))
        assert np.isfinite(reward)
        done = terminated or truncated
    assert "terminal_pnl" in info
    assert info["total_cost"] >= 0
