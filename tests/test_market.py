import numpy as np
from deephedge.market import simulate_gbm, bs_call_price_delta


def test_simulation_shape_and_reproducibility():
    a = simulate_gbm(n_paths=20, n_steps=10, seed=7)
    b = simulate_gbm(n_paths=20, n_steps=10, seed=7)
    assert a.shape == (20, 11)
    assert np.allclose(a, b)
    assert np.all(a > 0)


def test_call_delta_bounds_and_expiry():
    price, delta = bs_call_price_delta(100, 100, 0.5, 0.02, 0.2)
    assert float(price) > 0
    assert 0 <= float(delta) <= 1
    _, expired_delta = bs_call_price_delta(110, 100, 0, 0.02, 0.2)
    assert float(expired_delta) == 1.0
