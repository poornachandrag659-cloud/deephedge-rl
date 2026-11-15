# DeepHedge RL
### Reinforcement Learning for Risk-Sensitive Option Hedging

DeepHedge RL is a reproducible research project that trains a Proximal Policy Optimization (PPO) agent to dynamically hedge a short European call option. It benchmarks the learned policy against Black–Scholes delta hedging and an unhedged portfolio under low-, normal-, and high-volatility market regimes.

> **Research prototype only.** All results are generated from synthetic risk-neutral GBM paths. This is not a production trading system or financial advice.

## Research question

Can a PPO policy reduce terminal hedging risk while accounting for proportional transaction costs, compared with textbook delta hedging?

## Features

- Risk-neutral geometric Brownian motion market simulator.
- European call pricing and delta from Black–Scholes.
- Gymnasium environment with continuous hedge-ratio actions.
- PPO training using Stable-Baselines3.
- Proportional transaction costs.
- Terminal P&L and risk-sensitive objective.
- Out-of-sample comparison against delta hedge and no hedge.
- Low / normal / high volatility experiments.
- CSV metrics, P&L distributions, plots and Streamlit dashboard.
- Unit tests and reproducible random seeds.

## Methodology

### Market model

Under the risk-neutral measure:

\[
dS_t = rS_tdt + \sigma S_tdW_t
\]

The simulator uses exact discrete-time GBM transitions. The initial benchmark configuration is:

| Parameter | Value |
|---|---:|
| Initial spot | 100 |
| Strike | 100 |
| Maturity | 30 trading days |
| Risk-free rate | 2% |
| Low volatility | 10% |
| Normal volatility | 20% |
| High volatility | 40% |
| Proportional transaction cost | 10 bps |

### Portfolio

The portfolio begins short one European call. Initial premium is received at the Black–Scholes price. The agent trades the underlying stock to hedge the short option liability.

\[
\mathrm{PnL}_T=C_0+\sum_tq_t(S_{t+1}-S_t)
-\sum_tcS_t|q_t-q_{t-1}|-(S_T-K)^+
\]

### Agent

PPO observes normalized spot, moneyness, time to maturity, volatility, Black–Scholes delta, and previous hedge ratio. It selects a continuous target hedge ratio between zero and one.

### Reward design and caveat

The environment rewards incremental hedge P&L net of transaction costs and adds a terminal quadratic penalty on terminal P&L. This is a practical risk-sensitive training proxy, not an exact implementation of the variance objective in the Deep Hedging paper. We report the out-of-sample variance, standard deviation, mean P&L, mean absolute P&L, tail quantiles and transaction costs separately.

For a stronger research extension, train against a batch-level variance or convex risk measure (e.g. entropic risk / CVaR), and include stochastic volatility, jumps, liquidity constraints and market impact.

## Quick start

### 1. Create environment

Python 3.10 or newer is recommended.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 2. Run tests

```bash
pytest -q
```

### 3. Train PPO

```bash
python -m deephedge.train --timesteps 150000 --volatility 0.20
```

The trained model is saved under `models/`.

### 4. Evaluate across regimes

```bash
python -m deephedge.evaluate --paths 5000
```

Outputs:
- `results/metrics_by_regime.csv`
- `results/path_results.csv`

### 5. Generate plots

```bash
python -c "from deephedge.plots import create_plots; create_plots()"
```

### 6. Launch dashboard

```bash
streamlit run app.py
```

## Metrics

The final evaluation uses 5,000 simulated paths per volatility regime, for 15,000 path evaluations across three regimes.

### Normal-volatility results (20%)

| Strategy | Mean P&L | P&L Std. Dev. | Mean Absolute P&L | Mean Transaction Cost |
|---|---:|---:|---:|---:|
| PPO | -0.137455 | 1.160586 | 0.909756 | 0.250822 |
| Black-Scholes delta | -0.102200 | 0.441073 | 0.338432 | 0.222972 |
| No hedge | -0.036622 | 4.313260 | 3.362974 | 0.000000 |

### Results across all volatility regimes

| Regime | Strategy | Mean P&L | P&L Std. Dev. |
|---|---|---:|---:|
| Low (10%) | PPO | -0.153061 | 0.631785 |
| Low (10%) | Black-Scholes delta | -0.104050 | 0.243923 |
| Low (10%) | No hedge | 0.009334 | 2.101992 |
| Normal (20%) | PPO | -0.137455 | 1.160586 |
| Normal (20%) | Black-Scholes delta | -0.102200 | 0.441073 |
| Normal (20%) | No hedge | -0.036622 | 4.313260 |
| High (40%) | PPO | -0.186825 | 2.391594 |
| High (40%) | Black-Scholes delta | -0.124455 | 0.891972 |
| High (40%) | No hedge | 0.293606 | 8.583248 |

### Interpretation

- PPO substantially reduces P&L volatility compared with the unhedged portfolio in all three regimes.
- Black-Scholes delta achieves lower P&L standard deviation and mean absolute P&L than PPO in all three regimes.
- PPO incurs slightly higher average transaction costs than delta hedging in all three regimes.
- These experiments do not demonstrate PPO outperforming the classical delta-hedging benchmark.

PPO was trained at 20% volatility. The low- and high-volatility experiments evaluate transfer of that learned policy; they are not separate training runs.

Full metrics and path-level data are available in `results_v4/metrics_by_regime.csv` and `results_v4/path_results.csv`.

## Project structure

```text
deephedge-rl/
├── app.py
├── pyproject.toml
├── README.md
├── LICENSE
├── src/deephedge/
│   ├── market.py
│   ├── env.py
│   ├── baselines.py
│   ├── train.py
│   ├── evaluate.py
│   └── plots.py
├── tests/
│   ├── test_market.py
│   └── test_env.py
├── models/
└── results/
```

## Limitations

- GBM assumes constant volatility within each experiment.
- PPO is trained at the normal-volatility setting by default; regime transfer should be evaluated critically.
- The simple reward is a proxy for terminal P&L risk, not a batch variance optimizer.
- No discrete dividends, borrow costs, bid-ask spread asymmetry, market impact, margin or liquidity constraints.
- Synthetic simulation performance does not establish live-market profitability.

## Research references

1. Bühler, H., Gonon, L., Teichmann, J. & Wood, B. (2019). [Deep Hedging](https://arxiv.org/abs/1802.03042). *Quantitative Finance*.
2. Kolm, P. N. & Ritter, G. (2019). [Dynamic Replication and Hedging: A Reinforcement Learning Approach](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3281235). *The Journal of Financial Data Science*, 1(1), 159–171.
3. [Deep Hedging reference implementation](https://github.com/hansbuehler/deephedging).
4. [AI4Finance FinRL](https://github.com/AI4Finance-Foundation/FinRL).

## License

MIT for this original project code. The referenced Deep Hedging and FinRL repositories have their own licenses; this project does not copy their source code.
