"""Streamlit dashboard for DeepHedge RL results."""
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DeepHedge RL", page_icon="ðŸ“ˆ", layout="wide")
st.title("DeepHedge RL")
st.caption("Risk-sensitive reinforcement learning for short-call option hedging")
metrics_path = Path("results/metrics_by_regime.csv")
detail_path = Path("results/path_results.csv")
if not metrics_path.exists() or not detail_path.exists():
    st.info("No evaluation results found. Train a model and run the evaluation pipeline first.")
    st.code("python -m deephedge.train --timesteps 150000\npython -m deephedge.evaluate\npython -m deephedge.plots")
else:
    metrics = pd.read_csv(metrics_path)
    detail = pd.read_csv(detail_path)
    regime = st.selectbox("Volatility regime", ["low", "normal", "high"], index=1)
    st.subheader(f"Strategy comparison â€” {regime} volatility")
    selected = metrics[metrics["regime"] == regime].set_index("strategy")
    st.dataframe(selected.round(4), width="stretch")
    st.subheader("Terminal P&L distributions")
    st.bar_chart(selected[["pnl_std", "mean_transaction_cost"]])
    st.subheader("P&L samples")
    st.line_chart(detail[detail["regime"] == regime].groupby("strategy")["pnl"].head(150).reset_index(drop=True))
    st.caption("Synthetic risk-neutral simulations; not investment advice.")
