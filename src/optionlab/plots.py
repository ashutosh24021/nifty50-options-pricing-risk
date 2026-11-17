"""Readable publication-style chart generation, no optional plotting dependencies."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def save_charts(scored: pd.DataFrame, regime: pd.DataFrame, moneyness: pd.DataFrame,
                u_pnl: np.ndarray, h_pnl: np.ndarray, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    palette = {"Low": "#0891b2", "Medium": "#6366f1", "High": "#db2777"}

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    order = [x for x in ["Low", "Medium", "High"] if x in regime["vol_regime"].tolist()]
    view = regime.set_index("vol_regime").loc[order]
    ax.bar(order, view["mean_bias"], color=[palette[k] for k in order], width=0.55)
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set(ylabel="Market premium − BSM price (index pts)", title="BSM pricing bias by lagged volatility regime")
    _save(fig, out / "regime_bias.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    ax.scatter(scored["bs_price"], scored["market_price"], s=11, alpha=0.32, color="#4f46e5")
    high = max(float(scored["market_price"].quantile(0.99)), float(scored["bs_price"].quantile(0.99)))
    ax.plot([0, high], [0, high], color="#e11d48", lw=1.5, ls="--", label="Perfect agreement")
    ax.set(xlabel="BSM (index pts)", ylabel="Quoted premium (index pts)",
           title="Observed vs modeled European option premiums", xlim=(0, high * 1.05), ylim=(0, high * 1.05))
    ax.legend(frameon=False)
    _save(fig, out / "market_vs_model.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    order_m = [x for x in ["ITM", "ATM", "OTM"] if x in moneyness["moneyness_bucket"].tolist()]
    view = moneyness.set_index("moneyness_bucket").loc[order_m]
    ax.bar(order_m, view["mae"], color=["#2563eb", "#7c3aed", "#0d9488"][:len(order_m)])
    ax.set(ylabel="Mean absolute error (index pts)", title="Pricing accuracy by moneyness")
    _save(fig, out / "moneyness_mae.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    bound = np.quantile(np.concatenate([u_pnl, h_pnl]), [0.005, 0.995])
    ax.hist(u_pnl, bins=110, range=tuple(bound), density=True, alpha=0.48,
            color="#f97316", label="Unhedged short option")
    ax.hist(h_pnl, bins=110, range=tuple(bound), density=True, alpha=0.50,
            color="#0284c7", label="Initial delta hedge")
    ax.axvline(np.quantile(u_pnl, 0.01), color="#ea580c", ls="--", lw=1.5)
    ax.axvline(np.quantile(h_pnl, 0.01), color="#0369a1", ls="--", lw=1.5)
    ax.set(xlabel="One-horizon P&L (index pts)", ylabel="Probability density",
           title="Simulated P&L and 1st-percentile downside cutoffs")
    ax.legend(frameon=False)
    _save(fig, out / "hedging_var.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for key, subset in scored.groupby("vol_regime"):
        ax.scatter(subset["log_moneyness"], subset["pricing_error"], label=key,
                   color=palette.get(key, "#334155"), alpha=0.22, s=13)
    ax.axhline(0, lw=1, color="#475569")
    ax.set(xlabel="ln(spot / strike)", ylabel="Market − BSM premium (index pts)",
           title="Pricing residuals across strikes and regimes")
    ax.legend(frameon=False)
    _save(fig, out / "pricing_error_scatter.png")
