"""Cross-sectional model diagnostics. Market premiums are NOT fitted in-sample."""
from __future__ import annotations

import numpy as np
import pandas as pd

from optionlab.pricing import black_scholes, crr_binomial, greeks, implied_volatility


def score_options(quotes: pd.DataFrame, tree_steps: int = 201) -> pd.DataFrame:
    """Benchmark 21-day trailing HV BSM vs observed premiums; compare to CRR tree."""
    if quotes.empty:
        raise ValueError("empty quotes")
    rows = []
    for row in quotes.itertuples(index=False):
        a = dict(spot=float(row.spot), strike=float(row.strike), maturity=float(row.maturity),
                 rate=float(row.risk_free_rate), sigma=float(row.historical_volatility),
                 kind=str(row.option_type), dividend=float(row.dividend_yield))
        bs = black_scholes(**a)
        tree = crr_binomial(**a, steps=tree_steps)
        g = greeks(**a)
        # Implied IV is descriptive; it is NEVER substituted into BSM to "test" its own premium.
        try:
            iv = implied_volatility(float(row.market_price), a["spot"], a["strike"],
                                    a["maturity"], a["rate"], a["kind"], a["dividend"])
        except ValueError:
            iv = float("nan")
        rows.append((bs, tree, iv, g.delta, g.gamma, g.vega, g.theta, g.rho))
    cols = ["bs_price", "binomial_price", "market_implied_volatility", "delta", "gamma", "vega", "theta", "rho"]
    df = quotes.copy()
    for name, vals in zip(cols, zip(*rows)):
        df[name] = vals
    df["pricing_error"] = df["market_price"] - df["bs_price"]
    df["absolute_error"] = df["pricing_error"].abs()
    df["squared_error"] = df["pricing_error"] ** 2
    df["tree_bs_difference"] = df["binomial_price"] - df["bs_price"]
    return df


def metrics(df: pd.DataFrame) -> dict:
    return {"n_quotes": int(len(df)),
            "mae": float(df["absolute_error"].mean()),
            "rmse": float(np.sqrt(df["squared_error"].mean())),
            "mean_bias_market_minus_model": float(df["pricing_error"].mean()),
            "median_absolute_error": float(df["absolute_error"].median()),
            "mean_tree_abs_difference": float(df["tree_bs_difference"].abs().mean())}


def grouped_bias(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Quote-weighted error per market regime or moneyness bucket."""
    return (df.groupby(column, observed=True)
              .agg(n_quotes=("pricing_error", "size"), mean_bias=("pricing_error", "mean"),
                   mae=("absolute_error", "mean"), rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))))
              .reset_index())
