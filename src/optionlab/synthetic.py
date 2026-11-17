"""Deterministic, explicitly synthetic dataset for demonstrations and CI.

Not historical NIFTY data and NOT evidence for any resume performance number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from optionlab.data import prepare_quotes
from optionlab.pricing import black_scholes


def generate_synthetic(seed: int = 2025, n_days: int = 320) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mimic NIFTY scale across low/medium/high volatility environments.

    Price series follows piecewise GBM. Premium is priced using a stylized volatility
    surface plus small noise and market-making spread, whereas the tested BSM
    benchmark intentionally uses lagged 21-day historical volatility.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    regime_sigma = np.where(np.arange(n_days) < n_days / 3, 0.11,
                      np.where(np.arange(n_days) < 2 * n_days / 3, 0.22, 0.33))
    log_returns = (0.06 - 0.5 * regime_sigma**2) / 252 + regime_sigma / np.sqrt(252) * rng.normal(size=n_days)
    price = 18100 * np.exp(np.cumsum(log_returns))
    spot = pd.DataFrame({"date": dates, "close": price})
    # Use exactly the production feature engineering, including lag and 21-day warmup.
    features = spot.copy()
    rets = np.log(features["close"]).diff()
    features["historical_volatility"] = rets.rolling(21, min_periods=21).std().shift(1) * np.sqrt(252)
    snapshots = []
    for i in range(30, n_days - 10, 4):
        s = float(price[i])
        hv = float(features.at[i, "historical_volatility"])
        if not np.isfinite(hv):
            continue
        expiry = dates[i] + pd.Timedelta(days=28)
        for m in (-0.06, -0.03, 0.00, 0.03, 0.06):
            strike = float(round(s * np.exp(-m) / 50) * 50)
            for option_type in ("CE", "PE"):
                # Stylized short-end equity skew + risk premium, separate from benchmark HV.
                surface_vol = np.clip(0.025 + 0.92 * hv - 0.45 * m + 0.22 * m * m, 0.06, 0.85)
                fair = black_scholes(s, strike, 28/365, 0.065, float(surface_vol), option_type, 0.012)
                noisy = fair * (1 + rng.normal(0, 0.012)) + rng.uniform(-0.5, 0.5)
                snapshots.append({"quote_date": dates[i], "expiry": expiry, "spot": s,
                                  "strike": strike, "option_type": option_type,
                                  "market_price": max(0.01, noisy),
                                  "historical_volatility": hv,
                                  "risk_free_rate": 0.065, "dividend_yield": 0.012,
                                  "volume": int(rng.integers(100, 15000))})
    raw = pd.DataFrame(snapshots)
    return prepare_quotes(raw), spot
