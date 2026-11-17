"""Risk-neutral pricing MC, empirical error bootstrap and scenario hedge VaR."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from optionlab.pricing import black_scholes, black_scholes_array, greeks


@dataclass(frozen=True)
class MCPricingResult:
    price: float
    analytic_price: float
    standard_error: float
    ci95_lower: float
    ci95_upper: float
    n_paths: int
    mc_error: float


def monte_carlo_price(
    spot: float, strike: float, maturity: float, rate: float, sigma: float,
    kind: str = "call", dividend: float = 0.0, n_paths: int = 100_000, seed: int = 42,
) -> MCPricingResult:
    """Exact risk-neutral GBM terminal simulation with antithetic path pairs.

    Standard error uses independent *pair averages*, not correlated paths.
    """
    if n_paths < 100 or n_paths % 2:
        raise ValueError("n_paths must be even and at least 100")
    if min(spot, strike, maturity, sigma) <= 0:
        raise ValueError("spot, strike, maturity, sigma must be positive")
    kind = kind.upper()
    if kind not in ("CALL", "PUT", "CE", "PE"):
        raise ValueError("kind must be CE/PE or call/put")
    z = np.random.default_rng(seed).normal(size=n_paths // 2)
    drift = (rate - dividend - sigma**2 / 2) * maturity
    diffusion = sigma * sqrt(maturity) * z
    up = spot * np.exp(drift + diffusion)
    down = spot * np.exp(drift - diffusion)
    if kind in ("CE", "CALL"):
        payoff_a, payoff_b = np.maximum(up - strike, 0), np.maximum(down - strike, 0)
    else:
        payoff_a, payoff_b = np.maximum(strike - up, 0), np.maximum(strike - down, 0)
    pair_prices = exp(-rate * maturity) * 0.5 * (payoff_a + payoff_b)
    estimate = float(np.mean(pair_prices))
    stderr = float(np.std(pair_prices, ddof=1) / sqrt(len(pair_prices)))
    analytic = black_scholes(spot, strike, maturity, rate, sigma, kind, dividend)
    return MCPricingResult(estimate, analytic, stderr,
                           estimate - 1.96 * stderr, estimate + 1.96 * stderr,
                           n_paths, estimate - analytic)


def bootstrap_pricing_errors(errors: np.ndarray, n_draws: int = 100_000, seed: int = 42) -> dict:
    """Resample the observed market-minus-model residuals (descriptive, not IID proof).

    The 100k draws are NOT 100k independent market observations; no claim of
    predictive confidence or economic VaR is made from these alone.
    """
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) < 2 or n_draws < 100:
        raise ValueError("need >=2 error observations and >=100 draws")
    sample = np.random.default_rng(seed).choice(errors, size=n_draws, replace=True)
    return {"observed_quotes": len(errors), "bootstrap_draws": n_draws,
            "mean_error": float(np.mean(sample)), "std_error_distribution": float(np.std(sample)),
            "p01_error": float(np.quantile(sample, 0.01)),
            "median_error": float(np.median(sample)),
            "p99_error": float(np.quantile(sample, 0.99))}


@dataclass(frozen=True)
class HedgeVaRResult:
    var99_unhedged: float
    var99_hedged: float
    es99_unhedged: float
    es99_hedged: float
    var_reduction_pct: float
    initial_delta: float
    n_paths: int
    horizon_days: int
    transaction_cost_bps: float


def _var_and_es(pnl: np.ndarray) -> tuple[float, float]:
    losses = -np.asarray(pnl)
    threshold = float(np.quantile(losses, 0.99))
    return max(0.0, threshold), max(0.0, float(np.mean(losses[losses >= threshold])))


def simulate_delta_hedge_var(
    spot: float, strike: float, maturity: float, rate: float, sigma: float,
    kind: str = "call", dividend: float = 0.0, horizon_days: int = 1,
    n_paths: int = 100_000, seed: int = 42, realized_vol_multiplier: float = 1.0,
    iv_daily_shock: float = 0.035, spot_iv_correlation: float = -0.4,
    transaction_cost_bps: float = 1.0, return_pnl: bool = False,
) -> HedgeVaRResult | tuple[HedgeVaRResult, np.ndarray, np.ndarray]:
    """99% one-horizon mark-to-market VaR of a SHORT option vs short+delta hedge.

    Risk factors: GBM spot under a *physical* drift approximation, random
    lognormal IV shocks, and correlated spot/IV innovations. The hedge holds
    delta underlying units per short option; costs charge round-trip only when
    explicitly traded (here opening stock hedge only). P&L includes rate carry
    on premium/stock funding and continuous dividend approximation.

    VaR is a scenario-model estimate in INDEX POINTS, not a backtest. A constant
    delta is held over the horizon (no daily rebalance within that horizon).
    """
    if n_paths < 100:
        raise ValueError("n_paths must be >=100")
    if spot <= 0 or strike <= 0 or sigma <= 0 or maturity <= 0:
        raise ValueError("spot, strike, sigma, maturity must be positive")
    if not 0 < horizon_days / 365 < maturity:
        raise ValueError("horizon_days must be positive and less than calendar maturity")
    if realized_vol_multiplier <= 0 or iv_daily_shock < 0 or transaction_cost_bps < 0:
        raise ValueError("invalid volatility/cost inputs")
    if not -1 <= spot_iv_correlation <= 1:
        raise ValueError("spot_iv_correlation must be between -1 and 1")
    h = horizon_days / 365.0
    rng = np.random.default_rng(seed)
    zs, independent = rng.normal(size=(2, n_paths))
    zv = spot_iv_correlation * zs + np.sqrt(1 - spot_iv_correlation**2) * independent
    realized_sigma = sigma * realized_vol_multiplier
    s_new = spot * np.exp((rate - dividend - realized_sigma**2/2) * h + realized_sigma * np.sqrt(h) * zs)
    sigma_new = np.clip(sigma * np.exp(iv_daily_shock * np.sqrt(horizon_days) * zv), 0.01, 3.0)
    t_new = maturity - h
    option_new = black_scholes_array(s_new, strike, t_new, rate, sigma_new, kind, dividend)
    premium = black_scholes(spot, strike, maturity, rate, sigma, kind, dividend)
    delta = greeks(spot, strike, maturity, rate, sigma, kind, dividend).delta
    # Option seller: short an option; delta-neutralise with +delta units of index.
    unhedged = premium * np.exp(rate * h) - option_new
    hedge_carry = delta * spot * (np.exp(dividend * h) - np.exp(rate * h))
    opening_cost = abs(delta * spot) * transaction_cost_bps / 10_000
    hedged = unhedged + delta * (s_new - spot) + hedge_carry - opening_cost
    u_var, u_es = _var_and_es(unhedged)
    h_var, h_es = _var_and_es(hedged)
    reduction = 100 * (u_var - h_var) / u_var if u_var > 0 else float("nan")
    result = HedgeVaRResult(u_var, h_var, u_es, h_es, reduction, delta, n_paths,
                            horizon_days, transaction_cost_bps)
    if return_pnl:
        return result, unhedged, hedged
    return result


def empirical_delta_hedge_backtest(
    scored: "pd.DataFrame", n_draws: int = 100_000, seed: int = 42,
    transaction_cost_bps: float = 1.0,
    max_gap_days: int = 4,
) -> tuple[dict, "pd.DataFrame"]:
    """Close-to-close short-option P&L using actual consecutive contract observations.

    Pairs consecutive quotes of identical (expiry, strike, type); excludes gaps
    beyond max_gap_days. The initial delta uses ONLY the earlier quote's lagged
    historical volatility. VaR is the 99th percentile of an IID paired bootstrap
    (report empirical effective sample size and limitations). No ex-post fitting.
    """
    import pandas as pd

    required = {"quote_date", "expiry", "strike", "option_type", "market_price", "spot",
                "delta", "risk_free_rate", "dividend_yield"}
    if missing := required - set(scored.columns):
        raise ValueError(f"missing fields for backtest: {sorted(missing)}")
    df = scored.sort_values(["expiry", "strike", "option_type", "quote_date"]).copy()
    groups = df.groupby(["expiry", "strike", "option_type"], sort=False)
    for col in ("quote_date", "market_price", "spot"):
        df[f"next_{col}"] = groups[col].shift(-1)
    df["gap_days"] = (df["next_quote_date"] - df["quote_date"]).dt.days
    paired = df.loc[df["gap_days"].between(1, max_gap_days) &
                    (df["next_quote_date"] <= df["expiry"])].copy()
    if paired.empty:
        return {"status": "unavailable", "reason": "No repeated same-contract quotes within 1-4 calendar days",
                "n_observed_pairs": 0}, paired
    h = paired["gap_days"].to_numpy(float) / 365
    r, q = paired["risk_free_rate"].to_numpy(float), paired["dividend_yield"].to_numpy(float)
    d = paired["delta"].to_numpy(float)
    s = paired["spot"].to_numpy(float)
    s_next = paired["next_spot"].to_numpy(float)
    p = paired["market_price"].to_numpy(float)
    p_next = paired["next_market_price"].to_numpy(float)
    # Premium is received upfront and grows until next close.
    paired["unhedged_pnl"] = p * np.exp(r * h) - p_next
    carry = d * s * (np.exp(q * h) - np.exp(r * h))
    open_hedge_cost = np.abs(d * s) * transaction_cost_bps / 10_000
    paired["hedged_pnl"] = (paired["unhedged_pnl"].to_numpy(float)
                            + d * (s_next - s) + carry - open_hedge_cost)
    pnl_u = paired["unhedged_pnl"].to_numpy(float)
    pnl_h = paired["hedged_pnl"].to_numpy(float)
    if len(paired) < 30:
        return {"status": "insufficient_sample", "n_observed_pairs": len(paired),
                "reason": "At least 30 paired observations needed for descriptive bootstrapped VaR"}, paired
    idx = np.random.default_rng(seed).integers(0, len(paired), size=n_draws)
    u_var, u_es = _var_and_es(pnl_u[idx])
    h_var, h_es = _var_and_es(pnl_h[idx])
    return {
        "status": "descriptive_backtest", "n_observed_pairs": int(len(paired)),
        "bootstrap_draws": int(n_draws), "max_calendar_gap_days": int(max_gap_days),
        "unhedged_var99": u_var, "hedged_var99": h_var,
        "unhedged_es99": u_es, "hedged_es99": h_es,
        "var_reduction_pct": float(100 * (u_var - h_var)/u_var) if u_var > 0 else None,
        "note": "IID bootstrap of correlated overlapping contract-date pairs; not independent observations or an executable trading result",
    }, paired
