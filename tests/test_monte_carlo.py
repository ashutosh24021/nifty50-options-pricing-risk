import numpy as np
import pytest

from optionlab.monte_carlo import (
    bootstrap_pricing_errors, monte_carlo_price, simulate_delta_hedge_var,
)


def test_mc_prices_and_confidence_intervals():
    result = monte_carlo_price(100, 100, 1, .05, .2, "CE", n_paths=100_000, seed=123)
    assert result.n_paths == 100_000
    assert abs(result.mc_error) < 4 * result.standard_error
    assert result.ci95_lower < result.price < result.ci95_upper
    assert result.standard_error > 0


def test_empirical_bootstrap_is_reproducible():
    errors = np.array([-2, 1, 0, 3, 6, 0, -1], dtype=float)
    a = bootstrap_pricing_errors(errors, n_draws=10000, seed=5)
    b = bootstrap_pricing_errors(errors, n_draws=10000, seed=5)
    assert a == b
    assert a["observed_quotes"] == 7
    assert a["p01_error"] <= a["p99_error"]


def test_delta_hedge_var_reproducible_and_finite():
    a = simulate_delta_hedge_var(23000, 23000, 28/365, .06, .20, seed=12, n_paths=4000)
    b = simulate_delta_hedge_var(23000, 23000, 28/365, .06, .20, seed=12, n_paths=4000)
    assert a == b
    assert a.var99_unhedged > 0
    assert a.var99_hedged > 0
    assert a.var_reduction_pct > 0
    assert 0 < a.initial_delta < 1


def test_hedge_invalid_horizon():
    with pytest.raises(ValueError, match="horizon"):
        simulate_delta_hedge_var(100, 100, 1/365, .05, .20, horizon_days=1)


def test_paired_market_hedge_backtest_uses_repeated_contracts():
    import pandas as pd
    from optionlab.monte_carlo import empirical_delta_hedge_backtest
    from optionlab.pricing import black_scholes, greeks

    dates = pd.bdate_range("2025-01-06", periods=45)
    rng = np.random.default_rng(11)
    spot_series = 10000 * np.exp(np.cumsum(rng.normal(0, .012, size=45)))
    rows = []
    expiry = pd.Timestamp("2025-04-24")
    for d, s in zip(dates, spot_series):
        t = (expiry - d).days / 365
        sigma = .18
        p = black_scholes(float(s), 10000, t, .05, sigma, "CE", .01)
        g = greeks(float(s), 10000, t, .05, sigma, "CE", .01)
        rows.append({"quote_date": d, "expiry": expiry, "strike": 10000,
                     "option_type": "CE", "spot": s, "delta": g.delta,
                     "market_price": p * (1 + rng.normal(0, .003)),
                     "risk_free_rate": .05, "dividend_yield": .01})
    result, pairs = empirical_delta_hedge_backtest(pd.DataFrame(rows), seed=12, n_draws=10000)
    assert result["status"] == "descriptive_backtest"
    assert result["n_observed_pairs"] == 44
    assert len(pairs) == 44
    assert np.isfinite(result["var_reduction_pct"])
