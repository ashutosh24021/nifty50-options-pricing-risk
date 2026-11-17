from math import exp

import numpy as np
import pytest

from optionlab.pricing import (
    black_scholes, black_scholes_array, crr_binomial, greeks, implied_volatility,
)


def test_atm_black_scholes_reference_value():
    # Standard textbook at-the-money European call (q=0).
    assert black_scholes(100, 100, 1, 0.05, 0.20) == pytest.approx(10.450583572, abs=1e-8)
    assert black_scholes(100, 100, 1, 0.05, 0.20, "put") == pytest.approx(5.573526022, abs=1e-8)


def test_put_call_parity_with_dividend():
    c = black_scholes(120, 115, 0.75, 0.055, 0.28, "CE", 0.012)
    p = black_scholes(120, 115, 0.75, 0.055, 0.28, "PE", 0.012)
    assert c - p == pytest.approx(120 * exp(-0.012 * 0.75) - 115 * exp(-0.055 * 0.75), abs=1e-10)


@pytest.mark.parametrize("kind", ["CE", "PE"])
def test_tree_converges_to_analytic(kind):
    bs = black_scholes(19000, 19200, 50/365, 0.065, 0.19, kind, 0.012)
    tree = crr_binomial(19000, 19200, 50/365, 0.065, 0.19, steps=1000, kind=kind, dividend=0.012)
    assert abs(bs-tree) < 0.35


def test_implied_vol_recovers_true_sigma():
    price = black_scholes(23000, 22500, 0.15, 0.066, 0.215, "PE", 0.01)
    assert implied_volatility(price, 23000, 22500, 0.15, 0.066, "PE", 0.01) == pytest.approx(0.215, abs=1e-9)


def test_delta_gamma_vega_against_finite_differences():
    s, k, t, r, v, q = 100.0, 105.0, 0.8, 0.04, 0.23, 0.01
    g = greeks(s, k, t, r, v, "CE", q)
    ds, dv = 1e-2, 1e-4
    base = black_scholes(s, k, t, r, v, "CE", q)
    plus = black_scholes(s+ds, k, t, r, v, "CE", q)
    minus = black_scholes(s-ds, k, t, r, v, "CE", q)
    assert g.delta == pytest.approx((plus - minus)/(2*ds), rel=1e-6)
    assert g.gamma == pytest.approx((plus - 2*base + minus)/ds**2, rel=1e-4)
    vol_p = black_scholes(s, k, t, r, v + dv, "CE", q)
    vol_m = black_scholes(s, k, t, r, v - dv, "CE", q)
    assert g.vega == pytest.approx((vol_p-vol_m)/(2*dv), rel=1e-7)


def test_vectorized_matches_scalar():
    s = np.array([95, 100, 110])
    v = np.array([.15, .22, .37])
    for kind in ("CE", "PE"):
        out = black_scholes_array(s, 100, .33, .05, v, kind, .01)
        expected = np.array([black_scholes(float(x), 100, .33, .05, float(y), kind, .01)
                             for x, y in zip(s, v)])
        assert np.allclose(out, expected, atol=1e-10)


def test_no_arbitrage_bounds_and_input_errors():
    with pytest.raises(ValueError, match="bounds"):
        implied_volatility(200, 100, 100, 1, .05)
    with pytest.raises(ValueError):
        black_scholes(-100, 100, .5, .05, .2)
    with pytest.raises(ValueError):
        crr_binomial(100, 100, .5, .05, .2, steps=0)
    assert black_scholes(95, 100, 0, .05, .2, "PE") == pytest.approx(5)
    assert black_scholes(100, 100, 1, .05, 0, "CE") > 0
