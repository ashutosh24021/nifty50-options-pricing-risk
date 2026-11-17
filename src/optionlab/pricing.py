"""European BSM pricing, analytical Greeks, CRR convergence and implied volatility.

All prices are quoted in *index points per underlying unit*. Rates, dividend
rates and volatilities are decimals (e.g. 0.20 means 20% p.a.). Time is years.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr


def _validate(spot: float, strike: float, maturity: float, sigma: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be strictly positive")
    if maturity < 0 or sigma < 0:
        raise ValueError("maturity and sigma must not be negative")


def _kind(kind: str) -> str:
    k = str(kind).lower()
    if k in ("call", "ce", "c"):
        return "call"
    if k in ("put", "pe", "p"):
        return "put"
    raise ValueError("kind must be call/put (or CE/PE)")


def intrinsic(spot: float, strike: float, kind: str) -> float:
    k = _kind(kind)
    return max(spot - strike, 0.0) if k == "call" else max(strike - spot, 0.0)


def black_scholes(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> float:
    """European Black–Scholes–Merton option with continuous dividend yield."""
    _validate(spot, strike, maturity, sigma)
    k = _kind(kind)
    if maturity == 0:
        return intrinsic(spot, strike, k)
    if sigma == 0:
        terminal_forward = spot * exp((rate - dividend) * maturity)
        return exp(-rate * maturity) * intrinsic(terminal_forward, strike, k)
    vol_term = sigma * sqrt(maturity)
    d1 = (log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * maturity) / vol_term
    d2 = d1 - vol_term
    if k == "call":
        return float(spot * exp(-dividend * maturity) * ndtr(d1)
                     - strike * exp(-rate * maturity) * ndtr(d2))
    return float(strike * exp(-rate * maturity) * ndtr(-d2)
                 - spot * exp(-dividend * maturity) * ndtr(-d1))


def black_scholes_array(
    spot: np.ndarray,
    strike: float,
    maturity: float,
    rate: float,
    sigma: np.ndarray,
    kind: str = "call",
    dividend: float = 0.0,
) -> np.ndarray:
    """Vectorized repricing for Monte Carlo paths; sigma may vary by path."""
    s, v = np.broadcast_arrays(np.asarray(spot, dtype=float), np.asarray(sigma, dtype=float))
    if np.any(s <= 0) or np.any(v < 0):
        raise ValueError("all spots must be >0 and volatilities >=0")
    if strike <= 0 or maturity < 0:
        raise ValueError("strike must be >0 and maturity >=0")
    k = _kind(kind)
    if maturity == 0:
        return np.maximum(s - strike, 0) if k == "call" else np.maximum(strike - s, 0)
    tiny = v < 1e-12
    effective = np.maximum(v, 1e-12)
    d1 = (np.log(s / strike) + (rate - dividend + 0.5 * effective**2) * maturity) / (
        effective * np.sqrt(maturity)
    )
    d2 = d1 - effective * np.sqrt(maturity)
    if k == "call":
        out = s * np.exp(-dividend * maturity) * ndtr(d1) - strike * np.exp(-rate * maturity) * ndtr(d2)
    else:
        out = strike * np.exp(-rate * maturity) * ndtr(-d2) - s * np.exp(-dividend * maturity) * ndtr(-d1)
    if np.any(tiny):
        forward = s * np.exp((rate - dividend) * maturity)
        floor = np.maximum(forward - strike, 0) if k == "call" else np.maximum(strike - forward, 0)
        out = np.where(tiny, np.exp(-rate * maturity) * floor, out)
    return out


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float  # change in index points per +1.0 volatility, not per 1%
    theta: float  # change in index points per year of calendar time
    rho: float  # change in index points per +1.0 rate


def greeks(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> Greeks:
    """Analytical BSM Greeks for positive maturity and volatility."""
    _validate(spot, strike, maturity, sigma)
    if maturity <= 0 or sigma <= 0:
        raise ValueError("Greeks undefined or discontinuous at zero maturity/volatility")
    k = _kind(kind)
    st = sqrt(maturity)
    d1 = (log(spot / strike) + (rate - dividend + 0.5 * sigma**2) * maturity) / (sigma * st)
    d2 = d1 - sigma * st
    pdf = np.exp(-0.5 * d1 * d1) / sqrt(2.0 * np.pi)
    disc_q = exp(-dividend * maturity)
    disc_r = exp(-rate * maturity)
    common = spot * disc_q * pdf * sigma / (2 * st)
    gamma = disc_q * pdf / (spot * sigma * st)
    vega = spot * disc_q * pdf * st
    if k == "call":
        delta = disc_q * ndtr(d1)
        theta = -common - rate * strike * disc_r * ndtr(d2) + dividend * spot * disc_q * ndtr(d1)
        rho = strike * maturity * disc_r * ndtr(d2)
    else:
        delta = disc_q * (ndtr(d1) - 1)
        theta = -common + rate * strike * disc_r * ndtr(-d2) - dividend * spot * disc_q * ndtr(-d1)
        rho = -strike * maturity * disc_r * ndtr(-d2)
    return Greeks(float(delta), float(gamma), float(vega), float(theta), float(rho))


def crr_binomial(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    steps: int = 300,
    kind: str = "call",
    dividend: float = 0.0,
) -> float:
    """Cox–Ross–Rubinstein recombining tree, European exercise (O(steps²))."""
    _validate(spot, strike, maturity, sigma)
    k = _kind(kind)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if maturity == 0 or sigma == 0:
        return black_scholes(spot, strike, maturity, rate, sigma, k, dividend)
    dt = maturity / steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    prob_up = (exp((rate - dividend) * dt) - d) / (u - d)
    if not 0 <= prob_up <= 1:
        raise ValueError("invalid risk-neutral probability: raise tree steps or check inputs")
    j = np.arange(steps + 1)
    terminal = spot * (u ** j) * (d ** (steps - j))
    values = np.maximum(terminal - strike, 0) if k == "call" else np.maximum(strike - terminal, 0)
    discount = exp(-rate * dt)
    for _ in range(steps):
        values = discount * (prob_up * values[1:] + (1 - prob_up) * values[:-1])
    return float(values[0])


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    kind: str = "call",
    dividend: float = 0.0,
) -> float:
    """Implied IV from observed premium; reject prices violating European bounds."""
    if maturity <= 0:
        raise ValueError("implied volatility requires positive maturity")
    k = _kind(kind)
    discounted_s = spot * exp(-dividend * maturity)
    discounted_k = strike * exp(-rate * maturity)
    lower = max(0.0, discounted_s - discounted_k) if k == "call" else max(0.0, discounted_k - discounted_s)
    upper = discounted_s if k == "call" else discounted_k
    if not lower - 1e-8 <= price <= upper + 1e-8:
        raise ValueError("market premium violates European no-arbitrage bounds")
    if price <= lower + 1e-8:
        return 0.0
    if price >= upper - 1e-8:
        return float("nan")  # no finite IV at upper bound
    f = lambda v: black_scholes(spot, strike, maturity, rate, v, k, dividend) - price
    upper_sigma = 1.0
    while f(upper_sigma) < 0 and upper_sigma < 32.0:
        upper_sigma *= 2.0
    if f(upper_sigma) < 0:
        return float("nan")
    return float(brentq(f, 1e-12, upper_sigma, xtol=1e-12))
