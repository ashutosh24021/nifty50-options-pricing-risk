# Methodology and Assumptions

This document specifies a transparent pipeline rather than asserting that any simulated result occurred in the Indian options market.

## 1. Contracts, rates and units

- Underlying: NIFTY 50 index spot level, `S` in index points; strike `K` likewise.
- European-style exercise, call `CE` and put `PE`, with continuously compounded risk-free rate `r` and continuous dividend yield `q`.
- Maturity `T = (expiry_date - quote_date).days / 365`; all dates are end-of-day snapshots. No actual expiry settlement mechanics are reproduced.
- Annualized historical volatility uses the trailing standard deviation of **21 log returns**, times sqrt(252), shifted by one observation. For a quote at time `t`, no return from time `t` or later is used in the predictor.
- The demo uses **assumed** `r=6.5%`, `q=1.2%`, artificial spot shocks and a stylized volatility surface. These are **not observed** historical values.

## 2. Black–Scholes–Merton model

For maturity `T > 0` and volatility `σ > 0`:

```
d1 = [ln(S/K) + (r - q + 0.5*σ²)T] / (σ√T)
d2 = d1 - σ√T
Call = S exp(-qT) N(d1) - K exp(-rT) N(d2)
Put  = K exp(-rT) N(-d2) - S exp(-qT) N(-d1)
```

With no arbitrage, European call-put parity satisfies `C - P = S exp(-qT) - K exp(-rT)`.

**Greeks** are computed analytically:

```
Call delta = exp(-qT) N(d1)
Put delta  = exp(-qT) (N(d1) - 1)
Gamma      = exp(-qT) n(d1) / (S σ √T)
Vega       = S exp(-qT) n(d1) √T
Theta_call = -S exp(-qT)n(d1)σ/(2√T) - rK exp(-rT)N(d2) + qS exp(-qT)N(d1)
Theta_put  = -S exp(-qT)n(d1)σ/(2√T) + rK exp(-rT)N(-d2) - qS exp(-qT)N(-d1)
Rho_call   = KT exp(-rT) N(d2)
Rho_put    = -KT exp(-rT) N(-d2)
```

Here `N` is the standard-normal CDF and `n` its density. **Vega and Rho are quoted per change of 1.0**, so a one-percentage-point input change corresponds to `vega * 0.01` or `rho * 0.01`. Theta is per **year**; divide by 365 for an approximate calendar-day decay.

## 3. CRR binomial model cross-check

For `N` steps of length `dt = T/N`:

```
u = exp(σ√dt), d = 1/u
p = [exp((r-q)dt)-d] / (u-d)
V_node = exp(-r dt) [p * V_up + (1-p) * V_down]
```

Terminal values are `max(ST-K, 0)` for calls and `max(K-ST, 0)` for puts. European backward induction discounts the tree without allowing early exercise. The implementation rejects `p` outside `[0,1]` and performs convergence tests against BSM. Default tree depth is 201: finite numerical differences are expected.

## 4. Pricing bias and market regimes

For each quote `i`, calculate `error_i = observed_market_premium_i - BSM_price_i` using **lagged historical** `σ_i` (not `implied_volatility_i`).

```
Signed bias       = mean(error)
MAE               = mean(abs(error))
RMSE              = sqrt(mean(error²))
Binomial error    = binomial_price - BSM_price
```

For **descriptive segmentation only**, define Low, Medium and High regimes using terciles of median per-date lagged historical volatility. The regime thresholds are estimated over the full analyzed sample and are not offered as a trained forecasting classifier. The `ITM`, `ATM`, `OTM` feature uses `abs(log(S/K))<0.01` for ATM, and adjusts ITM by option side. Quotes with expired contracts, missing spots/volatility, nonpositive premiums, or zero reported volume are excluded. No option IV from the same quoted premium is used to validate that premium; market-implied IV is only an additional output column.

### Identifiability warning

The pricing residual is **not** uniquely attributable to Black–Scholes model misspecification: it can also reflect poor lagged realized volatility as a forecast, stale settlements, data mismatches, spreads, skew, interest rates, dividends and discrete trading. Check other vol estimators and strike/liquidity filters before interpreting signed bias.

## 5. Risk-neutral 100,000-path pricing Monte Carlo

For a single option snapshot:

```
Z ~ Normal(0,1)
ST(Z)  = S exp((r-q-0.5σ²)T + σ√T Z)
ST(-Z) = S exp((r-q-0.5σ²)T - σ√T Z)
MC price = discounted mean of call/put payoffs for all 100k paths
```

50,000 independent draws produce 100,000 **antithetic paths**. The standard error is computed across the **50,000 independent pair means** to avoid understating uncertainty by pretending the paired paths are independent. A normal-approximation 95% confidence interval is reported; compare its center with BSM. This is a numerical pricing-convergence check, not a market outperformance claim.

## 6. Model residual bootstrap (distinct from MC pricing)

Sampling 100,000 with replacement from observed `market - BSM` residuals gives a **descriptive resampled distribution** with median, tails and standard deviation. It does **not** create 100,000 independent observed option prices, establish predictive error rates or address overlapping dates/contracts. It should not be labeled as portfolio VaR.

## 7. Stochastic-volatility scenario VaR

Pick one most-recent available near-ATM call with more maturity remaining than the simulated horizon. Model a **short call** and a short call initially delta-neutralized by holding `+Delta` underlying units. For the default one-calendar-day horizon `h=1/365`, 100,000 scenario draws:

```
Zs, Zu ~ independent N(0,1)
Zv = ρ Zs + sqrt(1-ρ²) Zu; ρ = -0.4
S_new = S exp((r-q-0.5σ_realized²)h + σ_realized√h Zs)
σ_new = clip(σ_start * exp(0.035√(horizon_days) * Zv), 0.01, 3)
C_new = BSM(S_new, K, T-h, r, σ_new, q)
P&L_unhedged = C_start exp(rh) - C_new
P&L_hedged = P&L_unhedged + Delta*(S_new-S)
              + Delta*S*(exp(qh)-exp(rh))
              - abs(Delta*S)*(transaction_cost_bps/10_000)
```

**Important:** scenario drift `r-q` is a convenient approximation, **not** an estimated physical drift. For a real risk engine, calibrate the physical return distribution and spot/IV joint dynamics, and account for execution spread. The position is **one unit**, hedge is placed only at the start, and opening transaction cost defaults to 1 basis point of hedge notional. Delta exposure is conceptual index exposure rather than an actual replicating futures trade.

Loss is `L=-P&L`. Report the 99th percentile loss as `VaR99=max(0, quantile(L,0.99))`; `ES99` is mean loss in the tail beyond that empirical threshold. Then:

```
VaR reduction (%) = 100 * (VaR99_unhedged - VaR99_hedged) / VaR99_unhedged
```

A given percent depends strongly on option side, risk factors, maturity, costs and sample. **Never substitute a desired 18% in place of a measured number.** The default synthetic setup produces its own reported result.

## 8. Historical quote-pair delta-hedge backtest

For two successive observed quotes on **the same expiry, strike and option type**, separated by at most four calendar days and no later than expiry:

```
unhedged = Premium_t * exp(r*h) - Premium_t+1
hedged   = unhedged + Delta_t * (Spot_t+1 - Spot_t)
           + Delta_t * Spot_t * (exp(q*h) - exp(r*h))
           - abs(Delta_t * Spot_t) * cost_bps / 10_000
```

`Delta_t` uses only time-`t` information. Bootstrapping **paired** `(unhedged, hedged)` observed P&Ls with 100,000 repeated selections preserves the dependence **within** each observation but does not resolve overlapping contracts, serial dependence, underlying proxy issues or quote/trading synchronization. The analysis prints both actual paired quote count and bootstrap count. With under 30 observed pairs it declines to issue a descriptive 99%-VaR estimate. For production VaR, prefer date-block bootstrap, transaction-level execution and clean quote synchronization, with truly out-of-sample verification.

## 9. Material limitations

- Same-day EOD quote alignment might not represent executable prices; enforce realistic mid/bid/ask & liquidity filters.
- Historical volatility is a basic lagged baseline, not a stochastic-volatility calibration or volatility-surface fit.
- Actual NIFTY weekly expiry patterns, exchange lots, currency funding, volatility risk premia, trading holidays and futures basis are not automatically modeled.
- Low/Medium/High labels are retrospective groupings; cross-sectional error samples are dependent.
- Scenario delta-hedging is single-shot; no intraday/dynamic rebalancing or futures hedging slippage is included.
- `100,000` is a numerical sampling choice, not an assertion that there were that many historical trades.
- Nothing in the synthetic demo authenticates a resume result. See `docs/interview_walkthrough.md` for what evidence would be needed to substantiate it.

## 10. Further work

1. Add date-block/bootstrap confidence intervals and a fixed train/test split for volatility regime thresholds.
2. Replace constant-vol GBM scenarios with empirical joint spot/IV shocks, or Heston/SABR calibration when reliable data exist.
3. Model hedge execution with NIFTY futures, calendar-based roll, carry/basis, exchange fees, updated contract lot sizes, and intraday rebalancing.
4. Estimate risk on a **defined book of positions**, rather than one representative short call.
5. Compare realized-vol forecasts against term-structure and skew-aware models without leaking current market IV into evaluation.

## Data references

- [NSE equity derivatives F&O historical reports](https://www.nseindia.com/all-reports-derivatives)
- [NIFTY 50 derivatives product specifications](https://www.nseindia.com/static/products-services/equity-derivatives-nifty50)
- [NSE Data and Analytics historical F&O data background](https://archives.nseindia.com/content/press/Data_Details_F_n_O.pdf)

External sources are for context; the default dataset is programmatically synthesized.
