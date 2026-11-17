# Deterministic synthetic demo results

Reproduce from a clean checkout (after installing project dependencies):

```bash
optionlab demo --n-paths 100000 --seed 42 --n-days 320 --tree-steps 201 --horizon-days 1 --out results/demo
```

**Provenance:** Option quotes and NIFTY-scale spot prices are generated from a seeded, piecewise GBM process. Option premiums are constructed from a stylized skewed volatility surface, spreads and noise. **They are not actual NSE observations.** The data generator is in `src/optionlab/synthetic.py`.

## Model results

| Statistic | Value |
|---|---:|
| Quoted options evaluated | 700 |
| Historical-vol BSM MAE | 25.235 index pts |
| Historical-vol BSM RMSE | 30.339 index pts |
| Signed bias, market minus BSM | +10.985 index pts |
| Average absolute CRR-minus-BSM difference | 0.238 index pts |
| Sample BSM premium | 612.914 index pts |
| Sample risk-neutral MC premium | 613.677 index pts |
| MC standard error (antithetic pair average) | 2.151 index pts |
| MC 95% interval | [609.462, 617.892] |

## Hedge experiment (single synthetic short call)

- Snapshot: 2024-03-05, contract expiry 2024-04-02
- Spot = 16013.664; strike = 16000; lagged historical volatility = 0.324927
- Initial delta = 0.539196; 1-calendar-day horizon; 1 bp initial hedge transaction cost
- Simulated spot and volatility risk factors, including negative spot/IV shock correlation (-0.4)
- Market quote premium = 616.750; **risk scenario starts from theoretical premium = 612.914**

| Simulated 99% downside metric | Short call, unhedged | Short call, delta-hedged |
|---|---:|---:|
| VaR (index pts) | 378.832 | 72.433 |
| Expected shortfall (index pts) | 446.254 | 90.829 |

Under these **particular artificial assumptions**, the initial hedge reduced modeled VaR by **80.880%**. This **does not validate** the original reported **18%** result, guarantee an equivalent improvement in real markets or measure tradeable performance. The 100,000 paths are simulated scenarios, not 100,000 actual option trades.

Because this demo uses independent 28-day expiries for each sampled quote date, it intentionally has **no repeated same-contract close pairs**. The quote-pair historical hedge backtest therefore returns `unavailable`, rather than silently presenting synthetic scenario VaR as a historical backtest.

The canonical machine-readable output is generated at `results/demo/summary.json`; a copy from the committed run is provided as [`docs/demo_summary.json`](demo_summary.json).
