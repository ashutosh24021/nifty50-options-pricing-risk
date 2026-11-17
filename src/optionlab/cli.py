"""Command line entry point. Example: optionlab demo --n-paths 100000."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from optionlab.analysis import grouped_bias, metrics, score_options
from optionlab.data import load_nse_bhavcopy, load_spot_history, prepare_quotes
from optionlab.monte_carlo import (
    bootstrap_pricing_errors, empirical_delta_hedge_backtest, monte_carlo_price,
    simulate_delta_hedge_var,
)
from optionlab.plots import save_charts
from optionlab.pricing import black_scholes, crr_binomial, greeks
from optionlab.synthetic import generate_synthetic


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, allow_nan=False, default=str) + "\n")


def research_pipeline(
    quotes: pd.DataFrame, output_dir: Path, n_paths: int, seed: int,
    tree_steps: int, dataset_label: str, horizon_days: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = score_options(quotes, tree_steps=tree_steps)
    regime = grouped_bias(scored, "vol_regime")
    moneyness = grouped_bias(scored, "moneyness_bucket")
    # Single, reproducible hedge example: most-recent liquid ATM call with T > horizon.
    eligible = scored.loc[
        (scored["option_type"] == "CE") & (scored["maturity"] > horizon_days/365),
    ].copy()
    if eligible.empty:
        raise ValueError("need at least one call with maturity longer than the hedge horizon")
    eligible["abs_log_moneyness"] = eligible["log_moneyness"].abs()
    hedge_date = eligible["quote_date"].max()
    selected = eligible.loc[eligible["quote_date"] == hedge_date].sort_values(
        ["abs_log_moneyness", "maturity", "strike"]
    ).iloc[0]
    a = dict(spot=float(selected["spot"]), strike=float(selected["strike"]),
             maturity=float(selected["maturity"]), rate=float(selected["risk_free_rate"]),
             sigma=float(selected["historical_volatility"]), kind="CE",
             dividend=float(selected["dividend_yield"]))
    mc = monte_carlo_price(**a, n_paths=n_paths, seed=seed)
    hedge, u_pnl, h_pnl = simulate_delta_hedge_var(
        **a, n_paths=n_paths, seed=seed+1, horizon_days=horizon_days,
        return_pnl=True,
    )
    boot = bootstrap_pricing_errors(scored["pricing_error"].to_numpy(), n_draws=n_paths, seed=seed+2)
    empirical, matched = empirical_delta_hedge_backtest(scored, n_draws=n_paths, seed=seed+3)
    matched.to_csv(output_dir / "historical_hedge_pairs.csv", index=False)
    scored.to_csv(output_dir / "scored_quotes.csv", index=False)
    regime.to_csv(output_dir / "bias_by_regime.csv", index=False)
    moneyness.to_csv(output_dir / "bias_by_moneyness.csv", index=False)
    save_charts(scored, regime, moneyness, u_pnl, h_pnl, output_dir / "figures")
    summary = {
        "dataset": dataset_label,
        "note": "Synthetic results are illustrative only; never represent these as a real NSE backtest.",
        "model": {"volatility": "lagged 21-session realized volatility, annualized by sqrt(252)",
                  "maturity": "calendar days / 365", "tree_steps": tree_steps,
                  "error_sign": "market_price minus Black-Scholes price"},
        "pricing_metrics": metrics(scored),
        "error_distribution_bootstrap": boot,
        "hedge_snapshot": {
            "quote_date": str(selected["quote_date"].date()),
            "expiry": str(selected["expiry"].date()),
            "spot": a["spot"], "strike": a["strike"], "historical_volatility": a["sigma"],
            "market_premium": float(selected["market_price"]),
            "model_premium": float(selected["bs_price"]),
            "risk_free_rate": a["rate"], "dividend_yield": a["dividend"],
        },
        "monte_carlo_pricing": asdict(mc),
        "hedge_var": asdict(hedge),
        "historical_hedge_backtest": empirical,
        "simulation": {
            "n_paths": n_paths, "random_seed": seed,
            "risk_method": "GBM spot + correlated stochastic IV shocks, fixed initial BSM delta",
            "underlying_unit": "index points per one unit of the index option",
            "position": "short 1 European call; initial delta hedge via index exposure",
            "headline_18_percent_resume_claim_verified": False,
        },
    }
    _write_json(output_dir / "summary.json", summary)
    print(f"Dataset: {dataset_label}; {len(scored)} eligible option quotes")
    print(f"BSM MAE: {summary['pricing_metrics']['mae']:.3f} index pts")
    print(f"MC benchmark: {mc.price:.3f} ± {1.96*mc.standard_error:.3f} vs BSM {mc.analytic_price:.3f}")
    print(f"Short call VaR(99%): {hedge.var99_unhedged:.3f} -> {hedge.var99_hedged:.3f} index pts")
    print(f"Scenario VaR reduction: {hedge.var_reduction_pct:.2f}% (not a verified 18% backtest)")
    if empirical["status"] == "descriptive_backtest":
        print(f"Paired historical-quote bootstrap: {empirical['n_observed_pairs']} pairs; "
              f"VaR reduction {empirical['var_reduction_pct']:.2f}% (descriptive)")
    else:
        print(f"Historical hedge backtest: {empirical['status']}")
    print(f"Charts and CSVs: {output_dir.resolve()}")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optionlab", description="NIFTY 50 option pricing and risk analytics",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="fully synthetic, end-to-end reproducible research demo")
    analyze = commands.add_parser("analyze", help="analyze local normalized quotes or official NSE bhavcopy")
    for p in (demo, analyze):
        p.add_argument("--out", default="results/demo", help="directory to create")
        p.add_argument("--n-paths", type=int, default=100_000)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--tree-steps", type=int, default=201)
        p.add_argument("--horizon-days", type=int, default=1)
    demo.add_argument("--n-days", type=int, default=320)
    source = analyze.add_mutually_exclusive_group(required=True)
    source.add_argument("--quotes", help="clean normalized option-premium CSV")
    source.add_argument("--bhavcopy", nargs="+", help="one or more legacy/UDiFF NSE F&O CSV or ZIP files")
    analyze.add_argument("--spot-history", help="spot closes CSV with date,close columns")
    analyze.add_argument("--max-abs-log-moneyness", type=float, default=None,
                         help="optional performance filter: abs(ln(spot/strike)) <= value")
    analyze.add_argument("--max-days-to-expiry", type=int, default=None,
                         help="optional performance filter: days to expiry <= value")
    analyze.add_argument("--rate", type=float, default=0.065,
                         help="ASSUMED annualized continuously compounded risk-free rate")
    analyze.add_argument("--dividend", type=float, default=0.012,
                         help="ASSUMED annualized continuously compounded dividend yield")
    price = commands.add_parser("price", help="calculate BSM price, CRR benchmark and Greeks")
    price.add_argument("--spot", type=float, required=True)
    price.add_argument("--strike", type=float, required=True)
    price.add_argument("--days", type=int, default=30)
    price.add_argument("--rate", type=float, default=0.065)
    price.add_argument("--vol", type=float, required=True, help="annualized decimal volatility")
    price.add_argument("--dividend", type=float, default=0.012)
    price.add_argument("--type", choices=["CE", "PE"], default="CE")
    price.add_argument("--tree-steps", type=int, default=501)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    if args.command == "price":
        a = dict(spot=args.spot, strike=args.strike, maturity=args.days/365,
                 rate=args.rate, sigma=args.vol, kind=args.type, dividend=args.dividend)
        print(json.dumps({"bsm": black_scholes(**a), "crr": crr_binomial(**a, steps=args.tree_steps),
                          "greeks": asdict(greeks(**a))}, indent=2))
        return
    if args.n_paths < 100 or args.n_paths % 2:
        raise SystemExit("--n-paths must be an even integer >=100")
    if args.tree_steps <= 0 or args.horizon_days <= 0:
        raise SystemExit("--tree-steps and --horizon-days must be positive")
    output_dir = Path(args.out)
    if args.command == "demo":
        if args.n_days < 70:
            raise SystemExit("--n-days must be >=70")
        quotes, spot = generate_synthetic(seed=args.seed, n_days=args.n_days)
        output_dir.mkdir(parents=True, exist_ok=True)
        quotes.to_csv(output_dir / "synthetic_option_quotes.csv", index=False)
        spot.to_csv(output_dir / "synthetic_spot_history.csv", index=False)
        research_pipeline(quotes, output_dir, args.n_paths, args.seed,
                          args.tree_steps, "SYNTHETIC DEMO - NOT NSE", args.horizon_days)
        return
    quotes = (pd.concat([load_nse_bhavcopy(p) for p in args.bhavcopy], ignore_index=True)
              if args.bhavcopy else pd.read_csv(args.quotes))
    if args.bhavcopy:
        if args.spot_history is None:
            raise SystemExit("--spot-history is required for --bhavcopy")
        quotes["risk_free_rate"] = args.rate
        quotes["dividend_yield"] = args.dividend
    spot_history = load_spot_history(args.spot_history) if args.spot_history else None
    clean_quotes = prepare_quotes(quotes, spot_history)
    if args.max_abs_log_moneyness is not None:
        clean_quotes = clean_quotes.loc[clean_quotes["log_moneyness"].abs() <= args.max_abs_log_moneyness]
    if args.max_days_to_expiry is not None:
        clean_quotes = clean_quotes.loc[clean_quotes["maturity"] <= args.max_days_to_expiry/365]
    if clean_quotes.empty:
        raise SystemExit("all quotes removed by quality/performance filters")
    research_pipeline(clean_quotes, output_dir, args.n_paths, args.seed,
                      args.tree_steps, "USER-SUPPLIED MARKET QUOTES (SOURCE NOT VERIFIED)",
                      args.horizon_days)


if __name__ == "__main__":
    main()
