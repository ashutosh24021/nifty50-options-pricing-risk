import json

from optionlab.cli import main


def test_demo_cli_creates_auditable_outputs(tmp_path):
    out = tmp_path / "demo"
    main(["demo", "--n-paths", "2000", "--n-days", "90", "--tree-steps", "31",
          "--out", str(out), "--seed", "7"])
    assert (out / "summary.json").is_file()
    assert (out / "scored_quotes.csv").is_file()
    assert (out / "figures" / "regime_bias.png").is_file()
    assert (out / "figures" / "hedging_var.png").is_file()
    contents = json.loads((out / "summary.json").read_text())
    assert contents["simulation"]["n_paths"] == 2000
    assert contents["simulation"]["headline_18_percent_resume_claim_verified"] is False
    assert contents["dataset"].startswith("SYNTHETIC")


def test_udiff_bhavcopy_cli_with_local_spot_history(tmp_path):
    import pandas as pd
    from optionlab.synthetic import generate_synthetic

    quotes, spot = generate_synthetic(seed=9, n_days=90)
    d = quotes["quote_date"].max()
    same_date = quotes.loc[quotes["quote_date"] == d]
    udiff = pd.DataFrame({
        "FinInstrmTp": "IDO", "TckrSymb": "NIFTY",
        "XpryDt": same_date["expiry"].dt.strftime("%Y-%m-%d"),
        "StrkPric": same_date["strike"].to_numpy(),
        "OptnTp": same_date["option_type"].to_numpy(),
        "ClsPric": same_date["market_price"].to_numpy(),
        "TradDt": same_date["quote_date"].dt.strftime("%Y-%m-%d"),
        "TtlTradgVol": same_date["volume"].to_numpy(),
    })
    path = tmp_path / "nse.csv"
    udiff.to_csv(path, index=False)
    history = tmp_path / "spot.csv"
    spot.to_csv(history, index=False)
    out = tmp_path / "market"
    main(["analyze", "--bhavcopy", str(path), "--spot-history", str(history),
          "--n-paths", "2000", "--tree-steps", "25", "--out", str(out)])
    j = json.loads((out / "summary.json").read_text())
    assert j["pricing_metrics"]["n_quotes"] == len(same_date)
    assert j["dataset"].startswith("USER-SUPPLIED")
