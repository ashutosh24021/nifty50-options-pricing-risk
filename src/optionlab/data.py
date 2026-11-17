"""Normalize local NSE bhavcopies and construct look-ahead-safe market features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["quote_date", "expiry", "strike", "option_type", "market_price"]


def load_spot_history(path: str | Path) -> pd.DataFrame:
    """Read date,close CSV; compute 21-observation lagged annualized realized IV."""
    spot = pd.read_csv(path)
    if not {"date", "close"}.issubset(spot.columns):
        raise ValueError("spot history requires date and close columns")
    spot["date"] = pd.to_datetime(spot["date"], errors="coerce")
    spot["close"] = pd.to_numeric(spot["close"], errors="coerce")
    if spot[["date", "close"]].isna().any().any() or (spot["close"] <= 0).any():
        raise ValueError("spot history contains invalid date/price")
    spot = spot.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    # σ_t uses returns ending at t-1 (no same-day or future observations).
    log_ret = np.log(spot["close"]).diff()
    spot["historical_volatility"] = log_ret.rolling(21, min_periods=21).std().shift(1) * np.sqrt(252)
    return spot


def load_nse_bhavcopy(path: str | Path) -> pd.DataFrame:
    """Read local legacy/UDiFF NSE F&O .csv or single-member .zip; return NIFTY options.

    This is an adapter, not a scraper; download official NSE files manually.
    Schema selection uses column names, NOT an assumed changeover date.
    """
    raw = pd.read_csv(path, low_memory=False)
    if {"FinInstrmTp", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "ClsPric"}.issubset(raw.columns):
        raw = raw.loc[(raw["FinInstrmTp"].astype(str).str.strip() == "IDO") &
                      (raw["TckrSymb"].astype(str).str.strip() == "NIFTY")].copy()
        date_column = "TradDt" if "TradDt" in raw.columns else "BizDt"
        mapping = {date_column: "quote_date", "XpryDt": "expiry", "StrkPric": "strike",
                   "OptnTp": "option_type", "ClsPric": "market_price"}
        if "TtlTradgVol" in raw:
            mapping["TtlTradgVol"] = "volume"
    elif {"INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "CLOSE", "TIMESTAMP"}.issubset(raw.columns):
        raw = raw.loc[(raw["INSTRUMENT"].astype(str).str.strip() == "OPTIDX") &
                      (raw["SYMBOL"].astype(str).str.strip() == "NIFTY")].copy()
        mapping = {"TIMESTAMP": "quote_date", "EXPIRY_DT": "expiry", "STRIKE_PR": "strike",
                   "OPTION_TYP": "option_type", "CLOSE": "market_price"}
        if "CONTRACTS" in raw:
            mapping["CONTRACTS"] = "volume"
    else:
        raise ValueError("unrecognized NSE bhavcopy schema: expected legacy or UDiFF F&O")
    result = raw[list(mapping)].rename(columns=mapping)
    if result.empty:
        raise ValueError("no NIFTY index options found in NSE file")
    return result.reset_index(drop=True)


def prepare_quotes(quotes: pd.DataFrame, spot_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate and enrich premiums. Drop stale, expired and missing-volatility quotes.

    The historical-volatility input is estimated from trailing spot observations,
    NOT calibrated from the same market option premium being evaluated.
    """
    missing = set(REQUIRED) - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes missing required columns: {sorted(missing)}")
    df = quotes.copy()
    for col in ("quote_date", "expiry"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    for col in ("strike", "market_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["option_type"] = df["option_type"].astype(str).str.upper().str.strip().replace(
        {"CALL": "CE", "PUT": "PE"}
    )
    if spot_history is not None:
        history = spot_history[["date", "close", "historical_volatility"]].copy()
        df = df.merge(history.rename(columns={"date": "quote_date", "close": "spot_history_close",
                                               "historical_volatility": "lagged_hv"}),
                      on="quote_date", how="left", validate="many_to_one")
        if "spot" not in df:
            df["spot"] = df["spot_history_close"]
        else:
            df["spot"] = pd.to_numeric(df["spot"], errors="coerce").fillna(df["spot_history_close"])
        if "historical_volatility" not in df:
            df["historical_volatility"] = df["lagged_hv"]
        else:
            df["historical_volatility"] = pd.to_numeric(df["historical_volatility"], errors="coerce").fillna(df["lagged_hv"])
    required_extra = ["spot", "historical_volatility"]
    for col in required_extra:
        if col not in df:
            raise ValueError(f"{col} missing. Supply a spot-history CSV or add {col} to quotes")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, default in [("risk_free_rate", 0.065), ("dividend_yield", 0.012)]:
        if col not in df:
            # Defaults are explicit analytical assumptions, not contemporaneous market observations.
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.loc[df["volume"].fillna(0) > 0].copy()
    df["maturity"] = (df["expiry"] - df["quote_date"]).dt.days / 365.0
    valid = (df["spot"] > 0) & (df["strike"] > 0) & (df["market_price"] > 0) & (
        df["historical_volatility"] > 0
    ) & (df["maturity"] > 0) & df["option_type"].isin(["CE", "PE"])
    valid &= df[["quote_date", "expiry", "spot", "strike", "market_price", "historical_volatility",
                 "risk_free_rate", "dividend_yield", "maturity"]].notna().all(axis=1)
    df = df.loc[valid].copy()
    if df.empty:
        raise ValueError("no valid rows: check dates, volume, historical volatility and spot alignment")
    df = df.drop_duplicates(["quote_date", "expiry", "strike", "option_type"]).sort_values(
        ["quote_date", "expiry", "strike", "option_type"]
    ).reset_index(drop=True)
    df["log_moneyness"] = np.log(df["spot"] / df["strike"])
    df["moneyness_bucket"] = np.select(
        [df["log_moneyness"].abs() < 0.01,
         ((df["option_type"] == "CE") & (df["log_moneyness"] > 0)) |
         ((df["option_type"] == "PE") & (df["log_moneyness"] < 0))],
        ["ATM", "ITM"], default="OTM"
    )
    # Quantiles over distinct dates, so days with more listed strikes don't dominate cutoffs.
    daily_hv = df.groupby("quote_date")["historical_volatility"].median()
    cut1, cut2 = daily_hv.quantile([1/3, 2/3]).to_numpy()
    df["vol_regime"] = np.select(
        [df["historical_volatility"] <= cut1, df["historical_volatility"] <= cut2],
        ["Low", "Medium"], default="High"
    )
    return df
