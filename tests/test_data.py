import numpy as np
import pandas as pd
import pytest

from optionlab.data import load_nse_bhavcopy, load_spot_history, prepare_quotes
from optionlab.synthetic import generate_synthetic


def test_synthetic_quotes_are_repeatable():
    first, spot1 = generate_synthetic(seed=123, n_days=110)
    second, spot2 = generate_synthetic(seed=123, n_days=110)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(spot1, spot2)
    assert set(first["option_type"]) == {"CE", "PE"}
    assert (first["historical_volatility"] > 0).all()


def test_realized_vol_uses_prior_data_only(tmp_path):
    p1 = tmp_path / "spot1.csv"
    p2 = tmp_path / "spot2.csv"
    dates = pd.bdate_range("2025-01-01", periods=45)
    close = 100 * np.exp(np.cumsum(np.sin(np.arange(45)) * .01))
    pd.DataFrame({"date": dates, "close": close}).to_csv(p1, index=False)
    close[40:] *= 5  # mutate FUTURE of observation 35, must not change its σ
    pd.DataFrame({"date": dates, "close": close}).to_csv(p2, index=False)
    first, second = load_spot_history(p1), load_spot_history(p2)
    assert first.at[35, "historical_volatility"] == pytest.approx(second.at[35, "historical_volatility"])
    assert np.isnan(first.at[19, "historical_volatility"])
    assert first.at[23, "historical_volatility"] > 0


def test_normalize_legacy_bhavcopy(tmp_path):
    p = tmp_path / "legacy.csv"
    pd.DataFrame([
        {"INSTRUMENT": "OPTIDX", "SYMBOL": "NIFTY", "EXPIRY_DT": "26-Jun-2025",
         "STRIKE_PR": 24500, "OPTION_TYP": "CE", "CLOSE": 210, "TIMESTAMP": "02-JUN-2025",
         "CONTRACTS": 10},
        {"INSTRUMENT": "OPTSTK", "SYMBOL": "INFY", "EXPIRY_DT": "26-Jun-2025",
         "STRIKE_PR": 1000, "OPTION_TYP": "CE", "CLOSE": 20, "TIMESTAMP": "02-JUN-2025",
         "CONTRACTS": 1},
    ]).to_csv(p, index=False)
    result = load_nse_bhavcopy(p)
    assert len(result) == 1
    assert result.iloc[0]["market_price"] == 210


def test_normalize_udiff_bhavcopy(tmp_path):
    p = tmp_path / "udiff.csv"
    pd.DataFrame([
        {"FinInstrmTp": "IDO", "TckrSymb": "NIFTY", "XpryDt": "2025-06-26",
         "StrkPric": 24500, "OptnTp": "PE", "ClsPric": 80, "BizDt": "2025-06-02",
         "TtlTradgVol": 45},
        {"FinInstrmTp": "IDF", "TckrSymb": "NIFTY", "XpryDt": "2025-06-26",
         "StrkPric": 0, "OptnTp": "XX", "ClsPric": 24500, "BizDt": "2025-06-02",
         "TtlTradgVol": 45},
    ]).to_csv(p, index=False)
    result = load_nse_bhavcopy(p)
    assert len(result) == 1
    assert result.iloc[0]["option_type"] == "PE"


def test_prepare_quotes_reject_expired_and_missing_hv():
    raw = pd.DataFrame({"quote_date": ["2025-02-01"]*2,
                        "expiry": ["2025-03-01", "2025-01-01"],
                        "spot": [200, 200], "strike": [200, 200],
                        "option_type": ["CE", "PE"], "market_price": [5, 5],
                        "historical_volatility": [.2, .2]})
    out = prepare_quotes(raw)
    assert len(out) == 1
    assert out.iloc[0]["vol_regime"] == "Low"
    raw["historical_volatility"] = np.nan
    with pytest.raises(ValueError, match="no valid"):
        prepare_quotes(raw)
