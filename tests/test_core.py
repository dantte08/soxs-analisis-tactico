import numpy as np
import pandas as pd
import pytest

from soxs.alerts import record_alert
from soxs.core import BASE_FEATURES, LABEL, _walk_return, build_features


def sample_prices(n=80):
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 20 + np.cumsum(np.sin(np.arange(n) / 4) * 0.1 + 0.02)
    return pd.DataFrame({"Date": dates, "Open": close, "High": close + .2, "Low": close - .2,
                         "Close": close, "Volume": 1000 + np.arange(n) * 10})


def test_last_row_has_no_future_target():
    frame = build_features(sample_prices())
    assert pd.isna(frame.iloc[-1][LABEL])


def test_known_rows_have_binary_target():
    frame = build_features(sample_prices())
    observed = frame[LABEL].dropna().unique().tolist()
    assert set(observed).issubset({0.0, 1.0})


def test_fred_asof_merge_uses_only_available_observations():
    prices = sample_prices(25)
    release_date = prices.loc[10, "Date"]
    fred = pd.DataFrame({"series_id": ["DFF"], "date": [release_date - pd.Timedelta(days=2)],
                         "available_date": [release_date], "value": [5.25], "description": ["rate"]})
    frame = build_features(prices, fred)
    assert pd.isna(frame.loc[9, "DFF"])
    assert frame.loc[10, "DFF"] == 5.25
    assert frame.loc[11, "DFF"] == 5.25


def test_backtest_caps_individual_position_at_twenty_sessions():
    n = 35
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=n), "Close": 10.0,
                          "next_return": 0.001, LABEL: 1, "probability_up": .9})
    result = _walk_return(frame, frame["probability_up"].to_numpy(), max_sessions=20, commission=0)
    # The strategy exits at the cap and may re-enter on a subsequent fresh row prediction.
    runs = result["position"].groupby((result["position"] != result["position"].shift()).cumsum()).sum()
    assert int(runs.max()) <= 20


def test_neutral_probability_does_not_open_position():
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=3), "Close": 10.0,
                          "next_return": .01, LABEL: 1, "probability_up": .50})
    result = _walk_return(frame, frame["probability_up"].to_numpy(), max_sessions=20, commission=0)
    assert int(result["position"].sum()) == 0


def test_exit_threshold_closes_open_position():
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=3), "Close": 10.0,
                          "next_return": .01, LABEL: 1, "probability_up": [.8, .4, .8]})
    result = _walk_return(frame, frame["probability_up"].to_numpy(), max_sessions=20, commission=0)
    assert result["position"].tolist() == [1, 0, 1]


def test_macd_features_are_created_and_included_in_model_features():
    frame = build_features(sample_prices())
    assert {"macd_pct", "macd_signal_pct", "macd_hist_pct"}.issubset(BASE_FEATURES)
    assert np.isfinite(frame.iloc[-1][["macd_pct", "macd_signal_pct", "macd_hist_pct"]].to_numpy(dtype=float)).all()


def test_entry_return_deducts_commission_on_position_change():
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=1), "Close": [10.0],
                          "next_return": [.01], LABEL: [1]})
    result = _walk_return(frame, np.array([.7]), entry=.6, exit_=.4, max_sessions=20, commission=.001)
    assert result.loc[0, "turnover"] == 1
    assert result.loc[0, "strategy_return_net"] == pytest.approx(.009)


def test_backtest_rejects_holding_period_above_twenty_sessions():
    frame = pd.DataFrame({"Date": pd.bdate_range("2024-01-01", periods=1), "Close": [10.0],
                          "next_return": [.01], LABEL: [1]})
    with pytest.raises(ValueError, match="20 sesiones"):
        _walk_return(frame, np.array([.7]), max_sessions=21)


def test_unknown_fred_series_is_ignored():
    prices = sample_prices(25)
    fred = pd.DataFrame({"series_id": ["UNKNOWN"], "date": [prices.loc[10, "Date"]],
                         "available_date": [prices.loc[10, "Date"]], "value": [1.0], "description": ["test"]})
    frame = build_features(prices, fred)
    assert "UNKNOWN" not in frame.columns


def test_alert_history_deduplicates_consecutive_same_event(tmp_path):
    path = tmp_path / "alerts.csv"
    first = record_alert("COMPRA PAPER", "REVISAR ENTRADA", .7, "test", order_id="order-1", path=path)
    duplicate = record_alert("COMPRA PAPER", "REVISAR ENTRADA", .7, "test", order_id="order-1", path=path)
    history = pd.read_csv(path)
    assert first[0] is True
    assert duplicate[0] is False
    assert len(history) == 1
