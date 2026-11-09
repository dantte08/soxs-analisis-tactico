from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

BASE_FEATURES = ["return_1d", "rsi_14", "close_sma5", "close_sma20", "volatility_20", "volume_change",
                 "macd_pct", "macd_signal_pct", "macd_hist_pct"]
FRED_FEATURES = ["DFF", "DGS10", "VIXCLS", "UNRATE"]
LABEL = "target_next_day_up"


@dataclass
class Analysis:
    prices: pd.DataFrame
    featured: pd.DataFrame
    holdout: pd.DataFrame
    comparison: pd.DataFrame
    metrics: dict[str, float]
    model_name: str
    probability_up: float
    signal: str
    feature_cols: list[str]
    quality: dict[str, float]
    model: Any


def download_prices(period: str = "5y") -> pd.DataFrame:
    """Download adjusted daily SOXS OHLCV and retain a compact ETL quality report."""
    raw = yf.download("SOXS", period=period, auto_adjust=True, progress=False, threads=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance no devolvió datos de SOXS. Revisa tu conexión e inténtalo otra vez.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    raw_rows = len(raw)
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "Date"
    frame = raw.reset_index()
    needed = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in frame]
    if missing:
        raise RuntimeError("Faltan columnas de mercado: " + ", ".join(missing))
    for col in needed[1:]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    missing_values = int(frame[needed].isna().sum().sum())
    duplicates = int(frame["Date"].duplicated().sum())
    bad_close = int((frame["Close"] <= 0).fillna(False).sum())
    bad_volume = int((frame["Volume"] < 0).fillna(False).sum())
    bad_range = int((frame["High"] < frame["Low"]).fillna(False).sum())
    clean = frame.drop_duplicates("Date", keep="last").dropna(subset=needed).copy()
    clean = clean.loc[(clean.Close > 0) & (clean.Volume >= 0) & (clean.High >= clean.Low)]
    clean = clean.sort_values("Date").reset_index(drop=True)
    dropped = raw_rows - len(clean)
    quality = {
        "rows_downloaded": int(raw_rows),
        "rows_valid": int(len(clean)),
        "rows_discarded": int(dropped),
        "discarded_pct": float(dropped / raw_rows * 100) if raw_rows else 0.0,
        "duplicate_dates": duplicates,
        "nonpositive_close": bad_close,
        "negative_volume": bad_volume,
        "high_below_low": bad_range,
        "missing_market_values": missing_values,
        "memory_mb": float(clean.memory_usage(deep=True).sum() / 1024**2),
    }
    clean.attrs["quality"] = quality
    if len(clean) < 150:
        raise RuntimeError("Hay muy pocos datos válidos para entrenar y evaluar el modelo.")
    return clean


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    change = series.diff()
    gains = change.clip(lower=0).rolling(window).mean()
    losses = -change.clip(upper=0).rolling(window).mean()
    relative_strength = gains / losses.replace(0, np.nan)
    values = 100 - (100 / (1 + relative_strength))
    values = values.mask(losses.eq(0) & gains.gt(0), 100)
    return values.fillna(50)


def build_features(prices: pd.DataFrame, fred: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create features known at each close; FRED values are as-of merged by availability date."""
    frame = prices.copy().sort_values("Date").reset_index(drop=True)
    close = pd.to_numeric(frame["Close"], errors="coerce")
    frame["return_1d"] = close.pct_change(fill_method=None)
    frame["rsi_14"] = rsi(close)
    frame["close_sma5"] = close / close.rolling(5).mean() - 1
    frame["close_sma20"] = close / close.rolling(20).mean() - 1
    frame["volatility_20"] = frame["return_1d"].rolling(20).std()
    frame["volume_change"] = pd.to_numeric(frame["Volume"], errors="coerce").pct_change(fill_method=None)
    frame["volume_change"] = frame["volume_change"].replace([np.inf, -np.inf], np.nan).clip(-5, 5)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    frame["macd_pct"] = macd_line / close
    frame["macd_signal_pct"] = macd_signal / close
    frame["macd_hist_pct"] = (macd_line - macd_signal) / close
    if fred is not None and not fred.empty:
        base = frame.sort_values("Date")
        for series in FRED_FEATURES:
            subset = fred.loc[fred["series_id"] == series, ["available_date", "value"]].copy()
            if subset.empty:
                continue
            subset["available_date"] = pd.to_datetime(subset["available_date"]).dt.tz_localize(None)
            subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
            subset = subset.dropna().sort_values("available_date").drop_duplicates("available_date", keep="last")
            subset = subset.rename(columns={"available_date": "Date", "value": series})
            base = pd.merge_asof(base.sort_values("Date"), subset, on="Date", direction="backward")
        frame = base.sort_values("Date").reset_index(drop=True)
    frame["next_return"] = pd.to_numeric(frame["Close"], errors="coerce").shift(-1) / pd.to_numeric(frame["Close"], errors="coerce") - 1
    # Preserve a missing label on the last row: its next-session result is unknown.
    frame[LABEL] = np.where(frame["next_return"].notna(), (frame["next_return"] > 0).astype(float), np.nan)
    return frame.replace([np.inf, -np.inf], np.nan)


def _model(name: str):
    if name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("XGBoost no está instalado; instala requirements.txt.") from exc
        return XGBClassifier(n_estimators=250, max_depth=3, learning_rate=.04, subsample=.85,
                             colsample_bytree=.85, eval_metric="logloss", random_state=42, n_jobs=2)
    return RandomForestClassifier(n_estimators=350, max_depth=7, min_samples_leaf=4,
                                  class_weight="balanced", random_state=42, n_jobs=2)


def _scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _walk_return(test: pd.DataFrame, probs: np.ndarray, entry: float = .60, exit_: float = .40,
                 max_sessions: int = 20, commission: float = .001):
    if not 1 <= int(max_sessions) <= 20:
        raise ValueError("La permanencia por posición no puede superar 20 sesiones.")
    if not 0 <= commission <= .05:
        raise ValueError("La comisión debe estar entre 0 % y 5 % por cambio.")
    result = test[["Date", "Close", "next_return", LABEL]].copy().reset_index(drop=True)
    result["probability_up"] = probs
    result["prediction_up"] = (result["probability_up"] >= .50).astype(int)
    position, held, prior_position = 0, 0, 0
    positions, ages, returns, changes = [], [], [], []
    for _, row in result.iterrows():
        p = float(row["probability_up"])
        if position == 0 and p >= entry:
            position, held = 1, 0
        elif position == 1 and (p <= exit_ or held >= max_sessions):
            position, held = 0, 0
        turnover = abs(position - prior_position)
        positions.append(position)
        ages.append(held if position else 0)
        changes.append(turnover)
        returns.append((float(row["next_return"]) if position else 0.0) - turnover * commission)
        if position:
            held += 1
        prior_position = position
    result["position"] = positions
    result["holding_sessions"] = ages
    result["turnover"] = changes
    result["strategy_return_net"] = returns
    result["equity_strategy"] = (1 + result["strategy_return_net"]).cumprod()
    result["equity_soxs_reference"] = (1 + result["next_return"]).cumprod()
    for equity, col in [("equity_strategy", "drawdown_strategy"), ("equity_soxs_reference", "drawdown_reference")]:
        result[col] = result[equity] / result[equity].cummax() - 1
    return result


def analyze(prices: pd.DataFrame, fred: pd.DataFrame | None = None, test_fraction: float = .20,
            max_sessions: int = 20, commission: float = .001) -> tuple[Analysis, pd.DataFrame, dict[str, float]]:
    if not .10 <= test_fraction <= .40:
        raise ValueError("La proporción de prueba debe estar entre 10 % y 40 %.")
    if not 1 <= int(max_sessions) <= 20:
        raise ValueError("La permanencia por posición no puede superar 20 sesiones.")
    featured = build_features(prices, fred)
    feature_cols = BASE_FEATURES + [c for c in FRED_FEATURES if c in featured.columns and featured[c].notna().any()]
    valid = featured.dropna(subset=feature_cols + [LABEL, "next_return"]).reset_index(drop=True)
    latest = featured.dropna(subset=feature_cols).iloc[[-1]]
    if len(valid) < 100:
        raise RuntimeError("No hay suficientes filas completas para una prueba cronológica.")
    split = int(len(valid) * (1 - test_fraction))
    train, test = valid.iloc[:split], valid.iloc[split:]
    if train[LABEL].nunique() < 2:
        raise RuntimeError("El entrenamiento necesita ejemplos de ambas clases.")

    comparisons, fitted = [], {}
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train[feature_cols], train[LABEL].astype(int))
    base_pred = baseline.predict(test[feature_cols])
    comparisons.append({"Modelo": "Línea base: clase mayoritaria", **_scores(test[LABEL].astype(int), base_pred)})
    for name in ["Random Forest", "XGBoost"]:
        model = _model(name)
        model.fit(train[feature_cols], train[LABEL].astype(int))
        pred = model.predict(test[feature_cols]).astype(int)
        prob = model.predict_proba(test[feature_cols])[:, 1]
        score = _scores(test[LABEL].astype(int), pred)
        comparisons.append({"Modelo": name, **score})
        fitted[name] = {"model": model, "probabilities": prob, "metrics": score}
    best_name = max(fitted, key=lambda name: (fitted[name]["metrics"]["f1"], name == "Random Forest"))
    chosen = fitted[best_name]
    holdout = _walk_return(test, chosen["probabilities"], max_sessions=max_sessions, commission=commission)

    final_model = _model(best_name)
    final_model.fit(valid[feature_cols], valid[LABEL].astype(int))
    p_up = float(final_model.predict_proba(latest[feature_cols])[:, 1][0])
    signal = "REVISAR ENTRADA" if p_up >= .60 else "REVISAR SALIDA" if p_up <= .40 else "ESPERAR"
    metrics = {**chosen["metrics"], "test_rows": int(len(test)), "train_rows": int(len(train))}
    comparison = pd.DataFrame(comparisons)
    for key in ("accuracy", "precision", "recall", "f1"):
        comparison[key] = comparison[key].astype(float)
    quality = dict(prices.attrs.get("quality", {}))
    quality.update({"feature_rows": int(len(featured)), "model_rows": int(len(valid)),
                    "feature_rows_dropped_pct": float((len(featured) - len(valid)) / max(len(featured), 1) * 100),
                    "model_memory_mb": float(valid.memory_usage(deep=True).sum() / 1024**2)})
    analysis = Analysis(prices, featured, holdout, comparison, metrics, best_name, p_up, signal, feature_cols, quality, final_model)
    stats = {
        "strategy_return": float(holdout["equity_strategy"].iloc[-1] - 1),
        "reference_return": float(holdout["equity_soxs_reference"].iloc[-1] - 1),
        "strategy_drawdown": float(holdout["drawdown_strategy"].min()),
        "reference_drawdown": float(holdout["drawdown_reference"].min()),
        "position_changes": int(holdout["turnover"].sum()),
        "sessions_in_market": int(holdout["position"].sum()),
        "max_sessions": int(max_sessions),
    }
    return analysis, holdout, stats
