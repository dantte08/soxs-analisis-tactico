from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ALERT_COLUMNS = ["timestamp_utc", "event_key", "decision", "signal", "probability_up", "reason", "order_id", "telegram_status"]


def record_alert(decision: str, signal: str, probability_up: float, reason: str,
                 order_id: str | None = None, path: str | Path = "data/alerts.csv") -> tuple[bool, str]:
    """Append a local alert only when its event key differs from the last recorded event."""
    target = Path(path)
    event_key = str(order_id or f"{decision}|{signal}")
    previous_key = None
    if target.exists():
        try:
            existing = pd.read_csv(target)
            if not existing.empty and "event_key" in existing.columns:
                previous_key = str(existing.iloc[-1]["event_key"])
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            previous_key = None
    if event_key == previous_key:
        return False, event_key
    target.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event_key": event_key,
        "decision": decision,
        "signal": signal,
        "probability_up": float(probability_up),
        "reason": reason,
        "order_id": order_id or "",
        "telegram_status": "pendiente",
    }])
    row.to_csv(target, mode="a", header=not target.exists(), index=False)
    return True, event_key


def set_telegram_status(event_key: str, status: str, path: str | Path = "data/alerts.csv") -> None:
    target = Path(path)
    if not target.exists():
        return
    try:
        history = pd.read_csv(target)
        if not history.empty:
            history.loc[history["event_key"].astype(str) == str(event_key), "telegram_status"] = status
            history.to_csv(target, index=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, KeyError):
        return


def send_telegram_alert(token: str, chat_id: str, message: str, timeout: int = 10) -> None:
    """Send an optional Telegram notification; tokens are never written to local logs."""
    if not token or not chat_id:
        raise ValueError("Configura TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID para enviar la alerta.")
    response = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message[:3500], "disable_web_page_preview": True},
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"Telegram respondió HTTP {response.status_code}.")
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError("Telegram no confirmó el envío de la alerta.")
