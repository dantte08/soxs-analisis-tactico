from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests


class AlpacaPaper:
    """Alpaca PAPER-only client. The live endpoint is intentionally absent."""
    BASE_URL = "https://paper-api.alpaca.markets"
    ACTIVE_STATUSES = {"new", "accepted", "pending_new", "partially_filled", "pending_cancel"}

    def __init__(self, api_key: str, secret_key: str):
        if not api_key.strip() or not secret_key.strip():
            raise ValueError("Completa ambas credenciales de Alpaca Paper.")
        self.headers = {"APCA-API-KEY-ID": api_key.strip(), "APCA-API-SECRET-KEY": secret_key.strip(),
                        "Accept": "application/json", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs):
        response = requests.request(method, self.BASE_URL + path, headers=self.headers, timeout=20, **kwargs)
        if not response.ok:
            try:
                message = response.json().get("message", response.text)
            except ValueError:
                message = response.text
            raise RuntimeError(f"Alpaca Paper respondió {response.status_code}: {message}")
        return response.json() if response.text else None

    def account(self):
        return self._request("GET", "/v2/account")

    def positions(self):
        return self._request("GET", "/v2/positions") or []

    def orders(self, limit: int = 50):
        return self._request("GET", "/v2/orders", params={"status": "all", "limit": limit, "direction": "desc"}) or []

    def submit_soxs_order(self, side: str, qty: int = 1):
        if side not in {"buy", "sell"}:
            raise ValueError("Operación no permitida.")
        if not 1 <= int(qty) <= 100000:
            raise ValueError("Cantidad inválida.")
        payload = {"symbol": "SOXS", "qty": str(int(qty)), "side": side, "type": "market",
                   "time_in_force": "day", "client_order_id": "soxs-acad-" + uuid.uuid4().hex[:16]}
        return self._request("POST", "/v2/orders", json=payload)


def _load_state(path: str | Path) -> dict:
    file = Path(path)
    if not file.exists():
        return {}
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: str | Path, state: dict) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _entry_date_from_orders(orders: list[dict]) -> date | None:
    for order in orders:
        if order.get("symbol") == "SOXS" and order.get("side") == "buy" and order.get("status") == "filled":
            stamp = order.get("filled_at") or order.get("submitted_at")
            if stamp:
                return pd.Timestamp(stamp).date()
    return None


def sessions_since(entry_date: date, market_dates: pd.Series) -> int:
    dates = pd.to_datetime(market_dates).dt.date
    return int(((dates > entry_date) & (dates <= date.today())).sum())


def auto_paper_cycle(broker: AlpacaPaper, probability_up: float, market_dates: pd.Series,
                     max_sessions: int = 20, state_path: str | Path = "data/paper_bot_state.json",
                     manual_entry_date: date | None = None) -> dict:
    """One guarded PAPER-only cycle: at most one share per entry, full exit at signal/age limit."""
    if not 1 <= int(max_sessions) <= 20:
        raise ValueError("La permanencia máxima no puede superar 20 sesiones.")
    positions = broker.positions()
    soxs = next((p for p in positions if p.get("symbol") == "SOXS"), None)
    orders = broker.orders()
    pending = [o for o in orders if o.get("symbol") == "SOXS" and o.get("status") in broker.ACTIVE_STATUSES]
    state = _load_state(state_path)
    if pending:
        return {"decision": "SIN ACCIÓN", "reason": "Existe una orden SOXS pendiente; se evita duplicarla."}

    if soxs is None:
        if probability_up >= .60:
            today = date.today().isoformat()
            same_day_attempt = next((o for o in orders if o.get("symbol") == "SOXS" and o.get("side") == "buy"
                                     and str(o.get("submitted_at", "")).startswith(today)), None)
            if same_day_attempt:
                return {"decision": "SIN ACCIÓN", "reason": "Ya hubo un intento de entrada PAPER hoy; se evita repetirlo en ciclos posteriores."}
            order = broker.submit_soxs_order("buy", 1)
            state["entry_date"] = date.today().isoformat()
            state["last_order_id"] = order.get("id")
            _save_state(state_path, state)
            return {"decision": "COMPRA PAPER", "reason": "La probabilidad superó 0.60; se envió una participación.", "order_id": order.get("id")}
        return {"decision": "SIN ACCIÓN", "reason": "No hay señal de entrada suficiente."}

    if str(soxs.get("side", "long")).lower() != "long":
        return {"decision": "REVISIÓN MANUAL", "reason": "La posición existente no es larga; el bot no intentará modificarla."}

    entry_date = manual_entry_date
    if entry_date is None and state.get("entry_date"):
        entry_date = pd.to_datetime(state["entry_date"]).date()
    if entry_date is None:
        entry_date = _entry_date_from_orders(orders)
    if entry_date is None:
        return {"decision": "REVISIÓN MANUAL", "reason": "No se pudo verificar la fecha de entrada; no se automatiza el cierre por antigüedad."}

    age = sessions_since(entry_date, market_dates)
    try:
        qty = int(float(soxs.get("qty", 0)))
    except (TypeError, ValueError):
        qty = 0
    if probability_up <= .40 or age >= max_sessions:
        if qty <= 0:
            return {"decision": "REVISIÓN MANUAL", "reason": "La cantidad de SOXS no pudo verificarse."}
        order = broker.submit_soxs_order("sell", qty)
        _save_state(state_path, {})
        why = "Se alcanzó el máximo de sesiones; se solicitó cerrar toda la posición." if age >= max_sessions else "Se activó la señal de salida; se solicitó cerrar toda la posición."
        return {"decision": "VENTA PAPER", "reason": why, "age_sessions": age, "order_id": order.get("id")}
    if age >= max_sessions - 1:
        return {"decision": "VIGILANCIA PAPER", "reason": "La posición está por alcanzar el máximo de permanencia.", "age_sessions": age}
    return {"decision": "MANTENER PAPER", "reason": "La posición sigue dentro del límite y no se activó salida.", "age_sessions": age}
