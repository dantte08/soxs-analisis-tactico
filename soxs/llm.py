"""LLM-assisted academic recommendation via the OpenAI Responses API.

The module is opt-in: callers invoke it only after an explicit UI action.
It never places broker orders and never receives broker credentials.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests


RESPONSES_URL = "https://api.openai.com/v1/responses"
MODEL_DEFAULT = "gpt-4o-mini"
NUMBER_PATTERN = re.compile(r"(?<![\w])[+-]?\d+(?:[.,]\d+)?%?(?![\w])")
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendation": {
            "type": "string",
            "enum": ["ESPERAR", "REVISAR ENTRADA", "REVISAR SALIDA"],
        },
        "rationale": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["baja", "media", "alta"]},
        "limitations": {"type": "string"},
    },
    "required": ["recommendation", "rationale", "risks", "confidence", "limitations"],
    "additionalProperties": False,
}


class LLMError(RuntimeError):
    """Safe, user-displayable LLM integration error."""


def _known_numeric_values(value) -> set[float]:
    """Collect supplied numeric facts and common percentage/rounding forms."""
    found: set[float] = set()
    if isinstance(value, bool) or value is None:
        return found
    if isinstance(value, (int, float)):
        base = float(value)
        if base == base and abs(base) != float("inf"):
            for scaled in (base, base * 100):
                for decimals in range(5):
                    found.add(round(scaled, decimals))
        return found
    if isinstance(value, dict):
        for item in value.values():
            found.update(_known_numeric_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_known_numeric_values(item))
    return found


def _numeric_claims_are_supported(texts: list[str], data: dict) -> bool:
    allowed = _known_numeric_values(data)
    for text in texts:
        for token in NUMBER_PATTERN.findall(str(text)):
            try:
                number = float(token.rstrip("%").replace(",", "."))
            except ValueError:
                return False
            if not any(abs(number - known) <= 1e-6 for known in allowed):
                return False
    return True


def recommend(data: dict, api_key: str | None = None, model: str | None = None,
              timeout: int = 45) -> dict:
    """Request one structured recommendation from the configured LLM."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise LLMError("Falta OPENAI_API_KEY. Configúrala como variable de entorno local.")
    selected_model = model or os.getenv("OPENAI_MODEL", MODEL_DEFAULT)
    instructions = (
        "Eres un asistente de análisis académico para un proyecto sobre SOXS. "
        "Usa solo los datos JSON proporcionados; no inventes cifras ni hechos. "
        "Evita incluir dígitos en el texto libre; si mencionas una cifra, debe coincidir con un dato proporcionado. "
        "Resume incertidumbre, volatilidad, riesgo de pérdida y limitaciones de los datos. "
        "Devuelve una orientación de estudio de corto plazo, no asesoría financiera personalizada. "
        "No ordenes compras o ventas reales, no prometas rendimiento y no indiques una cantidad. "
        "Si los indicadores son contradictorios o insuficientes, elige ESPERAR. "
        "La salida solo puede proponer ESPERAR, REVISAR ENTRADA o REVISAR SALIDA; "
        "revisar no equivale a ejecutar una operación."
    )
    body = {
        "model": selected_model,
        "instructions": instructions,
        "input": json.dumps(data, ensure_ascii=False, allow_nan=False),
        "max_output_tokens": 300,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "soxs_academic_recommendation",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    }
    try:
        response = requests.post(
            RESPONSES_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError(f"No se pudo conectar con el servicio LLM: {exc}") from exc
    if not response.ok:
        # Avoid reflecting request headers or secrets to the UI/log.
        try:
            detail = response.json().get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            detail = ""
        raise LLMError(f"La API LLM respondió HTTP {response.status_code}. {detail}".strip())
    try:
        payload = response.json()
        output_text = payload.get("output_text")
        if not output_text:
            output_text = next(
                part["text"]
                for item in payload.get("output", [])
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
        result = json.loads(output_text)
    except (ValueError, KeyError, StopIteration, TypeError) as exc:
        raise LLMError("La respuesta del LLM no tenía el formato estructurado esperado.") from exc
    if result.get("recommendation") not in {"ESPERAR", "REVISAR ENTRADA", "REVISAR SALIDA"}:
        raise LLMError("La respuesta del LLM no pasó la validación de recomendación.")
    # Allow figures only when they match a value supplied to the model
    # (including common percentage and rounding forms).
    prose = [result.get("rationale", ""), result.get("limitations", ""), *result.get("risks", [])]
    if not _numeric_claims_are_supported(prose, data):
        raise LLMError("La respuesta incluyó una cifra que no coincide con los datos enviados; vuelve a intentarlo.")
    usage = payload.get("usage", {})
    result["usage"] = {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "model": selected_model,
    }
    return result


def log_recommendation(result: dict, data: dict,
                       path: str | Path = "data/llm_recommendations.jsonl") -> None:
    """Append a redacted, auditable record (no API key or raw prompt)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "completed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": result.get("usage", {}).get("model"),
        "input_tokens": result.get("usage", {}).get("input_tokens"),
        "output_tokens": result.get("usage", {}).get("output_tokens"),
        "recommendation": result.get("recommendation"),
        "confidence": result.get("confidence"),
        "rationale": result.get("rationale"),
        "risks": result.get("risks"),
        "limitations": result.get("limitations"),
        "market_date": data.get("market_date"),
        "ml_signal": data.get("ml_signal"),
        "ml_probability_up": data.get("ml_probability_up"),
    }
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def log_request_attempt(data: dict, path: str | Path = "data/llm_recommendations.jsonl") -> None:
    """Record an API attempt before sending it so failures count toward the cap."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": "request_started",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "market_date": data.get("market_date"),
        "ml_signal": data.get("ml_signal"),
        "ml_probability_up": data.get("ml_probability_up"),
    }
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def daily_request_count(path: str | Path = "data/llm_recommendations.jsonl") -> int:
    """Count this UTC day's completed API calls for a modest UI safeguard."""
    target = Path(path)
    if not target.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    with target.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
                if row.get("event") == "request_started" and row.get("timestamp_utc", "").startswith(today):
                    count += 1
            except json.JSONDecodeError:
                continue
    return count
