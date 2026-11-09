from __future__ import annotations

import re
from datetime import date, timedelta
from xml.etree import ElementTree

import pandas as pd
import requests
from bs4 import BeautifulSoup

FRED_SERIES = {
    "DFF": "Tasa efectiva de fondos federales",
    "DGS10": "Rendimiento Treasury a 10 años",
    "VIXCLS": "Índice VIX",
    "UNRATE": "Tasa de desempleo",
}
SEC_COMPANIES = {"NVIDIA": "0001045810", "AMD": "0000002488", "Intel": "0000050863"}


def fetch_fred(api_key: str, observation_start: date | None = None) -> pd.DataFrame:
    """Fetch first-release FRED observations; realtime_start is used as availability date."""
    if not api_key:
        return pd.DataFrame(columns=["series_id", "date", "available_date", "value", "description"])
    api_key = api_key.strip()
    if not re.fullmatch(r"[a-z0-9]{32}", api_key):
        raise ValueError("La clave FRED debe tener 32 caracteres alfanuméricos en minúscula. Revisa que no incluya espacios ni comillas.")
    frames = []
    start_date = observation_start or (date.today() - timedelta(days=365 * 5))
    end_date = date.today()
    for series_id, description in FRED_SERIES.items():
        # FRED output_type=4 returns first-release values, but one request may
        # include at most 2,000 vintage dates. Query disjoint 2-year observation
        # windows and allow 90 days for releases to become available.
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=730), end_date)
            params = {
                "series_id": series_id, "api_key": api_key, "file_type": "json",
                "output_type": 4,
                "realtime_start": chunk_start.isoformat(),
                # FRED rejects future real-time dates. Keep the release-lag
                # buffer for old windows, but cap the final vintage at today.
                "realtime_end": min(chunk_end + timedelta(days=90), end_date).isoformat(),
                "observation_start": chunk_start.isoformat(),
                "observation_end": chunk_end.isoformat(),
                "sort_order": "asc", "limit": 100000,
            }
            try:
                response = requests.get("https://api.stlouisfed.org/fred/series/observations", params=params, timeout=30)
            except requests.RequestException as exc:
                raise RuntimeError(f"No se pudo conectar con FRED ({type(exc).__name__}). Revisa tu conexión e inténtalo otra vez.") from None
            if not response.ok:
                message = "FRED rechazó la consulta. Verifica que la clave esté activa y que el identificador de serie sea válido."
                try:
                    body = response.json()
                    detail = body.get("error_message") or body.get("message")
                    if detail:
                        message = str(detail)
                except (ValueError, AttributeError):
                    try:
                        root = ElementTree.fromstring(response.text)
                        detail = root.attrib.get("message")
                        if detail:
                            message = detail
                    except ElementTree.ParseError:
                        pass
                # Never surface the request URL: FRED places api_key in its query string.
                message = message.replace(api_key, "[clave oculta]")
                if "api_key" in message.lower() or "not registered" in message.lower():
                    message = "FRED rechazó la clave API. Genera una clave activa en tu cuenta FRED y reemplaza la anterior."
                if "no vintage dates exist" in message.lower():
                    # A narrow recent window can contain no release (weekend,
                    # holiday, or publication delay). Keep the other periods.
                    chunk_start = chunk_end + timedelta(days=1)
                    continue
                raise RuntimeError(f"FRED respondió HTTP {response.status_code}: {message}")
            observations = response.json().get("observations", [])
            frame = pd.DataFrame(observations)
            if not frame.empty:
                frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
                frame["date"] = pd.to_datetime(frame["date"])
                frame["available_date"] = pd.to_datetime(frame["realtime_start"])
                frame = frame.dropna(subset=["value", "date", "available_date"])
                frame["series_id"] = series_id
                frame["description"] = description
                frames.append(frame[["series_id", "date", "available_date", "value", "description"]])
            chunk_start = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["series_id", "date", "available_date", "value", "description"])
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["series_id", "date"], keep="first").sort_values(["series_id", "date"]).reset_index(drop=True)


def recent_sec_filings(company: str, user_agent: str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 8):
    if company not in SEC_COMPANIES:
        raise ValueError("Empresa SEC no reconocida.")
    if not user_agent or "@" not in user_agent:
        raise ValueError("Configura SEC_USER_AGENT con nombre del proyecto y un correo de contacto válido.")
    cik = SEC_COMPANIES[company]
    headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
    response = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=headers, timeout=25)
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    output = []
    for index, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        accession = recent["accessionNumber"][index]
        document = recent["primaryDocument"][index]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{document}"
        output.append({"company": company, "form": form, "date": recent["filingDate"][index],
                       "accession": accession, "document": document, "url": url})
        if len(output) >= limit:
            break
    return output


SEC_TERMS = ["risk", "demand", "competition", "supply", "semiconductor", "revenue", "artificial intelligence"]


def read_sec_document(url: str, user_agent: str) -> dict:
    if not user_agent or "@" not in user_agent:
        raise ValueError("Configura SEC_USER_AGENT con nombre del proyecto y un correo de contacto válido.")
    response = requests.get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for node in soup(["script", "style", "table"]):
        node.decompose()
    text = re.sub(r"\s+", " ", " ".join(soup.stripped_strings)).strip()
    counts = [{"term": term, "frequency": len(re.findall(rf"\b{re.escape(term)}\b", text, flags=re.I))} for term in SEC_TERMS]
    excerpt = text[:4500]
    return {"characters_extracted": len(text), "excerpt": excerpt, "terms": counts}


POSITIVE_TERMS = {"rise", "rises", "gain", "gains", "growth", "strong", "beat", "beats", "surge", "record", "positive", "upgrade", "bullish", "sube", "crece", "fuerte", "supera", "aumenta", "positivo"}
NEGATIVE_TERMS = {"fall", "falls", "loss", "losses", "weak", "miss", "misses", "drop", "decline", "negative", "downgrade", "risk", "bearish", "baja", "cae", "débil", "perdida", "riesgo", "negativo", "reduce"}


def score_manual_texts(texts: list[str]) -> pd.DataFrame:
    rows = []
    for text in texts:
        tokens = re.findall(r"[\wáéíóúñ]+", text.lower())
        pos = sum(token in POSITIVE_TERMS for token in tokens)
        neg = sum(token in NEGATIVE_TERMS for token in tokens)
        raw = pos - neg
        sentiment = "POSITIVO" if raw > 0 else "NEGATIVO" if raw < 0 else "NEUTRAL"
        score = raw / max(pos + neg, 1)
        rows.append({"Texto": text, "Sentimiento": sentiment, "Score léxico": score, "Positivas": pos, "Negativas": neg})
    return pd.DataFrame(rows)
