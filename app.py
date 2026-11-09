from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from soxs.alerts import record_alert, send_telegram_alert, set_telegram_status
from soxs.core import FRED_FEATURES, analyze, download_prices
from soxs.llm import LLMError, daily_request_count, log_recommendation, log_request_attempt, recommend
from soxs.paper_broker import AlpacaPaper, auto_paper_cycle
from soxs.sources import FRED_SERIES, fetch_fred, read_sec_document, recent_sec_filings, score_manual_texts

load_dotenv()
st.set_page_config(page_title="SHORT – Inversiones SOXS", page_icon="📉", layout="wide")


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


st.title("SHORT – Inversiones SOXS")
st.caption("Equipo 4 · Analítica descriptiva, predictiva y prescriptiva · aplicación académica")
st.info(
    "SOXS es un ETF inverso apalancado cuyo objetivo es diario. Este sistema estima la dirección de la siguiente sesión "
    "y limita cada posición simulada a un máximo de 20 sesiones bursátiles. El objetivo diario de SOXS no se proyecta "
    "como un múltiplo constante para todo el mes."
)


@st.cache_data(ttl=300, show_spinner=False)
def cached_prices(period: str) -> pd.DataFrame:
    return download_prices(period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fred(api_key: str, start_iso: str) -> pd.DataFrame:
    return fetch_fred(api_key, pd.Timestamp(start_iso).date())


for state_key, initial in {"sentiment_rows": None, "sec_document": None, "sec_filings": None,
                           "llm_result": None, "bot_result": None}.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = initial

with st.sidebar:
    st.header("Configuración")
    period = st.selectbox("Historial para entrenar y evaluar", ["2y", "5y", "10y"], index=1)
    max_sessions = st.slider("Máximo de permanencia por posición (sesiones)", min_value=1, max_value=20, value=20)
    commission_pct = st.number_input("Comisión simulada por cambio de posición (%)", 0.0, 2.0, 0.10, 0.05)
    st.caption("La predicción es para la siguiente sesión. 20 sesiones bursátiles ≈ un mes.")
    st.divider()
    st.subheader("Conexiones opcionales")
    fred_key = st.text_input("FRED_API_KEY", value=get_setting("FRED_API_KEY"), type="password")
    alpaca_key = st.text_input("ALPACA_PAPER_KEY", value=get_setting("ALPACA_PAPER_KEY"), type="password")
    alpaca_secret = st.text_input("ALPACA_PAPER_SECRET", value=get_setting("ALPACA_PAPER_SECRET"), type="password")
    telegram_token = st.text_input("TELEGRAM_BOT_TOKEN (opcional)", value=get_setting("TELEGRAM_BOT_TOKEN"), type="password")
    telegram_chat = st.text_input("TELEGRAM_CHAT_ID (opcional)", value=get_setting("TELEGRAM_CHAT_ID"), type="password")
    st.caption("Las claves no se guardan en el proyecto. Usa solo credenciales PAPER para Alpaca.")
    if st.button("Actualizar precios y recalcular", use_container_width=True):
        cached_prices.clear()
        cached_fred.clear()

load_started = time.perf_counter()
try:
    with st.spinner("Cargando SOXS y entrenando los modelos con partición temporal..."):
        prices = cached_prices(period)
        fred = pd.DataFrame()
        fred_error = None
        if fred_key:
            try:
                start = (pd.to_datetime(prices["Date"].min()) - pd.Timedelta(days=370)).date().isoformat()
                fred = cached_fred(fred_key, start)
            except Exception as exc:
                fred_error = str(exc)
        analysis, backtest, bt = analyze(prices, fred if not fred.empty else None,
                                         max_sessions=max_sessions, commission=commission_pct / 100)
    dashboard_calc_seconds = time.perf_counter() - load_started
except Exception as exc:
    st.error(f"No se pudo completar el análisis: {exc}")
    st.stop()

if fred_error:
    st.sidebar.warning(f"FRED no se integró en esta ejecución: {fred_error}")
last = analysis.prices.iloc[-1]
last_features = analysis.featured.dropna(subset=analysis.feature_cols).iloc[-1]
market_date = pd.to_datetime(last["Date"]).date()
st.caption(f"Último dato de SOXS: {market_date} · ${float(last['Close']):.2f} · Yahoo Finance · caché de 300 segundos")

tabs = st.tabs(["Reporte descriptivo", "Reporte predictivo", "Backtesting", "Fuentes y ETL",
                "Reporte prescriptivo LLM", "Alpaca Paper", "Monitoreo y entrega"])

with tabs[0]:
    st.subheader("¿Qué pasó? — comportamiento reciente de SOXS")
    a, b, c, d = st.columns(4)
    a.metric("Cierre ajustado", f"${float(last['Close']):.2f}")
    a1 = analysis.featured.iloc[-1]
    b.metric("RSI (14)", f"{float(a1['rsi_14']):.1f}" if pd.notna(a1["rsi_14"]) else "—")
    c.metric("Volatilidad (20 sesiones)", f"{float(a1['volatility_20']):.2%}" if pd.notna(a1["volatility_20"]) else "—")
    d.metric("Filas de mercado válidas", str(analysis.quality.get("rows_valid", len(prices))))

    st.caption(f"MACD normalizado: {float(a1['macd_pct']):.3%} · señal: {float(a1['macd_signal_pct']):.3%} · histograma: {float(a1['macd_hist_pct']):.3%}. Se usa como variable técnica del modelo.")

    view = analysis.featured.tail(180)
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(x=view.Date, y=view.Close, name="SOXS", line={"width": 2}))
    price_fig.add_trace(go.Scatter(x=view.Date, y=view.Close.rolling(20).mean(), name="SMA 20", line={"dash": "dot"}))
    price_fig.update_layout(title="Precio ajustado y promedio móvil de 20 sesiones", yaxis_title="USD", height=350,
                            margin={"l": 10, "r": 10, "t": 50, "b": 10}, hovermode="x unified")
    st.plotly_chart(price_fig, use_container_width=True)
    vol_fig = go.Figure()
    vol_fig.add_trace(go.Bar(x=view.Date, y=view.Volume, name="Volumen"))
    vol_fig.update_layout(title="Volumen diario", height=260, margin={"l": 10, "r": 10, "t": 45, "b": 10})
    st.plotly_chart(vol_fig, use_container_width=True)

    st.markdown("**Indicadores de evaluación fuera de muestra**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Retorno neto estrategia", f"{bt['strategy_return']:.1%}")
    k2.metric("Retorno SOXS referencia", f"{bt['reference_return']:.1%}")
    k3.metric("Drawdown estrategia", f"{bt['strategy_drawdown']:.1%}")
    k4.metric("Cambios de posición", str(bt["position_changes"]))
    st.caption("La referencia SOXS es buy & hold dentro del mismo bloque de prueba, sin límite de tenencia; es solo comparador.")

    with st.expander("Cifras de calidad y las 3V"):
        q = analysis.quality
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Filas descargadas", str(q.get("rows_downloaded", "—")))
        q2.metric("Filas válidas", str(q.get("rows_valid", "—")))
        q3.metric("Descartadas", f"{q.get('discarded_pct', 0):.2f}%")
        q4.metric("Memoria OHLCV", f"{q.get('memory_mb', 0):.3f} MB")
        st.caption(f"Tiempo local de descarga/cálculo: {dashboard_calc_seconds:.2f} s. Registra por separado el tiempo público después del despliegue.")
        st.write(f"Duplicados: {q.get('duplicate_dates', 0)} · valores faltantes: {q.get('missing_market_values', 0)} · cierres no positivos: {q.get('nonpositive_close', 0)} · volumen negativo: {q.get('negative_volume', 0)} · High < Low: {q.get('high_below_low', 0)}")
        st.write("**Velocidad:** precios diarios; caché de 300 s. FRED/SEC se consultan bajo demanda y el texto se pega manualmente.")
        st.write("**Variedad:** OHLCV, indicadores técnicos, series macroeconómicas FRED, documentos SEC y texto manual.")

with tabs[1]:
    st.subheader("¿Qué podría pasar? — clasificación para la próxima sesión")
    st.write("**Objetivo:** 1 si el cierre ajustado de la próxima sesión supera el cierre actual; 0 en caso contrario. La última fecha sin cierre futuro no se etiqueta.")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Orientación del modelo", analysis.signal)
    p2.metric("Probabilidad estimada de alza", f"{analysis.probability_up:.1%}")
    p3.metric("Modelo seleccionado por F1", analysis.model_name)
    p4.metric("F1 fuera de muestra", f"{analysis.metrics['f1']:.1%}")
    st.markdown("**Comparación en el bloque de prueba cronológico**")
    display = analysis.comparison.rename(columns={"Modelo": "Modelo", "accuracy": "Accuracy", "precision": "Precisión", "recall": "Recall", "f1": "F1"})
    st.dataframe(display.style.format({"Accuracy": "{:.1%}", "Precisión": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.caption("La línea base siempre predice la clase mayoritaria observada en entrenamiento. Los modelos se evalúan sin barajar fechas; se selecciona el ML con mayor F1.")
    baseline_f1 = float(analysis.comparison.loc[analysis.comparison["Modelo"] == "Línea base: clase mayoritaria", "f1"].iloc[0])
    if analysis.metrics["f1"] > baseline_f1:
        st.success(f"En esta ejecución, el F1 del modelo elegido ({analysis.metrics['f1']:.1%}) supera la línea base ({baseline_f1:.1%}). Esto no demuestra desempeño futuro.")
    else:
        st.warning(f"En esta ejecución, el modelo no superó en F1 a la línea base ({baseline_f1:.1%}). No se afirma superioridad; revisa features, periodo y clases.")
    st.write("**Variables usadas:** " + ", ".join(analysis.feature_cols))
    confusion = confusion_matrix(analysis.holdout.target_next_day_up.astype(int), analysis.holdout.prediction_up.astype(int), labels=[0, 1]) if "prediction_up" in analysis.holdout else None
    if confusion is not None:
        st.write("Matriz de confusión del modelo seleccionado (filas = real 0/1; columnas = predicho 0/1).")
        st.dataframe(pd.DataFrame(confusion, index=["Real 0", "Real 1"], columns=["Predicho 0", "Predicho 1"]), use_container_width=True)
    st.download_button("Descargar predicciones fuera de muestra (CSV)", analysis.holdout.to_csv(index=False).encode("utf-8"), "soxs_predicciones.csv", "text/csv")

with tabs[2]:
    st.subheader("Backtesting de corto plazo")
    st.write(f"Reglas: revisar entrada si P(alza) ≥ 60 %; revisar salida si P(alza) ≤ 40 %; posición máxima {max_sessions} sesiones; comisión simulada {commission_pct:.2f} % por cambio.")
    st.caption("Las decisiones al cierre de una fecha se aplican al retorno de la sesión siguiente. El límite de 20 sesiones es por posición; el bloque de prueba puede cubrir varios meses.")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Retorno estrategia neto", f"{bt['strategy_return']:.1%}")
    r2.metric("Retorno referencia SOXS", f"{bt['reference_return']:.1%}")
    r3.metric("Caída máxima estrategia", f"{bt['strategy_drawdown']:.1%}")
    r4.metric("Caída máxima referencia", f"{bt['reference_drawdown']:.1%}")
    equity_fig = go.Figure()
    equity_fig.add_trace(go.Scatter(x=backtest.Date, y=backtest.equity_strategy, name="Estrategia táctica"))
    equity_fig.add_trace(go.Scatter(x=backtest.Date, y=backtest.equity_soxs_reference, name="SOXS referencia"))
    equity_fig.update_layout(title="Capital relativo (capital inicial = 1)", height=350, hovermode="x unified")
    st.plotly_chart(equity_fig, use_container_width=True)
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(x=backtest.Date, y=backtest.drawdown_strategy, name="Drawdown estrategia", fill="tozeroy"))
    dd_fig.add_trace(go.Scatter(x=backtest.Date, y=backtest.drawdown_reference, name="Drawdown referencia"))
    dd_fig.update_layout(title="Drawdown por fecha", yaxis_tickformat=".0%", height=300)
    st.plotly_chart(dd_fig, use_container_width=True)
    st.dataframe(backtest[["Date", "probability_up", "position", "holding_sessions", "next_return", "strategy_return_net", "equity_strategy"]].tail(40), use_container_width=True, hide_index=True)
    st.download_button("Descargar backtesting (CSV)", backtest.to_csv(index=False).encode("utf-8"), "soxs_backtesting.csv", "text/csv")

with tabs[3]:
    st.subheader("Fuentes, integración y ETL")
    st.dataframe(pd.DataFrame([
        {"Fuente": "Yahoo Finance / yfinance", "Tipo": "Estructurado", "Campos": "OHLCV ajustado SOXS", "Frecuencia": "Diaria; caché 300 s", "Uso": "Modelo, reporte y backtest"},
        {"Fuente": "FRED", "Tipo": "Estructurado", "Campos": "DFF, DGS10, VIXCLS, UNRATE", "Frecuencia": "Según publicación", "Uso": "Features opcionales; merge por disponibilidad"},
        {"Fuente": "SEC EDGAR", "Tipo": "Documental", "Campos": "10-K/10-Q de NVIDIA, AMD, Intel", "Frecuencia": "Bajo demanda", "Uso": "Contexto documental"},
        {"Fuente": "Texto del usuario", "Tipo": "No estructurado", "Campos": "Titulares/notas manuales", "Frecuencia": "Manual", "Uso": "Léxico demostrativo"},
        {"Fuente": "Alpaca Paper", "Tipo": "Simulación", "Campos": "Cuenta, posiciones y órdenes", "Frecuencia": "Manual / ciclo 5 min", "Uso": "Operaciones simuladas"},
    ]), use_container_width=True, hide_index=True)
    st.markdown("**Yahoo Finance:** OHLCV ajustado de SOXS, selector 2/5/10 años y caché de 300 segundos.")
    st.markdown("**FRED opcional:** DFF, DGS10, VIXCLS y UNRATE. El módulo solicita valores de primera publicación y alinea por `realtime_start` con `merge_asof` hacia atrás. Las series se incluyen como variables cuando la clave funciona.")
    if not fred.empty:
        st.success(f"FRED integrado: {fred['series_id'].nunique()} series, {len(fred):,} observaciones descargadas.")
        st.dataframe(fred.sort_values("available_date", ascending=False).head(40), use_container_width=True, hide_index=True)
    elif fred_key:
        st.warning("No hay observaciones FRED disponibles; el modelo continúa sin esas variables.")
    else:
        st.info("FRED es opcional. Configura FRED_API_KEY para habilitar sus series.")

    st.markdown("**SEC EDGAR bajo demanda:** documentos 10-K/10-Q de NVIDIA, AMD e Intel. Configura `SEC_USER_AGENT` con nombre del proyecto y correo de contacto; el documento se usa para análisis descriptivo, no como feature histórico entrenado.")
    sec_agent = st.text_input("SEC_USER_AGENT", value=os.getenv("SEC_USER_AGENT", ""), help="Ejemplo de formato: Equipo4 SOXS contacto@universidad.edu")
    company = st.selectbox("Empresa", ["NVIDIA", "AMD", "Intel"])
    if st.button("Consultar formularios SEC"):
        try:
            st.session_state.sec_filings = recent_sec_filings(company, sec_agent)
        except Exception as exc:
            st.error(f"No se pudo consultar SEC EDGAR: {exc}")
    filings = st.session_state.sec_filings or []
    if filings:
        labels = [f"{x['form']} · {x['date']} · {x['accession']}" for x in filings]
        picked = st.selectbox("Formulario reciente", range(len(labels)), format_func=lambda i: labels[i])
        if st.button("Leer y resumir documento seleccionado"):
            try:
                st.session_state.sec_document = read_sec_document(filings[picked]["url"], sec_agent)
            except Exception as exc:
                st.error(f"No se pudo leer el documento SEC: {exc}")
    if st.session_state.sec_document:
        doc = st.session_state.sec_document
        st.write(f"Caracteres extraídos: {doc['characters_extracted']:,}")
        st.dataframe(pd.DataFrame(doc["terms"]), use_container_width=True, hide_index=True)
        st.text_area("Extracto del documento", doc["excerpt"], height=180)

    st.markdown("**Texto no estructurado manual:** pega titulares o notas; se analiza con un léxico demostrativo en español/inglés. No se descargan noticias sociales y estos puntajes no se presentan como variables históricas.")
    raw_texts = st.text_area("Un titular o texto por línea", height=120, key="manual_news")
    if st.button("Analizar sentimiento del texto"):
        texts = [line.strip() for line in raw_texts.splitlines() if line.strip()]
        st.session_state.sentiment_rows = score_manual_texts(texts)
    if st.session_state.sentiment_rows is not None and not st.session_state.sentiment_rows.empty:
        st.dataframe(st.session_state.sentiment_rows, use_container_width=True, hide_index=True)

    st.markdown("**Validaciones ETL aplicadas:** fecha única, columnas numéricas, cierre positivo, volumen no negativo, High ≥ Low, eliminación de nulos, infinitos y objetivo futuro desconocido. No se winsorizan atípicos; se informa como limitación.")
    st.write(f"OHLCV: {analysis.quality.get('rows_downloaded', 0)} filas recibidas → {analysis.quality.get('rows_valid', 0)} válidas → {analysis.quality.get('rows_discarded', 0)} descartadas.")

with tabs[4]:
    st.subheader("¿Qué revisar? — reporte prescriptivo LLM")
    st.warning("Orientación académica, no asesoría financiera. El LLM no recibe claves de Alpaca ni puede enviar órdenes.")
    llm_key = get_setting("OPENAI_API_KEY")
    llm_model = get_setting("OPENAI_MODEL", "gpt-4o-mini")
    calls = daily_request_count()
    x1, x2, x3 = st.columns(3)
    x1.metric("API", "Configurada" if llm_key else "Falta OPENAI_API_KEY")
    x2.metric("Modelo", llm_model)
    x3.metric("Intentos hoy UTC", str(calls))
    st.write("El LLM recibe fecha/precio, señal y probabilidad ML, métricas de prueba, retorno/drawdown del backtest y límite de 20 sesiones.")
    st.caption("El costo por 100 informes se calcula con tokens medios reales de entrada/salida y la tarifa vigente del modelo; no se presenta un costo monetario sin medir ambos datos.")
    llm_input = {
            "instrument": "SOXS", "market_date": str(market_date), "price_usd": float(last.Close),
            "model": analysis.model_name, "ml_signal": analysis.signal,
            "ml_probability_up": float(analysis.probability_up),
            "validation_metrics": {k: float(analysis.metrics[k]) for k in ("accuracy", "precision", "recall", "f1")},
            "backtest": {k: float(bt[k]) for k in ("strategy_return", "reference_return", "strategy_drawdown", "reference_drawdown")},
            "max_holding_sessions": int(max_sessions), "commission_rate_per_change": float(commission_pct / 100),
            "macro_context": {c: float(last_features[c]) for c in FRED_FEATURES if c in analysis.feature_cols and pd.notna(last_features[c])},
            "manual_sentiment": (st.session_state.sentiment_rows[["Sentimiento", "Score léxico"]].to_dict("records")
                                 if st.session_state.sentiment_rows is not None else []),
            "sec_term_frequencies": (st.session_state.sec_document["terms"] if st.session_state.sec_document else []),
        }
    with st.expander("Ejemplo de entrada que recibirá el LLM"):
        st.json(llm_input)
    st.caption("Estimación para 100 informes = 100 × (tokens medios de entrada × tarifa de entrada + tokens medios de salida × tarifa de salida) / 1.000.000. Consulta tarifa vigente y usa tokens medidos.")
    if st.button("Solicitar orientación académica al LLM", type="primary", disabled=(not llm_key or calls >= 20)):
        try:
            llm_started = time.perf_counter()
            log_request_attempt(llm_input)
            result = recommend(llm_input, api_key=llm_key, model=llm_model)
            result["latency_seconds"] = time.perf_counter() - llm_started
            log_recommendation(result, llm_input)
            st.session_state.llm_result = result
        except LLMError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"No se pudo completar el informe LLM: {exc}")
    if not llm_key:
        st.info("Configura OPENAI_API_KEY en .env o Secrets. La llamada se realiza solo al pulsar el botón.")
    if calls >= 20:
        st.warning("Se alcanzó el límite local de 20 intentos diarios.")
    llm_result = st.session_state.llm_result
    if llm_result:
        st.markdown(f"### Orientación: {llm_result['recommendation']}")
        st.write(llm_result["rationale"])
        st.write("**Riesgos**")
        for risk in llm_result["risks"]:
            st.markdown(f"- {risk}")
        st.write(f"**Confianza:** {llm_result['confidence']}")
        st.caption(llm_result["limitations"])
        st.caption(f"Tokens entrada/salida: {llm_result.get('usage', {}).get('input_tokens', 0)}/{llm_result.get('usage', {}).get('output_tokens', 0)}")
        st.caption(f"Latencia de esta ejecución: {llm_result.get('latency_seconds', 0):.2f} s. Para H8, repetir y registrar también la latencia desde la URL pública.")

with tabs[5]:
    st.subheader("Alpaca Paper — solo dinero simulado")
    st.warning("El broker solo usa el endpoint paper-api.alpaca.markets. No hay modo de órdenes reales.")
    if not alpaca_key or not alpaca_secret:
        st.info("Ingresa ALPACA_PAPER_KEY y ALPACA_PAPER_SECRET en la barra lateral para consultar la cuenta demo.")
    else:
        try:
            broker = AlpacaPaper(alpaca_key, alpaca_secret)
            account = broker.account()
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Estado", str(account.get("status", "—")))
            ac2.metric("Efectivo PAPER", f"${float(account.get('cash', 0)):,.2f}")
            ac3.metric("Patrimonio PAPER", f"${float(account.get('equity', 0)):,.2f}")
            positions = broker.positions()
            position = next((p for p in positions if p.get("symbol") == "SOXS"), None)
            orders = broker.orders()
            pending = [o for o in orders if o.get("symbol") == "SOXS" and o.get("status") in broker.ACTIVE_STATUSES]
            if position:
                st.dataframe(pd.DataFrame([{"Cantidad": position.get("qty"), "Entrada promedio": position.get("avg_entry_price"),
                                            "Precio actual": position.get("current_price"), "P/L no realizado": position.get("unrealized_pl")}]), use_container_width=True, hide_index=True)
            else:
                st.caption("No hay posición SOXS abierta.")
            st.markdown("**Operación manual PAPER**")
            side = st.selectbox("Acción", ["buy", "sell"], format_func=lambda v: "Comprar 1 participación" if v == "buy" else "Vender posición SOXS completa")
            confirm_manual = st.checkbox("Confirmo que enviaré una orden exclusivamente simulada.")
            manual_block = bool(pending) or (side == "buy" and bool(position)) or (side == "sell" and not position)
            if pending:
                st.info("Hay una orden pendiente; se bloquean nuevas órdenes para evitar duplicados.")
            elif side == "buy" and position:
                st.info("Ya hay una posición SOXS abierta; se bloquea una compra duplicada.")
            elif side == "sell" and not position:
                st.info("No hay posición para vender.")
            if st.button("Enviar orden manual PAPER", disabled=(manual_block or not confirm_manual)):
                qty = 1 if side == "buy" else int(float(position.get("qty", 0)))
                order = broker.submit_soxs_order(side, qty)
                decision = "COMPRA PAPER" if side == "buy" else "VENTA PAPER"
                added, event_key = record_alert(decision, analysis.signal, analysis.probability_up,
                                                f"Orden manual {side} de {qty} participación(es).", str(order.get("id", "")) or None)
                if added and telegram_token and telegram_chat:
                    try:
                        send_telegram_alert(telegram_token, telegram_chat,
                                            f"SOXS {decision}: {qty} participación(es). Señal {analysis.signal}; P(alza) {analysis.probability_up:.1%}. Orden PAPER {order.get('id', '—')}.")
                        set_telegram_status(event_key, "enviada")
                    except Exception as exc:
                        set_telegram_status(event_key, f"error: {type(exc).__name__}")
                st.success(f"Orden PAPER enviada · estado {order.get('status', '—')} · ID {order.get('id', '—')}")

            st.markdown("**Bot automático opcional PAPER**")
            auto_on = st.checkbox("Activar ciclos automáticos cada 5 minutos mientras esta app permanezca abierta", key="paper_auto_on")
            confirm_auto = st.checkbox("Confirmo que el bot puede enviar órdenes simuladas de SOXS.", key="paper_auto_confirm")
            entry_date_override = None
            if position:
                has_date = st.checkbox("Registrar manualmente la fecha de entrada si no aparece en las órdenes", key="paper_set_date")
                if has_date:
                    entry_date_override = st.date_input("Fecha de entrada de la posición", value=date.today(), key="paper_entry_date")
            st.caption(f"Señal actual: {analysis.signal} · P(alza siguiente sesión): {analysis.probability_up:.1%} · max {max_sessions} sesiones. Compra máxima 1 participación por ciclo; salida completa por señal o antigüedad.")

            @st.fragment(run_every=300)
            def automatic_paper_monitor(enabled: bool, confirmed: bool, key: str, secret: str,
                                        selected_period: str, macro_key: str, max_hold: int,
                                        commission_rate: float, manual_date):
                if not enabled:
                    st.caption("Bot desactivado. No se enviarán órdenes.")
                    return
                if not confirmed:
                    st.warning("Marca la confirmación PAPER para habilitar los ciclos.")
                    return
                try:
                    fresh_prices = cached_prices(selected_period)
                    fresh_macro = pd.DataFrame()
                    if macro_key:
                        start_fred = (pd.to_datetime(fresh_prices["Date"].min()) - pd.Timedelta(days=370)).date().isoformat()
                        fresh_macro = cached_fred(macro_key, start_fred)
                    fresh_analysis, _, _ = analyze(fresh_prices, fresh_macro if not fresh_macro.empty else None,
                                                   max_sessions=max_hold, commission=commission_rate)
                    paper = AlpacaPaper(key, secret)
                    result = auto_paper_cycle(paper, fresh_analysis.probability_up, fresh_prices["Date"],
                                              max_hold, manual_entry_date=manual_date)
                    result["probability_up"] = fresh_analysis.probability_up
                    result["signal"] = fresh_analysis.signal
                    st.session_state.bot_result = result
                    decision = str(result.get("decision", ""))
                    order_id = str(result.get("order_id", "")) or None
                    if decision in {"COMPRA PAPER", "VENTA PAPER", "REVISIÓN MANUAL", "ERROR"}:
                        added, event_key = record_alert(decision, fresh_analysis.signal,
                                                        fresh_analysis.probability_up, str(result.get("reason", "")), order_id)
                        if added and telegram_token and telegram_chat:
                            try:
                                send_telegram_alert(telegram_token, telegram_chat,
                                                    f"SOXS {decision}: {result.get('reason', '')} Señal {fresh_analysis.signal}; P(alza) {fresh_analysis.probability_up:.1%}. Solo PAPER.")
                                set_telegram_status(event_key, "enviada")
                            except Exception as exc:
                                set_telegram_status(event_key, f"error: {type(exc).__name__}")
                    st.info(f"Último ciclo: **{result['decision']}** · señal {result['signal']} ({result['probability_up']:.1%}) · {result['reason']}")
                except Exception as exc:
                    st.error(f"Falló el ciclo PAPER; no se continúa: {exc}")

            automatic_paper_monitor(auto_on, confirm_auto, alpaca_key, alpaca_secret,
                                    period, fred_key, max_sessions, commission_pct / 100, entry_date_override)
            if orders:
                st.markdown("**Órdenes recientes**")
                order_rows = [{"Fecha": o.get("submitted_at"), "Símbolo": o.get("symbol"), "Lado": o.get("side"),
                               "Cantidad": o.get("qty"), "Estado": o.get("status"), "ID": o.get("id")} for o in orders[:25]]
                st.dataframe(pd.DataFrame(order_rows), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"No se pudo consultar Alpaca Paper: {exc}")

with tabs[6]:
    st.subheader("Monitoreo y lista de evidencias")
    monitor_path = Path("data/monitoring_history.csv")
    record = {"fecha": pd.Timestamp.now().isoformat(timespec="seconds"), "modelo": analysis.model_name,
              "accuracy": analysis.metrics["accuracy"], "precision": analysis.metrics["precision"],
              "recall": analysis.metrics["recall"], "f1": analysis.metrics["f1"],
              "signal": analysis.signal, "probability_up": analysis.probability_up,
              "strategy_return": bt["strategy_return"], "strategy_drawdown": bt["strategy_drawdown"]}
    if st.button("Registrar ejecución del modelo"):
        monitor_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([record]).to_csv(monitor_path, mode="a", header=not monitor_path.exists(), index=False)
        st.success("Registro guardado localmente.")
    if monitor_path.exists():
        history = pd.read_csv(monitor_path)
        st.dataframe(history.tail(30).iloc[::-1], use_container_width=True, hide_index=True)
        recent_f1 = pd.to_numeric(history.tail(5)["f1"], errors="coerce").mean()
        if len(history) >= 5 and pd.notna(recent_f1) and recent_f1 < .50:
            st.warning(f"Revisar modelo: F1 promedio de las últimas cinco ejecuciones = {recent_f1:.1%}.")
        else:
            st.info("Seguimiento descriptivo. El sistema no tiene un detector automático de drift.")
        st.download_button("Descargar monitoreo CSV", history.to_csv(index=False).encode("utf-8"), "soxs_monitoreo.csv", "text/csv")
    alert_path = Path("data/alerts.csv")
    st.markdown("**Historial de alertas**")
    if alert_path.exists():
        alerts = pd.read_csv(alert_path)
        st.dataframe(alerts.tail(50).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button("Descargar alertas CSV", alerts.to_csv(index=False).encode("utf-8"), "soxs_alertas.csv", "text/csv")
    else:
        st.caption("Aún no hay alertas. Las órdenes manuales y eventos importantes del bot PAPER se registran aquí.")
    if telegram_token and telegram_chat:
        st.caption("Telegram está configurado. Los avisos se envían solo para órdenes y eventos importantes, mientras la app está abierta.")
    else:
        st.caption("Telegram es opcional. Configura token y chat ID en el panel lateral para recibir avisos.")
    st.markdown("**Pendiente antes de declarar la entrega completa**")
    st.markdown("- Publicar el tablero en una URL pública y abrirla sin iniciar sesión.")
    st.markdown("- Subir el código al GitHub del equipo, revisar el historial y compartir el enlace.")
    st.markdown("- Ejecutar la llamada real al LLM, conservar tres ejemplos revisados y registrar tokens/latencia.")
    st.markdown("- Adjuntar capturas reales de los tres reportes, valores de las 3V, métricas, hallazgos y matriz de pruebas.")
    st.markdown("- Grabar el video de demostración de 3 a 5 minutos y completar cronograma/aportes del equipo.")

st.divider()
st.caption("Uso académico. SOXS tiene un objetivo inverso apalancado diario; el rendimiento en varios días puede diferir. Los resultados no garantizan desempeño futuro. Todas las órdenes del programa son Alpaca Paper.")
