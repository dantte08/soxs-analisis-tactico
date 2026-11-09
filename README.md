# SHORT – Inversiones SOXS | Equipo 4

Aplicación web académica con Streamlit para estudiar señales tácticas de SOXS. Incluye reportes descriptivo, predictivo y prescriptivo; modelos Random Forest y XGBoost; línea base de clase mayoritaria; backtesting y conexiones opcionales a FRED, SEC EDGAR, OpenAI Responses API y Alpaca Paper.

## Alcance de la estrategia

- SOXS es un ETF inverso apalancado con objetivo de rendimiento **diario**. El sistema estima la dirección del siguiente día bursátil; no proyecta el objetivo diario de SOXS como múltiplo constante durante un mes.
- Cada posición de la estrategia simulada se limita a un máximo configurable de 20 sesiones bursátiles (aproximadamente un mes). El selector 2/5/10 años define el historial usado para entrenar y evaluar, no cuánto tiempo se mantiene una posición.
- La referencia “SOXS sin límite” se muestra únicamente como comparación dentro del bloque de prueba; no es una estrategia recomendada.
- No se envían operaciones reales. Alpaca se fija en `https://paper-api.alpaca.markets`; la compra automática PAPER está desactivada por defecto y requiere habilitación/confirmación explícitas.

## Módulos

| Archivo | Responsabilidad |
|---|---|
| `app.py` | UI Streamlit y reportes descriptivo, predictivo, prescriptivo, fuentes, Paper y monitoreo. |
| `soxs/core.py` | Limpieza OHLCV, RSI, MACD, línea base, modelos, partición temporal, métricas y backtesting. |
| `soxs/sources.py` | FRED con valores de primera publicación, consultas SEC, extracción documental y léxico manual. |
| `soxs/llm.py` | OpenAI Responses API con JSON Schema estricto, registro y límite local de 20 intentos diarios. |
| `soxs/paper_broker.py` | Cliente Alpaca Paper y ciclo opcional con control de duplicados/antigüedad. |
| `soxs/alerts.py` | Registro CSV de alertas, deduplicación y envío opcional a Telegram. |
| `tests/` | 11 pruebas unitarias para features, etiquetas, backtest, FRED y alertas. |

## Ejecutar en Windows

1. Descomprime el ZIP y abre PowerShell en `soxs_academico_mejorado`.
2. Crea y activa el entorno virtual:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Instala dependencias e inicia Streamlit:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```

Abre la URL que muestra la terminal, normalmente `http://localhost:8501`. Déjala abierta mientras uses el tablero. Para iniciar otro día, activa `.venv` y vuelve a ejecutar `python -m streamlit run app.py`.

Si PowerShell bloquea la activación, ejecuta `Set-ExecutionPolicy -Scope Process Bypass` en esa ventana y vuelve a activar `.venv`. Alternativamente, en CMD usa `.venv\Scripts\activate.bat`.

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## Claves opcionales

Duplica `.env.example` como `.env` y completa solo las APIs que usarás. En Streamlit Community Cloud, coloca las mismas variables en **Secrets**. No subas `.env` al repositorio ni compartas capturas con claves.

| Variable | Uso |
|---|---|
| `FRED_API_KEY` | Series DFF, DGS10, VIXCLS y UNRATE. Los valores iniciales de publicación se alinean por disponibilidad con merge hacia atrás. |
| `SEC_USER_AGENT` | Identificación del equipo con correo de contacto para consultar SEC EDGAR. Ejemplo de formato: `Equipo4 SOXS contacto@universidad.edu`. Reemplaza el ejemplo por un correo válido. |
| `OPENAI_API_KEY` | Solicitud opcional de orientación académica estructurada. |
| `OPENAI_MODEL` | Modelo LLM; por defecto `gpt-4o-mini`. |
| `ALPACA_PAPER_KEY`, `ALPACA_PAPER_SECRET` | Cuenta de simulación Alpaca Paper. |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Avisos opcionales para órdenes y eventos importantes; se mantienen fuera de los registros. |

El análisis de mercado y el backtest funcionan sin estas credenciales. FRED/SEC/LLM/Alpaca son opcionales. FRED requiere una clave activa de 32 caracteres alfanuméricos en minúscula. La descarga de primeras publicaciones se divide en ventanas de dos años para respetar el máximo de vintages por consulta. Los errores de FRED se muestran sin imprimir la URL de solicitud ni la clave. La clave FRED y las claves de Alpaca también pueden escribirse en campos protegidos del panel lateral; no son guardadas en los CSV.

## Metodología y métricas

El proceso sigue CRISP-DM: comprensión del problema (señales tácticas para SOXS); comprensión/preparación de datos (Yahoo Finance y FRED opcional); modelado (RF/XGBoost y línea base); evaluación (métricas y backtest temporal); despliegue (Streamlit); y monitoreo (registro manual de resultados).

1. Yahoo Finance descarga OHLCV ajustado de SOXS; se deduplican fechas, se validan tipos y se descartan cierre no positivo, volumen negativo, High menor que Low y filas incompletas.
2. Las variables técnicas son retorno diario, RSI 14, cierre/MA5, cierre/MA20, volatilidad móvil 20, cambio de volumen y MACD (línea, señal e histograma normalizados por precio). FRED es opcional y se alinea usando valores de primera publicación con su fecha de disponibilidad. El sentimiento manual y los documentos SEC son contexto actual descriptivo y no se presentan como features históricas entrenadas.
3. La etiqueta es 1 si el cierre ajustado de la siguiente sesión supera al actual; la última fila queda sin etiqueta futura.
4. No se barajan las fechas. Se usa 80 % inicial para entrenar y 20 % final para probar; se comparan RF y XGBoost contra DummyClassifier de clase mayoritaria. El modelo ML con mayor F1 alimenta la señal/backtest.
5. El backtest aplica decisiones al retorno de la sesión siguiente, descuenta una comisión simulada por cambio de posición, limita cada tenencia a 20 sesiones y compara retorno/drawdown con SOXS de referencia.
6. Se muestran accuracy, precisión, recall, F1, retorno neto, drawdown, cambios de posición y cifras de calidad/3V. Las métricas dependen del periodo, fecha de descarga, APIs disponibles y comisión ingresada.

## LLM y seguridad

El informe LLM se ejecuta solo al pulsar el botón. Usa Responses API, salida JSON Schema estricta, categorías `ESPERAR`, `REVISAR ENTRADA` o `REVISAR SALIDA`, máximo de 300 tokens, máximo local de 20 intentos por día UTC y manejo de errores. El log JSONL omite la clave y el prompt íntegro. El LLM no recibe credenciales de Alpaca ni envía órdenes.

Estimación para 100 informes: `100 × (tokens medios de entrada × tarifa de entrada + tokens medios de salida × tarifa de salida) / 1.000.000`. Completa la tarifa vigente y el promedio de tokens observados cuando ejecutes llamadas reales; el proyecto no inventa un costo.

El broker está fijado al endpoint PAPER, símbolo SOXS, orden de mercado y entrada de máximo una participación por ciclo. El modo automático requiere dos confirmaciones y revisa órdenes pendientes, posición y edad; si no logra verificar antigüedad, pide revisión manual. Las órdenes manuales y eventos importantes del bot quedan en `data/alerts.csv`; Telegram es opcional y avisa solo cuando la aplicación está ejecutándose. La app nunca promete rendimiento ni constituye asesoría financiera.

## Pruebas

La matriz está en `docs/MATRIZ_PRUEBAS.md`. `tests/test_core.py` contiene 11 pruebas unitarias: etiqueta futura, disponibilidad FRED, MACD, límite de tenencia, comisión, zona neutral, salida y deduplicación de alertas. Para ejecutarlas en el entorno local:

```bash
pytest -q
```

El código de pruebas no equivale a evidencia de ejecución. Registra el resultado real y una captura después de ejecutar; las comprobaciones conectadas a APIs requieren las credenciales del equipo.

## Despliegue y entregables que dependen del equipo

Se incluye `Dockerfile` y `.streamlit/config.toml` para facilitar Streamlit Community Cloud. Para la entrega final el equipo aún debe crear/compartir GitHub, desplegar la URL pública, verificarla sin sesión, correr las APIs que se usarán con claves privadas, completar métricas/hallazgos y capturas reales, ejecutar pruebas, asignar aportes/cronograma y grabar el video de 3–5 minutos. Este ZIP no inventa URLs, resultados ni evidencia.

## Limitaciones

Las fuentes pueden retrasarse o corregirse; métricas pasadas no garantizan desempeño. El backtest simplifica ejecución diaria y no modela totalmente spreads, impuestos, liquidez, costos de préstamo ni desviación intradía. El léxico manual no sustituye un histórico de noticias con fecha. Una posición preexistente sin fecha verificable requiere intervención manual para controlar el máximo de sesiones. El CSV local puede no persistir en hosting efímero.

## Fuentes del concepto

- Direxion, SOXL/SOXS: https://www.direxion.com/product/daily-semiconductor-bull-bear-3x-etfs
- SEC, riesgos de ETF apalancados/inversos: https://www.sec.gov/files/investor/pubs/leveragedetfs-alert.htm
- FRED API, primeras publicaciones: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
