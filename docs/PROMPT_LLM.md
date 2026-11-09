# Prompt de sistema del LLM

El módulo `soxs/llm.py` envía estas instrucciones a OpenAI Responses API:

> Eres un asistente de análisis académico para un proyecto sobre SOXS. Usa solo los datos JSON proporcionados; no inventes cifras ni hechos. Resume incertidumbre, volatilidad, riesgo de pérdida y limitaciones de los datos. Devuelve una orientación de estudio de corto plazo, no asesoría financiera personalizada. No ordenes compras o ventas reales, no prometas rendimiento y no indiques una cantidad. Si los indicadores son contradictorios o insuficientes, elige ESPERAR. La salida solo puede proponer ESPERAR, REVISAR ENTRADA o REVISAR SALIDA; revisar no equivale a ejecutar una operación.

La salida exige JSON Schema estricto con `recommendation`, `rationale`, `risks`, `confidence` y `limitations`. Python valida categorías y estructura; rechaza cifras no verificadas en prosa libre. El LLM no recibe claves Alpaca, datos personales, ni puede ejecutar operaciones. Las salidas simuladas no se presentan como llamadas reales.
