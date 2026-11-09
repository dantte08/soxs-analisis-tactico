# Despliegue en Streamlit Community Cloud

1. Sube el contenido del proyecto a un repositorio GitHub del equipo, sin `.env`, claves, credenciales ni logs locales.
2. En Streamlit Community Cloud, crea una aplicación desde el repositorio y selecciona `app.py` como archivo principal.
3. Añade solo los nombres/valores secretos necesarios en el panel **Secrets** del servicio: `FRED_API_KEY`, `SEC_USER_AGENT`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `ALPACA_PAPER_KEY`, `ALPACA_PAPER_SECRET`.
4. Despliega y revisa logs. Abre la URL pública en ventana privada/sesión sin autenticar.
5. Captura el reporte descriptivo, predictivo, prescriptivo, el estado de fuentes y la versión pública. Oculta claves y datos sensibles.
6. Prueba la llamada OpenAI real y Alpaca Paper solo con la cuenta demo. Registra tiempos y resultados; no afirmes que las pruebas locales equivalen a evidencia en línea.

Este documento describe los pasos; no implica que el proyecto ya esté publicado ni que tenga URL verificada.
