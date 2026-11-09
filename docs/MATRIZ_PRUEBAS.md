# Matriz de pruebas del proyecto SOXS

Registra fecha, entorno, resultado real y captura al ejecutarla. No rellenar con resultados supuestos. Pruebas marcadas **Pendiente de ejecución en el equipo**.

| ID | Reporte | Caso / datos | Resultado esperado | Estado / evidencia |
|---|---|---|---|---|
| P01 | Descriptivo | Descarga OHLCV válida y serie con fecha duplicada | Fechas únicas y duplicados reportados/eliminados | Pendiente |
| P02 | Descriptivo | Cierre nulo/no positivo, volumen negativo y High < Low | Fila inválida descartada y contadores coherentes | Pendiente |
| P03 | Predictivo | Serie histórica con último día sin precio futuro | Última etiqueta queda nula y no entra en entrenamiento/prueba | Pendiente |
| P04 | Predictivo | División temporal 80/20 | Entrenamiento anterior a prueba; no se mezclan fechas | Pendiente |
| P05 | Predictivo | Clases de prueba y baseline de clase mayoritaria | Accuracy, precisión, recall y F1 calculadas para baseline y modelos | Pendiente |
| P06 | Backtesting | Probabilidad alta persistente por más de 20 sesiones | Una posición individual no supera 20 sesiones; salida y reingreso requieren señal de sesión nueva | Pendiente |
| P07 | Backtesting | Probabilidad entre 0.40 y 0.60 y luego salida ≤ 0.40 | No abre en zona neutral; sale con señal de salida | Pendiente |
| P08 | LLM | API key ausente / error HTTP / JSON no válido | Mensaje controlado; no se muestra orden; se cuenta intento | Pendiente |
| P09 | LLM | Respuesta JSON válida y llamada real | Categoría, riesgos, confianza y limitaciones visibles; registrar tokens/captura | Pendiente: requiere clave OpenAI |
| P10 | Broker | Clave PAPER válida, orden pendiente o posición duplicada | No duplicar órdenes; orden de entrada máxima de 1 participación | Pendiente: requiere cuenta Alpaca Paper |
| P11 | Broker | Posición con edad ≥20 sesiones verificable | Solicitar cierre PAPER de toda la posición; si edad no verificable, pedir revisión manual | Pendiente: requiere cuenta PAPER |
| P12 | Despliegue | Abrir URL pública en navegador sin iniciar sesión | Tablero accesible, tres reportes visibles y URL/capturas guardadas | Pendiente: requiere despliegue |
| P13 | Predictivo | Calcular línea MACD, señal e histograma | Las tres variables se crean y entran al conjunto de features | Pendiente de ejecución |
| P14 | Monitoreo | Registrar dos veces la misma orden PAPER con el mismo ID | El historial CSV conserva un solo evento | Pendiente de ejecución |
| P15 | Alertas | Configurar bot y chat de Telegram y producir un evento importante | El aviso llega sin mostrar token en pantalla ni guardar la clave | Pendiente: requiere credenciales de Telegram |
