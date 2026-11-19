from soxs.paper_broker import auto_paper_cycle


class DecisionAgent:
    """Coordina la señal del modelo con las reglas de riesgo y Alpaca Paper."""

    def run(self, broker, probability_up, market_dates, max_sessions: int = 20,
            manual_entry_date=None):
        return auto_paper_cycle(
            broker,
            probability_up,
            market_dates,
            max_sessions=max_sessions,
            manual_entry_date=manual_entry_date,
        )
