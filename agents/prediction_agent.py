from soxs.core import analyze


class PredictionAgent:
    """Evalúa modelos de ML y produce señal y backtest."""

    def run(self, prices, fred=None, max_sessions: int = 20, commission: float = 0.001):
        return analyze(prices, fred, max_sessions=max_sessions, commission=commission)
