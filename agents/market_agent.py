from soxs.core import download_prices


class MarketAgent:
    """Obtiene y prepara los datos históricos diarios de SOXS."""

    def run(self, period: str = "5y"):
        return download_prices(period)
