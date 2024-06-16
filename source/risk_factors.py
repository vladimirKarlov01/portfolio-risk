"""
RiskFactors class
"""

import numpy as np

PricesDict = dict[str, float]


class RiskFactors:
    """
    Risk factors implementation:
    - simulations for the period
    - instruments price predictions
    """

    def __init__(self):
        pass

    def predict_prices(self, risk_factors: np.array) -> PricesDict:
        """
        Predict instruments price based on risk factors
        """
        # todo: call strong model

    def simulate_prices(self, n_days: int = 1) -> list[PricesDict]:
        """
        Simulate instruments prices
        """
        # todo: simulate risk factors
        # todo: for each simulation predict prices
