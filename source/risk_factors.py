"""
RiskFactors class
"""

import random

PricesDict = dict[str, float]


class RiskFactors:
    """
    Risk factors implementation:
    - simulations for the period
    - instruments price predictions
    """

    def __init__(self):
        pass

    def predict_prices(self, n_days: int = 1, n_sim: int = 1000) -> list[PricesDict]:
        """
        Predict instruments price based on risk factors for n_days horizon
        Return list of n_sim simulations, each of which with M instruments price predictions
        """
        _ = self
        _ = n_days
        # todo: risk factors simulations
        return [
            {
                'SU26218RMFS6': round(random.uniform(100, 1500), 2),
                'SU26221RMFS0': round(random.uniform(100, 1500), 2),
                'SU26222RMFS8': round(random.uniform(100, 1500), 2),
                'SU26224RMFS4': round(random.uniform(100, 1500), 2),
                'SU26230RMFS1': round(random.uniform(100, 1500), 2),
                'GAZP': round(random.uniform(10, 10000), 2),
                'GMKN': round(random.uniform(10, 10000), 2),
                'LKOH': round(random.uniform(10, 10000), 2),
                'MAGN': round(random.uniform(10, 10000), 2),
                'MGNT': round(random.uniform(10, 10000), 2),
                'MOEX': round(random.uniform(10, 10000), 2),
                'ROSN': round(random.uniform(10, 10000), 2),
                'RUAL': round(random.uniform(10, 10000), 2),
                'SBER': round(random.uniform(10, 10000), 2),
                'VTBR': round(random.uniform(10, 10000), 2),
                'USD_RUB': round(random.uniform(80, 110), 2),
                'EUR_RUB': round(random.uniform(90, 120), 2),
            }
            for _ in range(n_sim)
        ]
