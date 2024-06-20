"""
RiskFactors class
"""

import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DATA_PATH

PricesDict = dict[str, float]


class RiskFactors:
    """
    Risk factors implementation:
    - simulations for the period
    - instruments price predictions
    """

    key_rate_params = {  # optimized in notebooks/rates_simulations.ipynb
        'a': 0.01,
        'b': 7.85,
        'sigma': 0.11,
    }

    # todo: remove crutch
    risk_free_params = {  # optimized in notebooks/rates_simulations.ipynb
        'a': 0.01,
        'b': 7.85,
        'sigma': 0.11,
    }

    def __init__(self, current_date: pd.Timestamp):
        self._current_date = current_date
        self.data = self.load_data()

    @property
    def _current_date_str(self):
        return f'{self._current_date:%Y-%m-%d}'

    @staticmethod
    def load_data() -> pd.DataFrame:
        """
        Load risk factors data
        """
        return pd.read_csv(DATA_PATH / 'all_data.csv', index_col='date')

    def plot_simulations(
        self,
        simulations: dict[str, np.array],
        risk_factor: str,
    ):
        plt.figure(figsize=(10, 5))

        plt.title(f'{risk_factor} simulations')
        plt.xlabel('Days')
        plt.ylabel('Value')

        for sim in simulations[risk_factor]:
            plt.plot(sim)

        target = self.data.loc[self._current_date_str:, risk_factor].iloc[:simulations[risk_factor].shape[1]]
        plt.plot(target, color='black', label='target')

        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=int(np.log(100))))
        plt.grid()
        plt.legend()
        plt.show()

    def _cir_sim(self, r_0, a, b, sigma, n_days, n_sim, deltas_W=None) -> np.array:
        """
        CIR model implementation
        """
        result = np.array([r_0] * n_sim).reshape(-1, 1)
        for i in range(1, n_days + 1):
            r_prev = result[:, -1].reshape(-1, 1)
            delta_t = 1
            if deltas_W is None:
                delta_W = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(n_sim, 1))
            else:
                delta_W = deltas_W[:, i - 1].reshape(-1, 1)
            r_t_i = r_prev + a * (b - r_prev) * delta_t + sigma * np.sqrt(r_prev) * delta_W
            r_t_i = np.clip(r_t_i, 0, 100000000)
            result = np.hstack([result, r_t_i])
        return result

    def simulate_rates(self, n_days: int = 1, n_sim: int = 1_000) -> tuple[np.array, np.array]:
        """
        Rates simulation for n_days forward
        """
        self.key_rate_params.update({'r_0': self.data.loc[self._current_date_str, 'cbr_key_rate']})
        self.risk_free_params.update(  # todo: remove crutch
            {'r_0': self.data.loc[self._current_date_str, 'cbr_key_rate']}
        )

        # COV_MATRIX = np.cov(self.data['cbr_key_rate'], self.data['risk_free_rate'])
        COV_MATRIX = np.array([  # todo: remove crutch
            [10.0, -2.00],
            [-2.00, 20.00],
        ])

        L = np.linalg.cholesky(COV_MATRIX)
        W_t = np.random.randn(n_sim, 2, n_days)
        W_t_corr = L @ W_t

        key_rate_sim = self._cir_sim(n_sim=n_sim, n_days=n_days, deltas_W=W_t_corr[:, 0], **self.key_rate_params)
        risk_free_sim = self._cir_sim(n_sim=n_sim, n_days=n_days, deltas_W=W_t_corr[:, 1], **self.risk_free_params)

        return key_rate_sim, risk_free_sim

    def simulate_all(self, n_days: int = 1, n_sim: int = 1_000) -> dict[str, np.array]:
        """
        Simulate all risk factors
        """
        key_rate_sim, risk_free_sim = self.simulate_rates(n_days, n_sim)
        return {
            'cbr_key_rate': key_rate_sim,
            'risk_free_rate': risk_free_sim,
        }

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
