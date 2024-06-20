"""
RiskFactors class
"""

import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import DATA_PATH
from source.utils import OPT_PARAMS

PricesDict = dict[str, float]


class RiskFactors:
    """
    Risk factors implementation:
    - simulations for the period
    - instruments price predictions
    """

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
        simulations: np.array,
        risk_factor: str,
    ):
        plt.figure(figsize=(10, 5))

        plt.title(f'{risk_factor} simulations')
        plt.xlabel('Days')
        plt.ylabel('Value')

        for sim in simulations:
            plt.plot(sim)

        target = self.data.loc[self._current_date_str:, risk_factor].iloc[:simulations.shape[1]]
        print(target)
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

    def _log_sim(self, x_0, r_f, r_d, sigma, n_days, n_sim, deltas_W=None) -> np.array:
        """
        x_0 - float
        r_f - np.array[n_sim x (n_days + 1)]
        r_d - np.array[n_sim x (n_days + 1)]
        sigma - float
        n_sim - int

        result - np.array[n_sim x (n_days + 1)]
        """
        result = np.array([x_0] * n_sim).reshape(-1, 1)
        for i in range(1, n_days + 1):
            x_prev = result[:, -1].reshape(-1, 1)
            delta_t = 1 / 365

            if deltas_W is None:
                delta_W = np.random.normal(loc=0, scale=np.sqrt(delta_t), size=(n_sim, 1))
            else:
                delta_W = deltas_W[:, i - 1].reshape(-1, 1)

            r_d_i = r_d[:, i - 1].reshape(-1, 1)
            r_f_i = r_f[:, i - 1].reshape(-1, 1)

            x_t_i = x_prev + x_prev * (r_f_i - r_d_i) * delta_t + sigma * x_prev * delta_W
            result = np.hstack([result, x_t_i])
        return result

    def simulate_rates(self, risk_factor: str, n_days: int = 1, n_sim: int = 1_000) -> np.array:
        """
        Rates simulation for n_days forward
        """
        model_params = OPT_PARAMS[risk_factor]
        model_params['r_0'] = self.data.loc[self._current_date_str, risk_factor],
        simulations = self._cir_sim(n_sim=n_sim, n_days=n_days, **model_params)
        return simulations

    def simulate_fx(
        self,
        risk_factor: str,
        domestic_rates: str,
        foreign_rates: str,
        n_days: int = 1,
        n_sim: int = 1_000,
    ) -> np.array:
        """
        FX simulation for n_days forward
        """
        COV_MATRIX = self.data[[foreign_rates, domestic_rates, risk_factor]].cov()
        L = np.linalg.cholesky(COV_MATRIX)
        W_t = np.random.randn(n_sim, 3, n_days)
        W_t_corr = L @ W_t

        r_f_params = OPT_PARAMS[foreign_rates]
        r_f_params['r_0'] = self.data.loc[self._current_date_str, foreign_rates]
        r_f = self._cir_sim(n_sim=n_sim, n_days=n_days, deltas_W=W_t_corr[:, 0], **r_f_params)

        r_d_params = OPT_PARAMS[domestic_rates]
        r_d_params['r_0'] = self.data.loc[self._current_date_str, domestic_rates]
        r_d = self._cir_sim(n_sim=n_sim, n_days=n_days, deltas_W=W_t_corr[:, 1], **r_d_params)

        model_params = OPT_PARAMS[risk_factor]
        model_params['x_0'] = self.data.loc[self._current_date_str, risk_factor]
        simulations = self._log_sim(
            n_sim=n_sim,
            n_days=n_days,
            r_f=r_f,
            r_d=r_d,
            deltas_W=W_t_corr[:, 2],
            **model_params,
        )

        return simulations

    def simulate_all(self, n_days: int = 1, n_sim: int = 1_000) -> dict[str, np.array]:
        """
        Simulate all risk factors
        """
        return {
            'cbr_key_rate': self.simulate_rates(risk_factor='cbr_key_rate', n_days=n_days, n_sim=n_sim),
            'pca_cbd': self.simulate_rates(risk_factor='pca_cbd', n_days=n_days, n_sim=n_sim),
            'usd_rub': self.simulate_fx(
                risk_factor='usd_rub',
                domestic_rates='cbr_key_rate',
                foreign_rates='sofr',
                n_days=n_days,
                n_sim=n_sim,
            ),
            'eur_rub': self.simulate_fx(
                risk_factor='eur_rub',
                domestic_rates='cbr_key_rate',
                foreign_rates='ecb_rate',
                n_days=n_days,
                n_sim=n_sim,
            ),
        }

    def predict_prices(self, n_days: int = 1, n_sim: int = 1000) -> list[PricesDict]:
        """
        Predict instruments price based on risk factors for n_days horizon
        Return list of n_sim simulations, each of which with M instruments price predictions
        """
        simulations_dict = self.simulate_all(n_days, n_sim)
        # todo: predict instruments prices based on risk factors
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
