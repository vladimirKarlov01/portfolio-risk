"""
RiskFactors class
"""

import warnings
from itertools import product

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso

from data import DATA_PATH
from source.utils import OPT_PARAMS, factor_final, factors_to_tickers_mapping

# from statsmodels.tsa.stattools import adfuller, acf, pacf

PricesDict = dict[str, float]


class ArimaFactors:
    """
    SARIMA simulations util
    """

    def __init__(self,
                 current_date: pd.Timestamp,
                 data: pd.DataFrame,
                 seasonal=0,
                 ):
        self._current_date = current_date
        self.data = data
        self.seasonal = seasonal
        self.models = dict()
        self.model_orders = dict()

    def find_last_acf_sign_lag(self, ts, drop_first=True):
        _, ci = sm.tsa.acf(ts, alpha=0.01)
        first_zero, second_zero = 0, 0
        for l in range(1, len(ci)):
            if (0 >= ci[l][0] and 0 <= ci[l][1]):
                if first_zero == 0:
                    first_zero = l - 1
                else:
                    second_zero = l - 1
                    break
        return first_zero

    def find_last_pacf_sign_lag(self, ts, drop_first=True):
        _, ci = sm.tsa.pacf(ts, alpha=0.01)
        first_zero, second_zero = 0, 0
        for l in range(1, len(ci)):
            if (0 >= ci[l][0] and 0 <= ci[l][1]):
                if first_zero == 0:
                    first_zero = l - 1
                else:
                    second_zero = l - 1
                    break
        return first_zero

    def fit_best_sarima_model(
        self,
        series: pd.Series,
        seasonal=0,
        ps=range(0, 3),
        d=[1],
        qs=range(0, 3),
        Ps=None,
        D=None,
        Qs=None,
    ):
        max_params = [
            self.find_last_pacf_sign_lag(series.diff().dropna()),
            self.find_last_acf_sign_lag(series.diff().dropna()),
            1, 1]

        ps = range(0, max_params[0] + 1) if not ps else ps
        d = [1] if not d else d
        qs = range(0, max_params[1] + 1) if not qs else qs
        Ps = range(0, max_params[2]) if not Ps else Ps
        D = [1] if not D else D
        Qs = range(0, max_params[3]) if not Qs else Qs

        grid = [ps, d, qs]

        if seasonal:
            grid += [Ps, D, Qs]
        parameters_list = list(product(*grid))
        results = []
        best_aic = float("inf")
        warnings.filterwarnings('ignore')

        for param in parameters_list:
            # try except нужен, потому что на некоторых наборах параметров модель не обучается
            order_list = (*param[:3],)
            seasonal_list = (*param[3:7],) if seasonal else (0, 0, 0, 0)

            try:
                model_sm = sm.tsa.SARIMAX(
                    series,
                    order=order_list,
                    seasonal_order=seasonal_list,
                ).fit(disp=-1)

            # выводим параметры, на которых модель не обучается и переходим к следующему набору
            except ValueError:
                print('wrong parameters:', param)
                continue
            aic = model_sm.aic

            # сохраняем лучшую модель, aic, параметры
            if aic < best_aic:
                best_model = model_sm
                best_aic = aic
                best_param = param

            results.append([param, model_sm.aic])

        return best_model, best_param

    def fit(self):
        for column in self.data.columns:
            self.models[column], self.model_orders = \
                self.fit_best_sarima_model(self.data[column])

    def forecast(self,
                 steps,
                 factor_list=None,
                 ):

        factor_list = self.data.columns

        predicts = [self.models[factor].get_forecast(steps=steps) for factor in factor_list]

        mean_series = [pred.predicted_mean.values for pred in predicts]
        var_series = [pred.var_pred_mean.values for pred in predicts]

        return mean_series, var_series

    def simulate(self, steps, factor_list=None):

        ANCHOR = self.data.shape[0]
        factor_list = self.data.columns if not factor_list else factor_list
        return [
            self.models[factor].simulate(
                nsimulations=steps, anchor=ANCHOR
            ).values for factor in factor_list
        ]


class RiskFactors:
    """
    Risk factors implementation:
    - simulations for the period
    - instruments price predictions
    """

    all_tickers = (
        'SU26218RMFS6', 'SU26221RMFS0', 'SU26222RMFS8', 'SU26224RMFS4', 'SU26230RMFS1',
        'GAZP', 'GMKN', 'LKOH', 'MAGN', 'MGNT', 'MOEX', 'ROSN', 'RUAL', 'SBER', 'VTBR',
        'USD_RUB', 'EUR_RUB',
    )

    def __init__(self, current_date: pd.Timestamp):
        self._current_date = current_date
        self.data = self.load_data()
        self._initialize_models()

    @property
    def _current_date_str(self):
        return f'{self._current_date:%Y-%m-%d}'

    @staticmethod
    def load_data() -> pd.DataFrame:
        """
        Load risk factors data
        """
        return pd.read_csv(DATA_PATH / 'all_data.csv', index_col='date')

    def _initialize_models(self):
        self._active_models = {
            ticker: self._fit_regression(ticker)
            for ticker in factor_final
        }

    def _fit_regression(self, ticker: str):
        X = self.data.loc[:self._current_date_str, factor_final[ticker]]
        y = self.data.loc[:self._current_date_str, ticker]
        lr = Lasso().fit(X, y)
        return lr

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
        return result[:, 1:]

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
        arima_factors = ['su26224_days_before_coupon',
                         'nickel',
                         'aluminum',
                         'brent',
                         'moex',
                         'moex_index',
                         'su26222_days_before_coupon',
                         'su26218_days_before_coupon',
                         'su26221_days_before_coupon',
                         'rtsi']

        arimas = ArimaFactors(
            self._current_date,
            self.data.loc[self._current_date_str:, arima_factors]
        )

        arimas.fit()

        return_dict = {
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
            'ecb_rate': self.simulate_rates(risk_factor='ecb_rate', n_days=n_days, n_sim=n_sim),
            'sofr': self.simulate_rates(risk_factor='sofr', n_days=n_days, n_sim=n_sim),
        }

        semi_dict = {}
        for name in arima_factors:
            semi_dict[name] = []

        for _ in range(n_sim):
            arimas_sim = arimas.simulate(n_days, arima_factors)
            for name, sim in zip(arima_factors, arimas_sim):
                semi_dict[name].append(sim)

        for name, array in semi_dict.items():
            return_dict[name] = np.array(array)

        return return_dict

    def predict_prices_for_ticker(self, ticker: str, risk_factors: pd.DataFrame) -> pd.Series:
        """
        Call model for instruments price
        risk_factors: N observations x M related factors
        """
        estimator = self._active_models[ticker]
        preds = estimator.predict(risk_factors)
        return preds

    def predict_prices(self, n_days: int = 1, n_sim: int = 1000) -> list[PricesDict]:
        """
        Predict instruments price based on risk factors for n_days horizon
        Return list of n_sim simulations, each of which with M instruments price predictions
        """
        simulations_dict = self.simulate_all(n_days, n_sim)
        prices = pd.DataFrame.from_dict(
            {
                factors_to_tickers_mapping[instrument_name]: self.predict_prices_for_ticker(
                    ticker=instrument_name,
                    risk_factors=pd.DataFrame.from_dict(
                        {
                            factor_name: simulations_dict[factor_name][:, -1:].flatten()
                            for factor_name in factor_final[instrument_name]
                        }
                    ),
                )
                for instrument_name in factor_final
            }
        )
        return [
            row.to_dict()
            for _, row in prices.iterrows()
        ]
