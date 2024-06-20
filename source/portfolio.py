"""
Portfolio class
"""

import numpy as np
import pandas as pd

from data import DATA_PATH
from source.risk_factors import RiskFactors, PricesDict


class Portfolio:
    """
    Portfolio implementation:
    - instruments rebalancing
    - fair value calculation
    - risk metrics evaluation
    """

    # all portfolio instruments are given below:
    bonds = ('SU26218RMFS6', 'SU26221RMFS0', 'SU26222RMFS8', 'SU26224RMFS4', 'SU26230RMFS1')
    stocks = ('GAZP', 'GMKN', 'LKOH', 'MAGN', 'MGNT', 'MOEX', 'ROSN', 'RUAL', 'SBER', 'VTBR')
    fx = ('USD_RUB', 'EUR_RUB')

    def __init__(
        self,
        start_date: str,
        investing_amounts: dict[str, float],
        max_relative_error: float,
    ):
        """
        Initialize base parameters and load prices
        start_date – date to start portfolio optimization for
        investing_amounts – amount of rubles to place in one instrument for all types
        max_relative_error – max deviation from target weight to start rebalancing
        """
        self._current_date = pd.Timestamp(start_date)
        self._investing_amounts = investing_amounts
        self._max_relative_error = max_relative_error
        self._prices = self.load_prices()

        self.total_amount = (
            investing_amounts['bonds'] * len(self.bonds)
            + investing_amounts['stocks'] * len(self.stocks)
            + investing_amounts['fx'] * len(self.fx)
        )

        # target weights for each instrument in group
        self.target_weights = {
            'bonds': investing_amounts['bonds'] / self.total_amount,
            'stocks': investing_amounts['stocks'] / self.total_amount,
            'fx': investing_amounts['fx'] / self.total_amount,
        }

        self.all_instruments = {
            'bonds': {ticker: 0 for ticker in self.bonds},
            'stocks': {ticker: 0 for ticker in self.stocks},
            'fx': {ticker: 0 for ticker in self.fx},
        }

        self.risk_factors = RiskFactors(current_date=self._current_date)

    @staticmethod
    def load_prices() -> pd.DataFrame:
        """
        Load prices data for all instruments
        """
        return (
            pd.read_csv(DATA_PATH / 'prices.csv')
            .assign(date=lambda df: pd.to_datetime(df['date']))
            .set_index(['date', 'ticker'])
            .sort_index()
        )

    def move_forward(self, n_days: int = 1):
        """
        Move current date to n_days forward
        """
        delta = pd.Timedelta(days=n_days)
        self._current_date += delta
        self.risk_factors._current_date += delta

    def get_last_price(self, ticker: str) -> float:
        """
        Retrieve last available prices for current date
        """
        last_prices = self._prices[:self._current_date]
        ticker_price = last_prices.loc[(slice(None), ticker), :].iloc[-1].item()
        return ticker_price

    def rebalance_portfolio(self):
        """
        Calculate weights for all instruments
        """
        for instr_type, amount in self._investing_amounts.items():
            for ticker in self.all_instruments[instr_type]:
                price = self.get_last_price(ticker)
                quantity = amount / price
                self.all_instruments[instr_type][ticker] = (
                    int(quantity) if instr_type != 'fx' else round(quantity, 2)
                )

    def is_rebalancing_needed(self) -> bool:
        """
        Return True if portfolio is not balanced
        """
        for instr_type, instruments in self.all_instruments.items():
            target_weight = self.target_weights[instr_type]
            for ticker, quantity in instruments.items():
                last_price = self.get_last_price(ticker)
                real_weight = last_price * quantity / self.total_amount
                if abs(real_weight - target_weight) / target_weight > self._max_relative_error:
                    return True
        return False

    def calc_fair_value(self, fair_prices: PricesDict) -> float:
        """
        Calculate fair value of portfolio based on predicted prices
        """
        return sum(
            fair_prices[ticker] * quantity
            for instr_type, instruments in self.all_instruments.items()
            for ticker, quantity in instruments.items()
        )

    def simulate_fair_value_dist(self, n_days: int = 1, n_sim: int = 1000) -> np.array:
        """
        Simulate fair value of portfolio
        """
        prices_simulations = self.risk_factors.predict_prices(n_days, n_sim)
        dist = np.array(
            [
                self.calc_fair_value(prices_obs)
                for prices_obs in prices_simulations
            ]
        )
        return dist
