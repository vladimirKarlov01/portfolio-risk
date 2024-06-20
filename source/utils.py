"""
Utils for portfolio risk management
"""

import numpy as np
import statsmodels.api as sm

OPT_PARAMS = {
    'cbr_key_rate': {'a': 0.01, 'b': 8.03, 'sigma': 0.09},
    'pca_cbd': {'a': 0.01, 'b': -0.03, 'sigma': 0.14},
    'sofr': {'a': 0.01, 'b': 1.69, 'sigma': 0.06},
    'ecb_rate': {'a': 0.01, 'b': 1.34, 'sigma': 0.04},
    'usd_rub': {'sigma': 4.351965526837073},
    'eur_rub': {'sigma': 4.673321811786814},
}


def cir_model_opt_params(params, r, dt):
    alpha, theta, sigma = params
    dw = np.random.normal(0, np.sqrt(dt))
    dr = alpha * (theta - r) * dt + sigma * np.sqrt(r) * dw
    return dr


def objective_function(params, r):
    dt = 1 / 365  # daily interval
    cum_error = 0
    for n in range(len(r)):
        cum_error += (r[n] - cir_model_opt_params(params, r[n], dt)) ** 2
    return cum_error


def calc_var(dist: np.array, level: float) -> float:
    """
    Calculate VaR for level given in %
    """
    return np.percentile(dist, q=level)


def calc_es(dist: np.array, level: float) -> float:
    """
    Calculate ES for level given in %
    """
    var = calc_var(dist, level)
    return dist[dist <= var].mean()


factor_final ={
    'gazp': ['gazp', 'su26224_days_before_coupon', 'aluminum', 'brent', 'moex_index',
        'nickel', 'rtsi'],
 'gmkn': ['gmkn', 'su26222_days_before_coupon', 'aluminum', 'moex_index', 'rtsi'],
 'lkoh': ['lkoh', 'ecb_rate', 'aluminum', 'cbr_key_rate', 'eur_rub', 'moex_index',
        'rtsi', 'pca_cbd'],
 'magn': ['magn', 'aluminum', 'eur_rub', 'moex_index', 'rtsi'],
 'mgnt': ['mgnt', 'su26222_days_before_coupon', 'su26221_days_before_coupon',
        'su26218_days_before_coupon', 'ecb_rate', 'aluminum', 'brent',
        'cbr_key_rate', 'eur_rub', 'moex_index', 'nickel', 'rtsi', 'pca_cbd'],
 'moex': ['moex', 'ecb_rate', 'eur_rub', 'moex_index', 'rtsi'],
 'rosn': ['rosn', 'su26222_days_before_coupon', 'su26218_days_before_coupon',
        'ecb_rate', 'aluminum', 'brent', 'cbr_key_rate', 'eur_rub',
        'moex_index', 'rtsi', 'pca_cbd'],
 'rual': ['rual', 'su26218_days_before_coupon', 'aluminum', 'brent',
        'cbr_key_rate', 'moex_index', 'nickel', 'rtsi', 'pca_cbd'],
 'sber': ['sber', 'eur_rub', 'moex_index', 'rtsi'],
 'vtbr': ['vtbr', 'moex_index', 'rtsi'],
 'su26218': ['su26218', 'su26224_days_before_coupon', 'moex_index', 'rtsi'],
 'su26221': ['su26221', 'su26224_days_before_coupon', 'moex_index', 'rtsi'],
 'su26222': ['su26222', 'su26224_days_before_coupon', 'su26222_days_before_coupon',
        'moex_index', 'rtsi'],
 'su26224': ['su26224', 'su26224_days_before_coupon', 'moex_index', 'rtsi'],
 'su26230': ['su26230', 'su26224_days_before_coupon', 'moex_index', 'rtsi']
 }
