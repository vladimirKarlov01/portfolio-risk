"""
Utils for portfolio risk management
"""

import numpy as np

OPT_PARAMS = {
    'year_1': {'a': 0.01, 'b': 7.73, 'sigma': 0.07},
    'year_3': {'a': 0.01, 'b': 8.2, 'sigma': 0.06},
    'year_5': {'a': 0.01, 'b': 7.99, 'sigma': 0.06},
    'year_10': {'a': 0.01, 'b': 8.75, 'sigma': 0.06},
    'year_15': {'a': 0.01, 'b': 9.04, 'sigma': 0.05},
    'year_20': {'a': 0.01, 'b': 9.46, 'sigma': 0.05},
    'cbr_key_rate': {'a': 0.01, 'b': 8.06, 'sigma': 0.09},
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
