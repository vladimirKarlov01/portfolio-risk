"""
Utils for portfolio risk management
"""

import numpy as np

OPT_PARAMS = {
    'cbr_key_rate': {'a': 0.01, 'b': 7.78, 'sigma': 0.09},
    'pca_cbd': {'a': 0.01, 'b': -0.03, 'sigma': 0.14},
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
