"""
Utils for portfolio risk management
"""

import numpy as np


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
