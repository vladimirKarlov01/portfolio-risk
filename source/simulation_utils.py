"""
Utils for portfolio risk management
"""

import numpy as np
import pandas as pd
from typing import List
import statsmodels.api as sm
import scipy.stats as sps
from itertools import product
import warnings
from statsmodels.tsa.stattools import adfuller, acf, pacf


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


def calculate_posterior_cov(
    factor_list: list,
    prior_cov: pd.DataFrame,
    est_var=list,
):
    """
    склеиваем оценки дисперсий факторов и их ковариации
    """
    assert list(prior_cov.columns)==list(prior_cov.index), 'unexpected corr matrix index'

    LEN = len(factor_list)

    # создаем матрицу и добавляем оценки дисперсий факторов 
    res_cov = np.zeros(shape=(LEN, LEN)) + np.diag(est_var)

    # убираем лишнее и отсортировываем матрицу
    prior_cov = prior_cov.loc[factor_list, factor_list].to_numpy()

    # оставляем только ковариации
    only_cov = prior_cov - np.diag(np.diagonal(prior_cov))

    return res_cov + only_cov


def sample_multivariate_normal(
    factor_list: list,
    prior_cov: pd.DataFrame,
    est_var: list,
    est_mean: list,
    size=100,
    return_array=True,
):
    """
    В функцию подаем имена риск-факторов, средние и дисперсии в одном порядке!
    """
    posterior_cov = calculate_posterior_cov(factor_list, prior_cov, est_var)
    
    sample = sps.multivariate_normal(
        mean=est_mean,
        cov=posterior_cov,
    ).rvs(size=size)

    return sample if return_array else pd.DataFrame(sample, factor_list)


def calculate_simulation_probability(
    factor_list: list,
    series_list: List[np.array],
    prior_cov: pd.DataFrame,
    est_var: List[np.array],
    est_mean: List[np.array],
    logging=False,
    logreturn=True,
):  
    """
    Returns smth similar to Log Likelihood s. t. Multivariate distribution
    using models individual forecasts and historical covariances
    """
    
    assert len(factor_list) == len(series_list), 'check number of factors'
    assert len(set(len(array) for array in series_list)) == 1, 'check num of days in provided series'

    prior_cov = prior_cov.loc[factor_list, factor_list]
    
    series_matrix = np.array(series_list).T
    est_var_matrix = np.array(est_var).T
    est_mean_matrix = np.array(est_mean).T

    LENGTH = series_matrix.shape[0]
    prob_list = [-1] * LENGTH
    
    for step in range(LENGTH):

        posterior_cov = calculate_posterior_cov(
            factor_list,
            prior_cov,
            est_var_matrix[step, :]
        )

        dist = sps.multivariate_normal(
            mean=est_mean_matrix[step, :],
            cov=posterior_cov
        )

        prob_list[step] = dist.pdf(series_matrix[step, :])

    if logging:
        print(prob_list)
        
    return np.sum(np.log(prob_list)) if logreturn else np.prod(prob_list)


def simulated_series_selection(
    factor_list: list,
    simulations: List[List[np.array]], # M iterations for N days for K factors
    prior_cov: pd.DataFrame,
    est_var: List[np.array],
    est_mean: List[np.array],
    threshold=None,
    quantile=0.05,
):
    
    NUM_SIM = len(simulations)
    probs = [0] * NUM_SIM
    
    for sim in range(NUM_SIM):
        
        probs[sim] = calculate_simulation_probability(
            factor_list,
            simulations[sim],
            prior_cov,
            est_var,
            est_mean,
            logging=False,
            logreturn=True
        )

    threshold = np.quantile(probs, quantile)
    
    for sim in range(NUM_SIM):
        if probs[sim] >= threshold:
            yield simulations[sim]


def find_last_acf_sign_lag(ts, drop_first = True):
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
    
def find_last_pacf_sign_lag(ts, drop_first = True):
    _, ci = sm.tsa.pacf(ts, alpha=0.01)
    first_zero, second_zero = 0, 0
    for l in range(1, len(ci)):
        if (0 >= ci[l][0] and 0 <= ci[l][1]):
            if first_zero ==0:
                first_zero = l - 1
            else: 
                second_zero = l - 1
                break
    return first_zero   


def fit_best_sarima_model(
    series: pd.Series, 
    seasonal = 0,
    ps = None,
    d = None,
    qs = None,
    Ps = None,
    D= None,
    Qs = None,
):
    max_params = [
        find_last_pacf_sign_lag(series.diff().dropna()), 
        find_last_acf_sign_lag(series.diff().dropna()),
    1, 1]
    
    # print(max_params)
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
        #try except нужен, потому что на некоторых наборах параметров модель не обучается
        order_list = (*param[:3],)
        seasonal_list = (*param[3:7],) if seasonal else (0, 0, 0, 0)
        
        try:
            model_sm = sm.tsa.SARIMAX(
                series,
                order=order_list, 
                seasonal_order=seasonal_list,
            ).fit(disp=-1)
            
        #выводим параметры, на которых модель не обучается и переходим к следующему набору
        except ValueError:
            print('wrong parameters:', param)
            continue
        aic = model_sm.aic
        
        #сохраняем лучшую модель, aic, параметры
        if aic < best_aic:
            best_model = model_sm
            best_aic = aic
            best_param = param
            
        results.append([param, model_sm.aic])
    
    return best_model, best_param  