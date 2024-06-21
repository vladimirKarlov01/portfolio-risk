"""
Utils for portfolio risk management
"""

import numpy as np
import pandas as pd
from typing import List
import scipy.stats as sps
from itertools import combinations


def calculate_posterior_cov(
    factor_list: list,
    prior_corr: pd.DataFrame,
    est_var=list,
):
    """
    склеиваем оценки дисперсий факторов на прогнозе и их исторические ковариации
    """
    assert list(prior_corr.columns)==list(prior_corr.index), 'unexpected corr matrix index'

    pairs = combinations(factor_list, 2)
    LEN = len(factor_list)

    # создаем матрицу и добавляем оценки дисперсий факторов 
    cov_matrix_i = pd.DataFrame(
        np.zeros((LEN, LEN)), 
        columns=factor_list,
        index=factor_list
    )

    cov_matrix_i += np.diag(est_var)

    # убираем лишнее и отсортировываем матрицу корреляций
    prior_corr = prior_corr.loc[factor_list, factor_list]

    # получаем новую ковариацию = corr(x, y) * std_x_sim * std_y_std
    
    std_dict = {name: np.sqrt(var) for name, var in zip(factor_list, est_var)}

    for i, j in pairs:
        
        cov_matrix_i[i][j] = prior_corr[i][j] * std_dict[i] * std_dict[j]
        cov_matrix_i[j][i] = cov_matrix_i[i][j]

    return cov_matrix_i


def sample_multivariate_normal(
    factor_list: list,
    prior_corr: pd.DataFrame,
    est_var: list,
    est_mean: list,
    size: int=100,
    return_array: bool=True,
):
    """
    В функцию подаем имена риск-факторов, средние и дисперсии в одном порядке!
    """
    posterior_cov = calculate_posterior_cov(factor_list, prior_corr, est_var)
    
    sample = sps.multivariate_normal(
        mean=est_mean,
        cov=posterior_cov,
    ).rvs(size=size)

    return sample if return_array else pd.DataFrame(sample, factor_list)


def calculate_simulation_probability(
    factor_list: list,
    series_list: List[np.array], # list of trajectories
    prior_corr: pd.DataFrame,
    est_var: List[np.array], 
    est_mean: List[np.array],
    logging=False,
    logreturn=True,
):  
    """
    Returns smth similar to Log Likelihood s. t. Multivariate distribution
    using models' individual forecasts and historical covariances
    """
    
    assert len(factor_list) == len(series_list), 'check number of factors'
    assert len(set(len(array) for array in series_list)) == 1, 'check num of days in provided series'

    prior_corr = prior_corr.loc[factor_list, factor_list]
    
    series_matrix = np.array(series_list).T
    est_var_matrix = np.array(est_var).T
    est_mean_matrix = np.array(est_mean).T

    LENGTH = series_matrix.shape[0]
    prob_list = [-1] * LENGTH
    
    for step in range(LENGTH):

        posterior_cov = calculate_posterior_cov(
            factor_list,
            prior_corr,
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
    """
    Drops <quantile>% less likely simulations depending
    """
    
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
