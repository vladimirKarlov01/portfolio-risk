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


def find_last_acf_sign_lag(ts, drop_first=True):
    acf, ci = sm.tsa.acf(ts, alpha=0.05)
    first_zero, second_zero = 0, 0
    for l in range(1, len(ci)):
        if (0 > ci[l][0] and 0 < ci[l][1]):
            if first_zero == 0:
                first_zero = l - 1
            else:
                second_zero = l - 1
                break
    return first_zero


def find_last_pacf_sign_lag(ts, drop_first=True):
    acf, ci = sm.tsa.pacf(ts, alpha=0.05)
    first_zero, second_zero = 0, 0
    for l in range(1, len(ci)):
        if (0 > ci[l][0] and 0 < ci[l][1]):
            if first_zero == 0:
                first_zero = l - 1
            else:
                second_zero = l - 1
                break
    return first_zero


def sarima_fit(curr_date, dataframe, column):
    dataframe = dataframe[dataframe.date <= curr_date]  # date.fromisoformat(curr_date)
    max_params = [find_last_acf_sign_lag(dataframe[column].diff().dropna()),
                  find_last_pacf_sign_lag(dataframe[column].diff().dropna()),
                  1, 1]
    print(max_params)
    ps = range(0, max_params[0] + 1)
    d = 1
    qs = range(0, max_params[1] + 1)
    Ps = range(0, max_params[2])
    D = 1
    Qs = range(0, max_params[3])

    from itertools import product

    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    print(len(parameters_list))

    results = []
    best_aic = float("inf")
    warnings.filterwarnings('ignore')
    import statsmodels.api as sm

    for param in tqdm(parameters_list, desc="Fitting SARIMA"):
        # try except нужен, потому что на некоторых наборах параметров модель не обучается
        try:

            model_sm = sm.tsa.statespace.SARIMAX(dataframe[column], order=(param[0], d, param[1]),
                                                 seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)

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

    warnings.filterwarnings('default')

    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    # print(result_table.sort_values(by = 'aic', ascending=True).head())
    return result_table.sort_values(by='aic', ascending=True).reset_index().head(), best_model_sm.get_prediction(0,
                                                                                                                 len(dataframe) + 10).predicted_mean


def calc_mutual_info(y, X):
    """
    Selects top N features by Mutual Information with target
    """
    # determine the mutual information
    mutual_info = mutual_info_regression(X.fillna(0), y)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = X.columns
    return list(mutual_info.sort_values(ascending=False).index)


def select_risk_factors(factor, risk_factors_list, top_factors=4):
    """
    Функция принимает на вход колонку-название актива и список его возможных риск-факторов из словаря
    Отсеиваются факторы с корреляцией между фактором и таргетом < 0.05
    Смотрится взаимная корреляция факторов, если она превышает 0.9, то оставляется один фактор с наибольшей MI с таргетом
    """
    target = all_data[factor].values
    risk_factors = all_data[risk_factors_list]
    drop_list_idx = []
    corr_list = []
    # считаем обычную корреляцию с таргетом
    for i in range(len(risk_factors.columns)):
        corr_with_target = np.corrcoef(x=risk_factors[risk_factors_list[i]].values, y=target)[0][1]
        corr_list.append(corr_with_target)
        if corr_with_target < 0.05:  # дропаем, если меньше 0.05
            drop_list_idx.append(i)

    # ранжируем фичи по MI с таргетом
    top_mutual_information_list = calc_mutual_info(target, risk_factors)

    # смотрим взаимную корреляцию фичей
    mutual_correlation = np.corrcoef(x=risk_factors, rowvar=False)
    a = np.where(mutual_correlation >= 0.9)[0]
    b = np.where(mutual_correlation >= 0.9)[1]
    correlated_features = []
    for i in range(len(a)):
        if a[i] != b[i]:
            correlated_features.append(tuple(sorted([a[i], b[i]])))
    correlated_features = set(correlated_features)  # тут сет пар индексов скоррелированных фичей

    for pair in correlated_features:  # смотрим, какая из фичей из пары на каком месте в ранжированном списке MI
        mi_1 = np.where(np.array(top_mutual_information_list) == risk_factors.columns[pair[0]])
        mi_2 = np.where(np.array(top_mutual_information_list) == risk_factors.columns[pair[1]])

        if mi_1 > mi_2:  # если первый признак из пары менее связан с таргетом (дальше от начала списка MI)
            drop_list_idx.append(pair[0])  # дропаем первый признак
        else:
            drop_list_idx.append(pair[1])

    drop_list_names = list(risk_factors[risk_factors.columns[drop_list_idx]].columns)
    print(drop_list_names)
    return all_data[[factor] + risk_factors_list].drop(drop_list_names, axis=1)
