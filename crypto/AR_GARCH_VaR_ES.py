# -*- coding: utf-8 -*-
"""
Assessing the Risk Characteristics of the Cryptocurrency Market: A GARCH-EVT-Copula Approach
https://www.mdpi.com/1911-8074/15/8/346
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA as ARIMA # ARIMA Models
import scipy.optimize as opt
from scipy import stats

### INPUT (from Kaggle)
dir_ = r"...\\crypto\\data"
os.chdir(dir_)
# Get all files (crypto gistoricals) from the directory
files = os.listdir(dir_)
# Put BTC as first file by switching with the first file
a, b = files.index('coin_Bitcoin.csv'), 0
files[b], files[a] = files[a], files[b]

inputs = []
for f in files:
    inputs.append(pd.read_csv(f))
# Merge the input files by Date (Close column, rename to crypto name)
merged = pd.DataFrame()
merged["Date"] = inputs[0]["Date"]
merged.index = inputs[0]["Date"]
for i in inputs:
    i.index = i.Date
    merged = pd.merge(merged, i[["Date", "Close"]], on = "Date", how = "left")
    merged.columns = list(merged.columns)[:-1] + [i["Symbol"][0]]

###
# LOG GROWTHS
for c in merged.columns:
    if c != "Date":
        merged[c + "_lgwth"] = np.log(merged[c] / merged[c].shift(1))
        
###
# PLOTS
merged["BTC"].plot()
merged["BTC_lgwth"].plot()
merged["ETH"].plot()
merged["ETH_lgwth"].plot()
merged["EOS_lgwth"].plot()

# SUMMARY OF DATA ON LOG GROWTH
merged_to_descr = merged[[c for c in merged.columns if "lgwth" in c]]
descr_ = merged_to_descr.describe()
descr_.loc['var'] = merged_to_descr.var().tolist()
descr_.loc['skew'] = merged_to_descr.skew().tolist()
descr_.loc['kurt'] = merged_to_descr.kurtosis().tolist()
descr_.loc['var_95'] = [np.nanpercentile(merged_to_descr[x], 5) for x in merged_to_descr.columns]
descr_.loc['var_99'] = [np.nanpercentile(merged_to_descr[x], 1) for x in merged_to_descr.columns]
descr_.loc['ES_95'] = [np.mean(merged_to_descr[x][merged_to_descr[x] < np.nanpercentile(merged_to_descr[x], 5)]) for x in merged_to_descr.columns]
descr_.loc['ES_99'] = [np.mean(merged_to_descr[x][merged_to_descr[x] < np.nanpercentile(merged_to_descr[x], 1)]) for x in merged_to_descr.columns]
# Correlation with BTC and ETH
descr_.loc['BTC_corr'] = [merged_to_descr[["BTC_lgwth", c]].corr()['BTC_lgwth'][c] if c != "BTC_lgwth" else 1 for c in merged_to_descr.columns]
descr_.loc['ETH_corr'] = [merged_to_descr[["ETH_lgwth", c]].corr()['ETH_lgwth'][c] if c != "ETH_lgwth" else 1 for c in merged_to_descr.columns]
print(descr_)

# TURNING POINT TEST
#Test statistic
#We say i is a turning point if the vector X1, X2, ..., Xi, ..., Xn is not monotonic at index i. The number of turning points is the number of maxima and minima in the series.[4]
#Letting T be the number of turning points then for large n, T is approximately normally distributed with mean (2n − 4)/3 and variance (16n − 29)/90. The test statistic[7]
# test is failed for all currencies except for BTC and LTC
turn_point_stats = []
for c in merged_to_descr.columns:
    no_turn_points = 0 
    for i in range(1, merged_to_descr[c].shape[0] - 1):
        if (merged_to_descr[c][i] > merged_to_descr[c][i - 1]  and merged_to_descr[c][i] > merged_to_descr[c][i + 1]) or \
           (merged_to_descr[c][i] < merged_to_descr[c][i - 1]  and merged_to_descr[c][i] < merged_to_descr[c][i + 1]):
              no_turn_points += 1
    # https://en.wikipedia.org/wiki/Turning_point_test
    n = merged_to_descr["BTC_lgwth"].shape[0]
    no_turn_points_stat = abs(no_turn_points - (2 * n - 4) / 3) / np.sqrt(((16 * n - 29) / 90)) 
    turn_point_stats.append(no_turn_points_stat)
descr_.loc["turn_point_stat_0.05"] = [x > 1.64 for x in turn_point_stats]


# AR / GARCH model estimation
var_95_estim = []
var_99_estim = []
es_95_estim = []
es_99_estim = []
# garch(1,1)
def garch_filter(alpha0, alpha1, beta, eps):
    """
    GARCH recursion
    """
    iT = len(eps)
    sigma_2 = np.zeros(iT)
    
    for i in range(iT):
        if i == 0:
            # Variance at the start equals unconditional variance
            sigma_2[i] = alpha0 / (1 - alpha1 - beta)
        else:
            sigma_2[i] = alpha0 + alpha1 * eps[i - 1] ** 2 + beta * sigma_2[i - 1]
    
    return sigma_2
            
def garch_loglike(vP, eps):
    """
    GARCH loglikelihood function
    """
    iT = len(eps)
    alpha0 = vP[0]
    alpha1 = vP[1]
    beta = vP[2]
    
    sigma_2 = garch_filter(alpha0, alpha1, beta, eps)
    
    logL = -np.sum(-np.log(sigma_2) - eps ** 2 / sigma_2)
    
    return logL
# The dynamics of the conditional mean μt and the conditional volatility σt are to be modelled by an AR(1) and a GARCH(1, 1) model, 
# whereby the process could be extended to further specialised models. 
# AutoRegressive (AR) models predict future outcomes based on the p past observations (lags). 
# An AR model with one lag, denoted as AR(1), uses the most recent observation to predict a future outcome.
# mu ~ AR(10)
# TBD: run for all cyrptos
for c in ['BTC_lgwth', "ETH_lgwth"]:#merged_to_descr.columns:
    print(c)
    Y = merged_to_descr[c][1:]
    Y.index = merged["Date"][1:]
    Y = Y.dropna()
    # acf / pacf
    plot_acf(Y, lags=15)
    
    model_AR = ARIMA(endog= Y, order= (1,0,0))
    model_AR = model_AR.fit()
    model_AR.summary()
    # Calculate fitted values
    mu_hat = pd.DataFrame(model_AR.predict())
    mu_hat["Date"] = mu_hat.index
    
    # GARCH
    # maximize
    cons = ({'type' : 'ineq', 'func' : lambda x: np.array(x)})
    vP0 = (0.1, 0.05, 0.92)
    
    res = opt.minimize(garch_loglike, vP0, args = (Y),
                       bounds = ((0.0001,None), (0.0001, None), (0.0001, None)),
                       options ={'disp': True})
    
    alpha0_est = res.x[0]
    alpha1_est = res.x[1]
    beta_est = res.x[2]
    sigma2 = pd.DataFrame(garch_filter(alpha0_est, alpha1_est, beta_est, Y))
    sigma2.index = Y.index
    sigma = np.sqrt(sigma2)
    sigma = sigma.dropna()
    
    sigma.plot(rot = 45)
    sigma["Date"] = sigma.index
    sigma.columns = ["sigma", "Date"]
    # Calculate standardized residuals:
    # z(t) = (x(t) - mu_hat(t)) / sigma_hat(t)
    # Take the inner product of mu_hat and sigma_hat
    fitted_vals = pd.merge(mu_hat, sigma[["sigma"]], left_index = True, right_index = True)
    fitted_vals.columns = ["mu_hat", "Date", "sigma_hat"]
    test_vals = pd.merge(merged[[c]][1:], fitted_vals, left_index = True, right_index = True)
    test_vals["z"] = (test_vals[c] - test_vals["mu_hat"]) / test_vals["sigma_hat"]
    # Check if iid norm distrib
    # Close to 0
    test_vals["z"].mean()
    # Clsoe to 1
    test_vals["z"].var()
    # skewed to the left
    test_vals["z"].skew()
    # Fat tail
    test_vals["z"].kurtosis()
    # acf
    plot_acf(test_vals["z"])
    # test passed
    sm.stats.acorr_ljungbox(test_vals["z"], lags=[10])
    
    # VaR / ES at t+1
    # VaR_hat(t+1) = mu_hat(t+1) + sigma_hat(t+1) * z_q
    # ES_hat(t+1) = mu_hat(t+1) + sigma_hat(t+1) * E(Z | Z > z_q)
    VaR_hat_95 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.percentile(test_vals["z"], 5)
    VaR_hat_99 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.percentile(test_vals["z"], 1)
    VaR_hat_99_9 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.percentile(test_vals["z"], 0.1)
    ES_hat_95 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.mean(test_vals["z"][test_vals["z"] < np.percentile(test_vals["z"], 5)])
    ES_hat_99 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.mean(test_vals["z"][test_vals["z"] < np.percentile(test_vals["z"], 1)])
    ES_hat_99_9 = test_vals["mu_hat"][-1] + test_vals["sigma_hat"][-1] * np.mean(test_vals["z"][test_vals["z"] < np.percentile(test_vals["z"], 0.1)])
    # Append results
    var_95_estim.append(VaR_hat_95)
    var_99_estim.append(VaR_hat_99)
    es_95_estim.append(ES_hat_95)
    es_99_estim.append(ES_hat_99)
    
    
### PORTFOLIO
# Start with BTC: 70%, ETH: 30%
# Later include other cryptos and calc weights based on current market cap
w = [0.7, 0.3]
# Calc cum return in the last 252 trading days
port_hist_ret = w[0] * merged["BTC_lgwth"] + w[1] * merged["ETH_lgwth"]
port_hist_ret = port_hist_ret.dropna()
port_hist_cum_ret = np.cumprod(np.exp(port_hist_ret[-252:]))
port_hist_cum_ret.plot(rot=45)
#  fit a t-Student Copula to the daily logarithmic returns to estimate their joint density function
