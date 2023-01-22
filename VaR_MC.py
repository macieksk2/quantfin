import numpy as np
import numpy.matlib
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
import math as m
import random as r

###############################################################################
# I simulate the equity price paths and portfolio value
###############################################################################
###############################################################################
# I.1 INPUTS
###############################################################################
# inital share prices (five stocks)
S_0 = np.array([[120],[40],[60],[100],[150]])
# Equity volatilities
sigma = np.array([[0.25],[0.2],[0.3],[0.5],[0.6]])
# correlations in between stocks
pho_matrix = np.array([[1  ,  0.2, 0.4, 0.2, 0],
                       [0.2, 1   , 0.6, 0.4, 0.7],
                       [0.4, 0.6 , 1  , 0.2, 0.5], 
                       [0.2, 0.4 , 0.2, 1  , 0.2],
                       [0, 0.7 , 0.5, 0.2, 1]])
# Cholesky Decomposition
L = np.linalg.cholesky(pho_matrix)
# rf rate
r = 0.05
# time = 1 year
T = 1
# number of simulations = 100000
no_sim = 100000
# alpha = 1%
alpha = 0.01
###############################################################################
# I.2 FUNCTIONS
###############################################################################
def s_price_term(S_0, risk_free_rate, sigma, Z, T):
    """
    Generates the terminal share price with iid normal Z
    """
    return S_0 * np.exp((risk_free_rate - sigma ** 2/2) * T + sigma*np.sqrt(T) * Z)

def s_path(S_0, risk_free_rate, sigma, Z, dT):
    """
    Generates the terminal share price with iid normal Z
    """
    return S_0 * np.exp(np.cumsum((risk_free_rate - sigma ** 2/2) * dT + sigma*np.sqrt(dT) * Z,1))
###############################################################################
# I.3 CALCULATION
###############################################################################
np.random.seed(0)
# current portfolio value
portf_val_curr = np.sum(S_0)
# Creating N simulations for future portfolio values
Z = np.matmul(L, norm.rvs(size= [len(S_0), no_sim]))
portf_val_ft = np.sum(s_price_term(S_0, r, sigma, Z, T), axis=0)
# 1. calculating the expected returns | simulated paths
# 2. sort this list
# 3. calculate VaR as the negative of the alpha (1%)
# 4. calculating portfolio returns
portf_ret = (portf_val_ft - portf_val_curr) / portf_val_curr
# sort
portf_ret = np.sort(portf_ret)
# calculate VaR
sim_VaR = -portf_ret[int (np.floor(alpha*no_sim))-1]
print("Simulated VaR: ", sim_VaR, " with alpha = ", alpha)
###############################################################################
# I.4 PLOT
###############################################################################
plt.hist(portf_ret, bins=30, ec="black")
plt.show()
###############################################################################
# II Applying Historical Simulation to Estimate VaR
###############################################################################
# II.1 INPUTS
###############################################################################
# Historical simulation
s0 = 25
sigma = 0.1
r = 0.1
T = 1
alpha = 0.01
no_sim = 10000
dT = 1/365
no_years = 5
no_days = 365
###############################################################################
# II.2 CALCULATION
###############################################################################
# generate synthetic share data
Z_hst = norm.rvs(size = [len(S_0), no_years * no_days])
corr_Z = np.transpose(np.matmul(L, Z_hst))
p_path = s_path(s0, r, sigma, corr_Z, dT)
# determine the historical portfolio value and returns
# current portfolio value as the sum of the most recent share price
hist_s0 = p_path[-1]
portf_val_h = np.sum(hist_s0)
# initialize a vector to capture simulated portfolio returns
portf_ret_h = [None] * no_sim
# determining historical log returns
log_ret_h = np.log(p_path[1:]) - np.log(p_path[0: -1])
# 1. sample returns
# 2. calculate historical VaR estimate
# 3. apply historicals to project future returns
for i in range(no_sim):
    sample_ = uniform.rvs(size = no_days) * (len(p_path)-1)
    sample_ = [int(x) for x in sample_]
    shr_ret = log_ret_h[sample_]
    s_term = hist_s0 * np.exp(np.sum(shr_ret, axis=0))
    portf_ret_h[i] = (np.sum(s_term) - portf_val_h)/portf_val_h
# Sorting portfolio returns
portf_ret_h = np.sort(portf_ret_h)
# Historical VaR estimate
hist_VaR = -portf_ret_h[int(np.floor(alpha * no_sim))-1]
print("historical VaR estimate: ", hist_VaR, " with alpha = ", alpha)
###############################################################################
# II.3 PLOT
###############################################################################
plt.figure(figsize=(15,8))
plt.title("distribution of historical returns")
plt.xlabel("portfolio returns")
plt.ylabel("times calculated")
plt.hist(portf_ret_h, bins=30, ec = "black")
plt.show()
