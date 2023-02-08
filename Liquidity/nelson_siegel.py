# -*- coding: utf-8 -*-
"""
-> Nelson Siegel yield curve model
-> Fit to actual yield curve (using Nelder Mead algorithm)
-> Plot the YC
    
"""
import copy as copylib
import numpy as np
import os
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import enum
import scipy.stats as st
import datetime
from datetime import datetime
from scipy.optimize import minimize

### INPUTS
prices = pd.read_excel("...\\Nelson_Siegel_Prices.xlsx")
time_horizon_weeks = 52
max_mat = 30

### FUNCTIONS
def YC_calc(b0_ = 0.02, b1_ = 0.005, b2_ = -0.5, lambda_ = 10):
    yield_curve = [b0_ + b1_*(1 - np.exp(-lambda_ * (x + 1)))/(lambda_ * (x + 1)) + b2_ * ((1 - np.exp(- lambda_ * (x + 1)))/( lambda_ * (x + 1)) - np.exp(- lambda_* (x + 1))) for x in range(max_mat)]
    yield_curve = pd.DataFrame(yield_curve)
    yield_curve.index = range(1, max_mat + 1)
    return yield_curve
    
### FITTING NELSON-SIEGEL MODEL
# Initial values of NS model
def fit_ns(b0_ = 0.02, b1_ = 0.005, b2_ = -0.5, lambda_ = 10):

    ### ESTIMATION
    # Calculate initial yield curve
    yield_curve = YC_calc(b0_, b1_, b2_, lambda_)
    
    # Calculate Value of bond
    prices["Value"] = 0#
    for i in range(len(prices)):
        prices["Value"][i] = 100 / (1 + yield_curve[0][i + 1]) ** prices["Maturity"][i] + \
                            np.sum([prices["Coupon"][i] / (1 + yield_curve[0][j + 1]) ** j for j in range(i)])
    
    # Calculate mispricing
    mispricing = np.sum((prices["Value"] - prices["Price"]) ** 2)
    return mispricing

# Optimize
x0 = np.array([0.02, 0.005, -0.5, 100])
res = minimize(fit_ns, x0, method='nelder-mead',options={'xatol': 1e-8, 'disp': True})
fit_ns(x0)
fit_ns(res.x)

# Plot the YC
b0_, b1_, b2_, lambda_ = res.x

yield_curve = YC_calc(b0_, b1_, b2_, lambda_)

plt.plot(yield_curve)
plt.xlabel("t")
plt.ylabel("YC(t)")
plt.grid()
plt.show()   

# Change lambda to 10
lambda_ = 10

yield_curve = YC_calc(b0_, b1_, b2_, lambda_)

plt.plot(yield_curve)
plt.xlabel("t")
plt.ylabel("YC(t)")
plt.grid()
plt.show()  
