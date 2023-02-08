# -*- coding: utf-8 -*-
"""
-> Vasicek model
-> Calibrate to historical rates
-> Simulate N paths for the next 52 weeks
    
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

### INPUTS
interest = pd.read_excel("...\\hist_interest.xlsx")
time_horizon_weeks = 52

### ESTIMATION
# Create columns Delta (diff) and Lag (lagged r)
interest["diff"] = interest["Interest rate"].diff()
interest["lag"] =  interest["Interest rate"].shift(1)

# Fit polynomial of order 1 for -estim for Vasicek parameters
estim = np.polyfit(interest["lag"].dropna(), interest["diff"].dropna(), 1)
a = -estim[0]
b = estim[1] / a

# Interest rate as of now
int_0 = interest["Interest rate"].iloc[-1]

# Interest rate 52 weeks ahead forecast 
int_fcst = int_0 * np.exp(- a * time_horizon_weeks) + b * (1 - np.exp(- a * time_horizon_weeks))

# Calc model fit, residuals to obtain sigma
fitted_values = interest["lag"] * np.exp(- a * 1) + b * (1 - np.exp(- a * 1))
residuals = fitted_values - interest["Interest rate"]
sigma = np.std(residuals)

interest_variance = sigma ** 2 / (2 * a) * (1 - np.exp(- 2 * a * time_horizon_weeks))
interest_vol = np.sqrt(interest_variance)

interest_long_run = b
interest_long_run_vol = sigma / np.sqrt(2 * a)

### SIMULATIONS
# perform N simulations according to Vasicek SDE for the next year
# drt = a(b - rt) dt + sigma dWt
n_paths = 1000
n_steps = 52
t = 52
s = 0
Z = np.random.normal(0.0, 1.0, [n_paths, n_steps])
# Initialize matrix W
W = np.zeros([n_paths,n_paths + 1])

# Determine the time step in the interval [t0, T]
dt_1 = s / float(n_paths)
for i in range(0, n_steps):
    # normalize the samples -> mean = 0, variance = 1
    Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
    W[:,i+1] = W[:,i] + pow(dt_1,0.5) * Z[:,i]
        
# W_s - last column of W
W_s = W[:,-1]
# 1. For every path W(s) perform sub-simulation until time t and calculate
# 2. Calculate the expectation
# time-step from [s,t]
# 1. For every path W(s) perform sub-simulation until time t and calculate
# 2. Calculate the expectation
# time-step from [s,t]
dt_2 = (t - s) / float(n_steps);
W_t  = np.zeros([n_paths, n_steps + 1]);
rt  = np.zeros([n_paths, n_steps + 1]);
rt[:, 0] = int_0

# Store the results
for i in range(0, n_paths):
    # Simulate from s to t
    W_t[:,0] = W_s[i];
    Z = np.random.normal(0.0, 1.0,[n_paths, n_steps])
    for j in range(0, n_steps):
        # normalize the samples -> mean = 0, variance = 1
        Z[:,j] = (Z[:,j] - np.mean(Z[:,j])) / np.std(Z[:,j]);
        # Simulate from s to t
        W_t[:,j + 1] = W_t[:,j] + pow(dt_2, 0.5) * Z[:,j];        
        rt[:, j + 1] = rt[:,j] +  a * (b - rt[:,j]) + sigma * (W_t[:,j + 1] - W_t[:,j])
    
    # Plot the paths
    if i == 0:
        plt.plot(np.linspace(0, s, n_steps + 1),rt[0,:])
        for j in range(0,n_paths):
            plt.plot(np.linspace(s,t,n_steps + 1),rt[j,:])
        plt.xlabel("t")
        plt.ylabel("r(t)")
        plt.grid()
        plt.show()   
        
# Average rate at the end
E_r_t = np.mean(rt[:,-1])
# St Dev
STD_r_t = np.std(rt[:,-1])
