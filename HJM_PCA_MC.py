# -*- coding: utf-8 -*-
"""
-> Heath Jarrow Morton Model (HJM)
-> PCA
-> Monte Carlo Simulation
-> Caplet pricing
"""
from mpl_toolkits.mplot3d import Axes3D
import copy as copylib
from progressbar import *
import numpy as np
import os
import pandas as pd
import pylab
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (16, 4.5)
np.random.seed(0)
# Load Historical Data
# Loads historical short-rate curves from the file. 
# represents instantenous forward rate for period 
# This rates can be obtained from prices of zero-coupons bond
 
#### INPUT
directory_     = 'C:\\Users\\macie\\OneDrive\\Desktop\\interviews\\Standard Chartered'
data_filename_ = 'hjm_data.csv'

class InputHandling():
    def __init__(self, dir_, data_filename_):
        self.dir_ = dir_
        self.data_filename_ = data_filename_
        
    def read_data(self):
        os.chdir(self.dir_)
        # Convert interest rates to %
        dataframe =  pd.read_csv(self.data_filename_).set_index('time') / 100 
        pd.options.display.max_rows = 10
        display(dataframe)
        return dataframe
    
    def tenors(self):
        dataframe = self.read_data()
        hist_timeline = list(dataframe.index)
        tenors = [eval(x) for x in dataframe.columns]
        return tenors
    
    def hist_rates(self):
        dataframe = self.read_data()
        hist_timeline = list(dataframe.index)
        hist_rates = np.matrix(dataframe)
        return hist_rates
    
    def plot_data(self):
        hist_rates =  self.hist_rates()
        tenors = self.tenors()
        plt.figure(1)
        plt.grid()
        plt.plot(hist_rates) 
        plt.xlabel(r'Time $t$')
        plt.title(r'Historical $f(t,\tau)$ by $t$')
        plt.show()
        
        plt.figure(1)
        plt.grid()
        plt.plot(tenors, hist_rates.transpose())
        plt.xlabel(r'Tenor $\tau$')
        plt.title(r'Historical $f(t,\tau)$ by $\tau$')
        plt.show()

        # Differentiate historical rates
        # Differentiate matrix of historical rates by 
        diff_rates = np.diff(hist_rates, axis=0)
        assert(hist_rates.shape[1]==diff_rates.shape[1])
        plt.plot(diff_rates), plt.xlabel(r'Time $t$'), plt.title(r'$df(t,\tau)$ by $t$');
        
    def diff_data(self):
        diff_rates = np.diff(self.hist_rates(), axis=0)
        return diff_rates

data_interest = InputHandling(dir_ = directory_, data_filename_ = data_filename_)
tenors = data_interest.tenors()
hist_rates = data_interest.hist_rates()
data_interest.plot_data()
data_interest_ = data_interest.diff_data()

### PCA
# Principal component analysis
# Extract principal components from the 
# Calculate covariance matrix
class PCA_():
    def __init__(self, data_interest_):
        self.data_interest_ = data_interest_

    def vol(self):
        diff_rates = self.data_interest_
        sigma = np.cov(diff_rates.transpose())
        print("Sigma shape : " + str(sigma.shape))
        # Source data are daily rates, therefore annualize covariance matrix
        sigma *= 252
        # Calculate eigenvalues and eigenvectors
        eigval, eigvec = np.linalg.eig(sigma)
        eigvec= np.matrix(eigvec)
        assert type(eigval) == np.ndarray
        assert type(eigvec) == np.matrix
        print(eigval)
        return eigval, eigvec

    # Determine principal components
    # Select eigen vectors with highest eigenvalues. 
    # Link between tenors and eigenvectors is not guaranteed.
    def pca_calc(self):
        factors=3
        # highest principal component first in the array
        eigval, eigvec = self.vol()
        index_eigvec = list(reversed(eigval.argsort()))[0:factors]   
        princ_eigval = np.array([eigval[i] for i in index_eigvec])
        princ_comp = np.hstack([eigvec[:,i] for i in index_eigvec])
        print("Principal eigenvalues")
        print(princ_eigval)
        print("Principal eigenvectors")
        print(princ_comp)
        plt.plot(princ_comp, marker='.'), plt.title('Principal components'), plt.xlabel(r'Time $t$');
        return princ_eigval, princ_comp

pca_ = PCA_(data_interest_ = data_interest_)
pca_.vol()
princ_eigval_, princ_comp_ = pca_.pca_calc()

### VOLATILITY
# Calculate discretized volatility function from principal components
class Vol_():
    def __init__(self, princ_eigval_, princ_comp_):
        self.princ_eigval_ = princ_eigval_
        self.princ_comp_ = princ_comp_
        
    def discr_vol(self):
        sqrt_eigval = np.matrix(self.princ_eigval_ ** .5)
        # resize matrix (1,factors) to (n, factors)
        tmp_m = np.vstack([sqrt_eigval for i in range(self.princ_comp_.shape[0])])  
        # multiply matrice element-wise
        vols = np.multiply(tmp_m, self.princ_comp_) 
        print('vols shape: ' + str(vols.shape))
        plt.plot(vols, marker='.'), plt.xlabel(r'Time $t$'), \
                 plt.ylabel(r'Volatility $\sigma$'), plt.title('Discretized volatilities');
        plt.show()
        return vols

v_ = Vol_(princ_eigval_, princ_comp_)
vols = v_.discr_vol()

# Volatility Fitting
# We need to fit discretized volatility functions from the previous step using cubic interpolators. 
# The reason is that these interpolators will be later integrated numerically 
# in order to calculate risk-neutral drift.

# Fit Volatility Functions 
# from discretized versions
# Fitting is done using cubic spline
def get_matrix_column(mat, i):
    return np.array(mat[:,i].flatten())[0]

class PolynomialInterpolator:
    def __init__(self, params):
        assert type(params) == np.ndarray
        self.params = params
    def calc(self, x):
        n = len(self.params)
        C = self.params
        X = np.array([x ** i for i in reversed(range(n))])
        return sum(np.multiply(X, C))
# We will approximate the first principal component with interpolator with 0 degree (straight line). 
# This approximates well parallel movements of interest rates.

fitted_vols = []
# 2nd and 3rd principal component will be approximated using cubic interpolator with degree 3.

def fit_volatility(i, degree, title):
    vol = get_matrix_column(vols, i)
    fitted_vol = PolynomialInterpolator(np.polyfit(tenors, vol, degree))    
    plt.plot(tenors, vol, marker='.', label='Discretized volatility')
    plt.plot(tenors, [fitted_vol.calc(x) for x in tenors], label='Fitted volatility')
    plt.title(title), plt.xlabel(r'Time $t$'), plt.legend();
    fitted_vols.append(fitted_vol)
    
plt.subplot(1, 3, 1), fit_volatility(0, 0, '1st component');
plt.subplot(1, 3, 2), fit_volatility(1, 3, '2nd component');
plt.subplot(1, 3, 3), fit_volatility(2, 3, '3rd component');
### MONTE CARLO
# Monte Carlo Simulation
# Define function for numeric integration
# We will use trapezoidal rule:
def integrate(f, x0, x1, dx):
    n = (x1-x0)/dx+1
    out = 0
    for i, x in enumerate(np.linspace(x0, x1, int(n))):
        if i==0 or i==n-1:
            out += 0.5 * f(x)
        else:
            # not adjusted by *0.5 because of repeating terms x1...xn-1 - see trapezoidal rule
            out += f(x)  
    out *= dx
    return out
# Discretize fitted volatilities for the purpose of drift calculation
mc_tenors = np.linspace(0,25,51)
# Discretize fitted volfuncs for the purpose of monte carlo simulation
mc_vols = np.matrix([[fitted_vol.calc(tenor) for tenor in mc_tenors] for fitted_vol in fitted_vols]).transpose()
plt.plot(mc_tenors, mc_vols, marker='.'), plt.xlabel(r'Time $t$'), plt.title('Volatilities');

# Drift calculation is calculated using numerical integration over fitted volatility functions (No Mursiela parameterisation for now)
def m(tau, fitted_vols):
    #This funciton carries out integration for all principal factors. 
    #It uses the fact that volatility is function of time in HJM model
    out = 0.
    for fitted_vol in fitted_vols:
        assert isinstance(fitted_vol, PolynomialInterpolator)
        out += integrate(fitted_vol.calc, 0, tau, 0.01) * fitted_vol.calc(tau)
    return out
mc_drift = np.array([m(tau, fitted_vols) for tau in mc_tenors])
plt.plot(mc_drift, marker='.'), plt.xlabel(r'Time $t$'), plt.title('Risk-neutral drift');
# Perform Monte Carlo Simulation (one path)
# Few formulas
# Mursiela parameterisation

# The multi-factor HJM framework is implemented with this SDE:
# We simulate by 
# where Musiela HJM SDE is 
# today's instantenous forward rates 
curve_spot = np.array(hist_rates[-1,:].flatten())[0]
plt.plot(mc_tenors, curve_spot.transpose(), marker='.'), plt.ylabel('$f(t_0,T)$'), plt.xlabel("$T$");

def simulation(f, tenors, drift, vols, timeline):
    assert type(tenors)==np.ndarray
    assert type(f)==np.ndarray
    assert type(drift)==np.ndarray
    assert type(timeline)==np.ndarray
    assert len(f)==len(tenors)
    # 3 rows, T columns
    vols = np.array(vols.transpose())  
    len_tenors = len(tenors)
    len_vols = len(vols)
    yield timeline[0], copylib.copy(f)
    for it in range(1, len(timeline)):
        t = timeline[it]
        dt = t - timeline[it-1]
        sqrt_dt = np.sqrt(dt)
        fprev = f
        f = copylib.copy(f)
        random_numbers = [np.random.normal() for i in range(len_vols)]
        for iT in range(len_tenors):
            val = fprev[iT] + drift[iT] * dt
            #
            sum = 0
            for iVol, vol in enumerate(vols):
                sum += vol[iT] * random_numbers[iVol]
            val += sum * sqrt_dt
            #
            # if we can't take right difference, take left difference
            iT1 = iT+1 if iT<len_tenors-1 else iT-1   
            dfdT = (fprev[iT1] - fprev[iT]) / (iT1 - iT)
            val += dfdT * dt
            #
            f[iT] = val
        yield t,f
proj_rates = []
proj_timeline = np.linspace(0,5,500)
for i, (t, f) in enumerate(simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline)):
    print(i)
    proj_rates.append(f)
proj_rates = np.matrix(proj_rates)
plt.plot(proj_timeline.transpose(), proj_rates), plt.xlabel(r'Time $t$'), plt.ylabel(r'Rate $f(t,\tau)$');
plt.title(r'Simulated $f(t,\tau)$ by $t$'), plt.show();
plt.plot(mc_tenors, proj_rates.transpose()), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Rate $f(t,\tau)$');
plt.title(r'Simulated $f(t,\tau)$ by $\tau$'), plt.show();

# Pricing of trade using Monte Carlo
# Define integrator for instatenous rates
# Integrating 
#  over 
#  discrete samples:
 
# Simple LIBOR forward rate (e.g. 3M libor)
class Integrator:
    def __init__(self, x0, x1):
        assert x0 < x1
        self.sum, self.n, self.x0, self.x1= 0, 0, x0, x1
    def add(self, value):
        self.sum += value
        self.n += 1
    def get_integral(self):
        return (self.x1 - self.x0) * self.sum / self.n
### CAPLET
# Price caplet
# Caplet with strike 
# , expiring in 
# , maturing in 
# , notional 

t_exp, t_mat = 1., 2.
K, notional = .03, 1e6
n_simulations, n_timesteps = 500, 50

proj_timeline = np.linspace(0,t_mat, n_timesteps)
simulated_forecast_rates = []
simulated_df = []
simulated_pvs = []
pv_convergence_process = []
for i in range(0, n_simulations):
    print(i)
    rate_forecast = None                    # Forecast rate between t_exp and t_mat for this path
    rate_discount = Integrator(0, t_exp)      # cont.compounded discount rate for this path
    for t, curve_fwd in simulation(curve_spot, mc_tenors, mc_drift, mc_vols, proj_timeline):
        f_t_0 = np.interp(0., mc_tenors, curve_fwd)  # rate $f_t^0$
        rate_discount.add(f_t_0)
        if t>=t_exp and rate_forecast is None:  # t is expiration time
            Tau = t_mat-t_exp
            rate_forecast = Integrator(0, Tau) # integrate all inst.fwd.rates from 0 till 1Y tenor to get 1Y spot rate
            for s in np.linspace(0, Tau, 15): # $\int_0^T f(t,s)ds$
                f_texp_s =  np.interp(s, mc_tenors, curve_fwd)
                rate_forecast.add(f_texp_s) 
            rate_forecast = rate_forecast.get_integral()
    plt.plot(mc_tenors, curve_fwd), plt.xlabel(r'Tenor $\tau$'), plt.ylabel(r'Rate $f(t_{exp},\tau)$');   # Plot forward curve at expiration
    df = np.exp(-rate_discount.get_integral())     # Discount factor
    simulated_forecast_rates.append(rate_forecast)
    simulated_df.append(df)
    pv = max(0, rate_forecast - K) * (t_mat-t_exp) * notional * df
    simulated_pvs.append(pv)
    pv_convergence_process.append(np.average(simulated_pvs))
plt.show()
#
plt.scatter(simulated_df, simulated_forecast_rates), plt.xlabel('Discount Factor'), plt.ylabel('Forecast Rate')
plt.show()
#
plt.plot(pv_convergence_process[10:]), plt.title('Convergence of PV'), plt.xlabel("Simulations"), plt.ylabel("V");
print("Final value: %f" % pv_convergence_process[-1])
