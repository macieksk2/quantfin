#%%
"""
Hull-White Model
MC simulation
Comparison of market and MC Zero Coupon Bond prices
"""

import numpy as np
import matplotlib.pyplot as plt
##### INPUT
n_paths   = 25000
n_steps   = 25
T         = 40
r         = 0.1
# HW parameters
lambd     = 0.02
eta       = 0.02
# Lambda statement defining Zero Coupon Bond price curve (as if market)
P0T = lambda r_, t: np.exp(-r_ * t)

class HWPaths:
    def __init__(self, n_paths, n_steps, T, r, lambd, eta, P0T): 
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = T
        self.lambd = lambd
        self.eta = eta
        self.r = r
        self.P0T = P0T
    def f0T(self, t):
        # time-step needed for differentiation
        dt = 0.01    
        # Define the forward rate equation 
        expr = - (np.log(self.P0T(self.r, t + dt)) - np.log(self.P0T(self.r, t - dt)))/(2 * dt)
        return expr
    def HW_path_generator(self):  
        
        # Initial interest rate is a forward rate(t0 --> 0)
        r0 = self.f0T(0.01)
        # Define theta(t) for HW model such that
        # dr(t) = lambda * (theta(t) - r(t))dt + eta * dW(t)
        theta = lambda t: \
                1.0 / lambd * (self.f0T(t + dt) - self.f0T(t - dt))/(2.0 * dt) + self.f0T(t) + \
                self.eta * self.eta / (2.0 * self.lambd * self.lambd) * (1.0 - np.exp(-2.0 * self.lambd * t))      
        # Generate standard normal random variables
        Z = np.random.normal(0.0,1.0,[self.n_paths,self.n_steps])
        # Initiate W
        W = np.zeros([self.n_paths,self. n_steps+1])
        # Initiate interest rate paths
        R = np.zeros([self.n_paths, self.n_steps+1])
        R[:,0] = r0
        M = np.zeros([self.n_paths,self. n_steps+1])
        M[:,0]= 1.0
        time = np.zeros([self.n_steps+1])
        dt = self.T / float(self.n_steps)
        for i in range(0, self.n_steps):
            # standardize
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            # Cumulate W as W(t + 1) = W(t) + Z(t + 1) * sqrt(dt)
            W[:,i+1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]
            # r(t + 1) = r(t) + lambda * (theta(t + 1) - r(t + 1)) * dt + eta * (W(t + 1) - W(t))
            R[:,i+1] = R[:,i] + lambd * (theta(time[i]) - R[:,i]) * dt + eta * (W[:,i+1] - W[:,i])
            M[:,i+1] = M[:,i] * np.exp((R[:,i+1] + R[:,i]) * 0.5 * dt)
            time[i+1] = time[i] + dt
        # Outputs
        paths = {"time" : time,"R" : R, "M": M}
        return paths
    def Plot_ZCB_vs_MC(self):
        # In this experiment we compare ZCB from the Market and Monte Carlo
        paths = self.HW_path_generator()
        M = paths["M"]
        t_i = paths["time"]         
        # Here we compare the price of an option on a ZCB from Monte Carlo the Market  
        p_MC = np.zeros([self.n_steps + 1])
        for i in range(0,self.n_steps + 1):
             p_MC[i] = np.mean(1.0 / M[:,i])
             plt.figure(1)
             plt.grid()
             plt.xlabel('T')
             plt.ylabel('P(0,T)')
             plt.plot(t_i, np.exp(-self.r * t_i))
             plt.plot(t_i, p_MC,'--r')
             plt.legend(['P(0,t) from market instruments','P(0,t) MC'])
             plt.title('ZCBs as at HW Model')
             plt.show()
             
a = HWPaths(n_paths = n_paths, n_steps = n_steps, T = T,  r = r, lambd = lambd, eta = eta, P0T = P0T)
a.HW_path_generator()
a.Plot_ZCB_vs_MC()

b = HWPaths(n_paths = n_paths, n_steps = n_steps, T = T,  r = 0.03, lambd = 0.019, eta = 0.02, P0T = P0T)
b.HW_path_generator()
b.Plot_ZCB_vs_MC()
