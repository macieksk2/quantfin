#%%
"""
Generate stochastic paths from:
    - Poisson process
    - Compensated Poisson Process (the martingale) -> basically, deduct lambda in order to "detrend" the process
    - Jump Merton process
https://almostsuremath.com/2010/06/24/poisson-processes/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
################### INPUT #################################################################
n_paths_ = 25
n_steps_ = 500
T_ = 50
lambda_ = 1
muJ_ = 0
sigmaJ_ = 0.25
sigma_ = 0.25
S0_ = 100
r_ = 0.1

class PoissonMertonProcess:
    def __init__(self, n_paths_, n_steps_, T_, lambda_, S0_, muJ_, sigmaJ_, r_, sigma_): 
        self.n_paths_ = n_paths_
        self.n_steps_ = n_steps_
        self.T_ = T_ 
        self.lambda_ = lambda_
        self.S0_ = S0_
        self.muJ_ = muJ_
        self.sigmaJ_ = sigmaJ_
        self.r_ = r_
        self.sigma_ = sigma_
    def PoissonPathCreator(self):    
        # Initiate data structures to store the process realizations
        X = np.zeros([self.n_paths_, self.n_steps_ + 1])
        X_prim = np.zeros([self.n_paths_, self.n_steps_ + 1])
        t_ = np.zeros([self.n_steps_ + 1])
                    
        dt = self.T_ / float(self.n_steps_)
        
        Z = np.random.poisson(self.lambda_ * dt,[self.n_paths_, self.n_steps_])
        
        for i in range(0, self.n_steps_):
            # Standardize the values
            X[:,i+1]  = X[:,i] + Z[:,i]
            X_prim[:,i+1] = X_prim[:,i]  - self.lambda_ * dt + Z[:,i]
            t_[i+1] = t_[i] +dt
            
        paths_ = {"Time" : t_, "X" : X, "Xprim" : X_prim}
        return paths_
    def MertonPathCreator(self):    
        # Initiate data structures to store the process realizations
        X_ = np.zeros([self.n_paths_, self.n_steps_ + 1])
        S_ = np.zeros([self.n_paths_, self.n_steps_ + 1])
        t_ = np.zeros([self.n_steps_ + 1])
                    
        dt = self.T_ / float(self.n_steps_)
        X_[:,0] = np.log(self.S0_)
        S_[:,0] = self.S0_
        
        # Calculate expected value of exp(Jump) E(e^J),  J ~ N()
        EeJ = np.exp(self.muJ_ + 0.5 * self.sigmaJ_ ** 2)
        
        ZPois_ = np.random.poisson(self.lambda_ * dt,[self.n_paths_,self.n_steps_])
        ZNorm_ = np.random.normal(0.0,1.0,[self.n_paths_,self.n_steps_])
        
        J = np.random.normal(self.muJ_, self.sigmaJ_, [self.n_paths_, self.n_steps_])
        
        for i in range(0,self.n_steps_):
            # Standardize the values
            if self.n_paths_ > 1:
                ZNorm_[:,i] = (ZNorm_[:,i] - np.mean(ZNorm_[:,i])) / np.std(ZNorm_[:,i])
            X_[:,i+1]  = X_[:,i] + (self.r_ - self.lambda_*(EeJ - 1) - 0.5 * self.sigma_ ** 2) * dt + \
                         self.sigma_ * np.sqrt(dt) * ZNorm_[:,i] + J[:,i] * ZPois_[:,i]
            t_[i+1] = t_[i] + dt
            
        S_ = np.exp(X_)
        paths = {"Time": t_,"X": X_,"S": S_}
        return paths
    
    def plotPaths(self):
    
        _poisson_paths_ = self.PoissonPathCreator()
        timeGrid = _poisson_paths_["Time"]
        X = _poisson_paths_["X"]
        Xprim = _poisson_paths_["Xprim"]
        _merton_paths = self.MertonPathCreator()
        timeGrid = _merton_paths["Time"]
        S_m = _merton_paths["S"]  
        X_m = _merton_paths["X"]  
           
        plt.figure(1)
        plt.plot(timeGrid, np.transpose(X),'-r')   
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("X(t) - Poisson")
        
        plt.figure(2)
        plt.plot(timeGrid, np.transpose(Xprim),'-b')   
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("X(t) - Compensated Poisson")
                     
        plt.figure(3)
        plt.plot(timeGrid, np.transpose(S_m),'-r')   
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("S(t) - Merton with Jump")

        plt.figure(4)
        plt.plot(timeGrid, np.transpose(X_m),'-b')   
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("X(t) - Merton with Jump")

a = PoissonMertonProcess(n_paths_= n_paths_, n_steps_ = n_steps_, T_ = T_, lambda_ = lambda_, \
                   S0_ = S0_, muJ_ = muJ_, sigmaJ_ = sigmaJ_, r_ = r_, sigma_ = sigma_)
a.PoissonPathCreator()
a.MertonPathCreator()
a.plotPaths()


