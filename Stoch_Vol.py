#%%
"""
Generate paths from Heston model
Correlated Brownian motions (Stock process and volatility)
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
############### Inputs ######################

n_paths_ = 1
n_steps_ = 500
T_ = 1.0
S0_ = 100
r_ = 0.03
sigma_LT_ = 0.2
kappa_ = 2
gamma_ = 0.3 # vol-of-vol parameter
rho_ = -0.9
############### Heston class ######################
class HestonModel:
  # Initate params
    def __init__(self, n_paths_, n_steps_, T_, S0_, sigma_LT_, r_, kappa_, gamma_, rho_): 
        self.n_paths_ = n_paths_
        self.n_steps_ = n_steps_
        self.T_ = T_ 
        self.S0_ = S0_
        self.sigma_LT_ = sigma_LT_
        self.r_ = r_
        self.kappa_ = kappa_
        self.gamma_ = gamma_
        self.rho_ = rho_
     # Generate Heston paths of vol and stock price based on correlated Wiener processes  
    def GenerateHestonPaths(self):    
        ZV = np.random.normal(0.0, 1.0, [self.n_paths_,self.n_steps_])
        ZX = np.random.normal(0.0, 1.0, [self.n_paths_,self.n_steps_])
        WV = np.zeros([self.n_paths_, self.n_steps_ + 1])
        WX = np.zeros([self.n_paths_, self.n_steps_ + 1]) 
        v = np.zeros([self.n_paths_, self.n_steps_ + 1]) 
        X = np.zeros([self.n_paths_, self.n_steps_ + 1])    
        S = np.zeros([self.n_paths_, self.n_steps_ + 1])    
        
        dt = self.T_ / float(self.n_steps_)
        v[:,0] = self.sigma_LT_
        X[:,0] = np.log(self.S0_)
        S[:,0] = self.S0_
        time = np.zeros([self.n_steps_ + 1])
        for i in range(0,self.n_steps_):
            # Correlated incremental Brownian Motions
            ZV[:,i]= self.rho_ * ZX[:,i] + np.sqrt(1.0 - self.rho_ ** 2) * ZV[:,i]
            # Generate the v process (volatility)
            v[:,i + 1]  = v[:,i] + self.kappa_ * (self.sigma_LT_ - v[:,i]) * dt + self.gamma_ * np.sqrt(v[:,i] * dt) *ZV[:,i]
            # Generate the X process
            X[:,i + 1]  = X[:, i] + (self.r_ - v[:,i + 1] / 2) * dt +  np.sqrt(v[:,i + 1] * dt) * ZX[:,i]
            # Wiener process
            WV[:,i+1] = WV[:,i] + np.power(dt, 0.5) * ZV[:,i]
            WX[:,i+1] = WX[:,i] + np.power(dt, 0.5) * ZX[:,i]
            
            time[i+1] = time[i] + dt
        # Transform from log growths to levels 
        S = np.exp(X)    
        #Store the results
        paths = {"time" : time, "S" : S, "X": X, "v" : v, "WV": WV, "WX": WX}
        return paths


############### Sensitivity wrt pho ######################
rho_ = [-0.9, 0, 0.9]
for r in rho_:
    print(r)
    a = HestonModel(n_paths_ = n_paths_, n_steps_ = n_steps_, T_ = T_, S0_ = S0_, \
                        sigma_LT_ = sigma_LT_, r_ = r_, kappa_ = kappa_, gamma_ = gamma_, rho_ = r)
    Paths = a.GenerateHestonPaths()
    timeGrid = Paths["time"]
    S = Paths["S"]
    WV = Paths["WV"]
    WX = Paths["WX"]
    v = Paths["v"]
    
    plt.figure()
    plt.plot(timeGrid, np.transpose(S))   
    plt.grid()
    plt.title("Corr " + str(r))
    plt.xlabel("time")
    plt.ylabel("S(t)")
    
    plt.figure()
    plt.plot(timeGrid, np.transpose(WV))   
    plt.plot(timeGrid, np.transpose(WX))   
    plt.grid()
    plt.title("Corr " + str(r))
    plt.xlabel("time")
    plt.ylabel("W(t)")
    
    plt.figure()
    plt.plot(timeGrid, np.transpose(v))   
    plt.grid()
    plt.title("Corr " + str(r))
    plt.xlabel("time")
    plt.ylabel("v(t)")


