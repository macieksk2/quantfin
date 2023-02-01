#%%
"""
Pricing a Barrier option
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
import pandas as pd
class BarrierOption:
    def __init__(self, n_paths, n_steps, S, T, r, sigma, S_0, payoff, Su): 
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = T
        self.S = S
        self.sigma = sigma
        self.r = r
        self.S_0 = S_0
        self.payoff = payoff
        self.Su = Su
        
    def DigitalPayoffValuation(self, S_T):
        # S is a vector of Monte Carlo samples at T
        return np.exp(- self.r * self.T) * np.mean(self.payoff(S_T))
    
    def GeneratePathsGBMEuler(self):    
        Z = np.random.normal(0.0,1.0,[self.n_paths, self.n_steps])
        W = np.zeros([self.n_paths, self.n_steps + 1])
       
        # Euler Approximation
        S1 = np.zeros([self.n_paths, self.n_steps + 1])
        S1[:,0] = self.S_0
        
        time = np.zeros([self.n_steps + 1])
            
        dt = self.T / float(self.n_steps)
        for i in range(0, self.n_steps):
            # Normalization
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]
            
            S1[:,i + 1] = S1[:,i] + self.r * S1[:,i] * dt + self.sigma * S1[:,i] * (W[:,i+1] - W[:,i])
            time[i + 1] = time[i] + dt
            
        # Retun S1 and S2
        paths = {"time": time,"S": S1}
        
        return paths
    
    def UpAndOutBarrier(self, S_paths):
            
        # handle a barrier
        n1, n2 = S_paths.shape
        barrier = np.zeros([n1,n2]) + self.Su
        
        hitM = S_paths > barrier
        hitVec = np.sum(hitM, 1)
        hitVec = (hitVec == 0.0).astype(int)
        
        V_0 = np.exp(-self.r * self.T) * np.mean(payoff(S_paths[:,-1] * hitVec))
        
        return V_0

    def OptionPrice(self):
        paths   = self.GeneratePathsGBMEuler()        
        S_paths = paths["S"]
        S_T = S_paths[:,-1]
        
        # Plot
        S_T_grid = np.linspace(50, self.S_0 * 1.5,200)
        
        plt.figure(1)
        plt.plot(S_T_grid, self.payoff(S_T_grid))
        
        # Valuation
        val_t0 = self.DigitalPayoffValuation(S_T)
        print("Value of the contract at t0 ={0}".format(val_t0))
        
        # barrier pricing
        barrier_price = self.UpAndOutBarrier(S_paths)
        
        print("Value of the barrier contract at t0 ={0}".format(barrier_price))
        return barrier_price

### INPUT
n_paths = 10000
n_steps = 250 
S_0     = 100.0
S       = 100
r       = 0.05
T       = 5
sigma   = 0.2
Su      = 300
# Payoff setting    
K       = 100.0
K2      = 140.0
# Payoff specification
payoff = lambda S: np.maximum(S-K, 0.0)# - np.maximum(S-K2,0)

DPCR = BarrierOption(n_paths = n_paths, n_steps = n_steps, S = S, T = T, r = r, sigma = sigma, \
                           S_0 = S_0, payoff = payoff, Su = Su)
paths = DPCR.GeneratePathsGBMEuler()
DPCR.OptionPrice()

# Plot the paths against barrier
plt.figure(3)
plt.plot(paths["time"], np.transpose(paths["S"]))
plt.plot(paths["time"], [Su] * len(paths["time"]), 'r')
plt.title('Stock price process simulations vs the barrier')
plt.xlabel('Number of Steps')
plt.ylabel('Stock price')
plt.grid()
### SENS

### SENSITIVITY
# wrt to Su
Su_rng = [100, 150, 200, 250, 300, 350, 500, 700, 900, 1000]
prices = []
for su in Su_rng:
    DPCR = BarrierOption(n_paths = n_paths, n_steps = n_steps, S = S, T = T, r = r, sigma = sigma, \
                               S_0 = S_0, payoff = payoff, Su = su)
    price = DPCR.OptionPrice()
    prices.append(price)
pd.DataFrame(prices).plot()

# wrt to K
K_rng = [20, 50, 90, 130, 180, 220, 250, 500]
prices = []
for k in K_rng:
    # Payoff specification
    payoff = lambda S: np.maximum(S-k,0.0)# - np.maximum(S-K2,0)
    DPCR = BarrierOption(n_paths = n_paths, n_steps = n_steps, S = S, T = T, r = r, sigma = sigma, \
                               S_0 = S_0, payoff = payoff, Su = Su)
    price = DPCR.OptionPrice()
    prices.append(price)
pd.DataFrame(prices).plot()
