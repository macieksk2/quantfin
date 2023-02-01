"""
Calculate P&L of the portfolio:
    1. constructed from a equity call option, 
    2. delta hedged, 
    3. reevaluated at each period (no transaction costs)
    4. Under BS model
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum 
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator


# Put / Call Option
class OptionType(enum.Enum):
    """
    Define European Call Option (1) or Put (-1)
    """
    CALL = 1.0
    PUT = -1.0

class BSHedging:
    def __init__(self, n_paths, n_steps, T, r, sigma, S_0, K, CP): 
        """
        Initiate parameters
        """
        self.n_steps   = n_steps
        self.n_paths   = n_paths
        self.r         = r
        self.S_0       = S_0
        self.K         = K
        self.CP        = CP
        self.sigma     = sigma
        self.T         = T
    def GBM_Path_Generator(self):    
        Z = np.random.normal(0.0, 1.0, [self.n_paths, self.n_steps])
        X = np.zeros([                  self.n_paths, self.n_steps + 1])
        W = np.zeros([                  self.n_paths, self.n_steps + 1])
        time = np.zeros([self.n_steps + 1])
            
        X[:,0] = np.log(self.S_0)
        
        dt = self.T / float(self.n_steps)
        for i in range(0, self.n_steps):
            # Normalize
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]
            X[:,i+1] = X[:,i] + (self.r - 0.5 * self.sigma ** 2) * dt + self.sigma * (W[:,i + 1] - W[:, i])
            time[i+1] = time[i]  +dt
            
        # Exponentiate BM to GBM
        S = np.exp(X)
        paths = {"time": time,"S": S}
        return paths
    
    # Black-Scholes Call option price
    def Euro_Option_Price(self, t, K, S_0):
        K  = np.array(K).reshape([len(K), 1])
        d1 = (np.log(S_0 / K) + (self.r + 0.5 * np.power(self.sigma,2.0)) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - self.sigma * np.sqrt(self.T - t)
        if self.CP == OptionType.CALL:
            value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-self.r * (self.T - t))
        elif self.CP == OptionType.PUT:
            value = st.norm.cdf(-d2) * K * np.exp(-self.r * (self.T - t)) - st.norm.cdf(-d1) * S_0
        return value
    
    def BS_Delta(self, t, K, S_0):
        # when defining a time-grid it may happen that the last grid point is slightly after the maturity
        if t - self.T > 10e-20 and self.T - t < 10e-7:
            t = self.T
        K     = np.array(K).reshape([len(K) ,1])
        d1    = (np.log(S_0 / K) + (self.r + 0.5 * np.power(self.sigma,2.0)) * (self.T-t)) / (self.sigma * np.sqrt(self.T-t))
        if self.CP == OptionType.CALL:
            value = st.norm.cdf(d1)
        elif self.CP == OptionType.PUT:
           value = st.norm.cdf(d1) - 1.0
        return value
    
    def PnL(self):
        np.random.seed(1)
        Paths = self.GBM_Path_Generator()
        time  = Paths["time"]
        S     = Paths["S"]
        
        # Define Call and Delta price
        C = lambda t, K, S_0: self.Euro_Option_Price(t, K, S_0)
        Delta = lambda t, K, S_0: self.BS_Delta(t, K, S_0)
        
        # Setting up initial portfolio
        # P&L
        PnL        =  np.zeros([self.n_paths, self.n_steps + 1])
        # Initial value of Delta greek
        delta_init =  Delta(0.0, self.K, self.S_0)
        # Initial Portfolio = Option Price(0, K, S0) - Delta * S_0
        PnL[:,0]   =  C(0.0, self.K, self.S_0) - delta_init * self.S_0
        # Storage place for values of Calls and Delta
        CallM       =  np.zeros([self.n_paths, self.n_steps + 1])
        CallM[:,0]  =  C(0.0, self.K, self.S_0)
        DeltaM      =  np.zeros([self.n_paths, self.n_steps + 1])
        DeltaM[:,0] =  Delta(0.0, self.K, self.S_0)
        # Iterate through steps
        for i in range(1, self.n_steps + 1):
            dt         = time[i] - time[i - 1]
            # Calc Delta at t - 1
            delta_old  = Delta(time[i - 1], self.K, S[:,i - 1])
            # Calc Delta at t
            delta_curr = Delta(time[i],     self.K, S[:,i])
            # P&L(t) = P&L(t - 1) * exp(r * dt) - (Delta(t) - Delta(t - 1)) * S(t)
            # Add interest from Deposit
            # Subtract stock * change in Deltas (hedging)
            PnL[:,i]    =  PnL[:,i-1] * np.exp(self.r * dt) - (delta_curr - delta_old) * S[:,i]
            # Recalculate option price at t
            CallM[:,i]  =  C(time[i], self.K, S[:,i])
            # Reevalute Delta at t
            DeltaM[:,i] =  delta_curr
            
        # At the end of the analysis:
        # 1. Payment of the option (if in the money);
        # 2. Sell the hedge
        PnL[:,-1] = PnL[:,-1] - np.maximum(S[:,-1] - self.K, 0) +  DeltaM[:,-1] * S[:,-1]
        
        # Plot - a single realization
        path_id = 13
        plt.figure(1)
        plt.plot(time,S[path_id,:])
        plt.plot(time,CallM[path_id,:])
        plt.plot(time,DeltaM[path_id,:])
        plt.plot(time,PnL[path_id,:])
        plt.legend(['Stock','CallPrice','Delta','PnL'])
        plt.grid()
        
        # Plot the histogram of PnL
        plt.figure(2)
        plt.hist(PnL[:,-1],50)
        plt.grid()
        plt.xlim([-0.1,0.1])
        plt.title('histogram of P&L')

### INPUT
n_steps   = 100
n_paths   = 10000
T         = 10.0
S_0       = 100
r         = 0.1
sigma     = 0.2
K         = [90]
CP        = OptionType.CALL
    
# DEFINE CLASS
BS_Hedge = BSHedging(n_paths = n_paths, n_steps = n_steps, T = T, r = r, sigma = sigma, S_0 = S_0, K = K, CP = CP)
GBM_Paths = BS_Hedge.GBM_Path_Generator()
Opt_Price = BS_Hedge.Euro_Option_Price(0, K, S_0)
Delta = BS_Hedge.BS_Delta(0, K, S_0)
BS_Hedge.PnL()
