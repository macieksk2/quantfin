#%%
"""
Model: Black-Scholes with Jump volatility
Z ~ N(0,1)
J ~ N(mu(J), sigma(J))
dX(t) = (r - 0.5 * J(t) ^ 2) dt + J(t) * sqrt(dt) * Z(t)
S(t) = S(0) * exp(X(t))
Assess how it would affect the pricing of european options (Monte Carlo simulations with conditional expectations)
"""
import numpy as np
import matplotlib.pyplot as plt
import enum
import scipy.stats as st

# Class defining European puts and calls
class Euro_Opt_Type(enum.Enum):
    CALL = 1.0
    PUT = -1.0

######################## INPUT 
n_paths = 25
n_steps = 500
T = 5
t = 0
mu_J = 0.005
sigma_J = 0.3
S0 = 100
r  = 0.00
CP = Euro_Opt_Type.CALL # Euro_Opt_Type.PUT
S = 100
K = 80
J = 0

######################## CLASSES
class OptionPrice_MC_CondExp:
    def __init__(self, n_paths, n_steps, S0, T, mu_J, sigma_J, r, CP, S, K, J, t):
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.S0 = S0
        self.T = T
        self.mu_J = mu_J
        self.sigma_J = sigma_J
        self.r = r
        self.CP = CP
        self.S = S
        self.K = K
        self.J = J
        self.t = t
        
    def Sim_Paths(self):    
        # Create empty matrices for Poisson process and for compensated Poisson process
        X = np.zeros([self.n_paths, self.n_steps + 1])
        S = np.zeros([self.n_paths, self.n_steps + 1])
        time = np.zeros([self.n_steps+1])
                    
        dt = self.T / float(self.n_steps)
        X[:,0] = np.log(self.S0)
        S[:,0] = self.S0
        
        # draw Z - standard normal distribution
        Z = np.random.normal(0.0,1.0,[self.n_paths,self.n_steps])
        # draw J - N(mu_J, sigma_J)
        J = np.random.normal(self.mu_J,self.sigma_J,[self.n_paths,self.n_steps])
        for i in range(0,self.n_steps):
            # normalize samples
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
                
            X[:,i+1]  = X[:,i] + (self.r - 0.5 * J[:,i] ** 2.0)*dt +J[:,i]*np.sqrt(dt)* Z[:,i] 
            time[i+1] = time[i] +dt
            
        S = np.exp(X)
        paths = {"time":time,"X":X,"S":S,"J":J}
        return paths

    def Calc_Euro_Option_MC(self):
        # S is a vector of Monte Carlo samples at T
        if self.CP == Euro_Opt_Type.CALL:
            return np.exp(-self.r * self.T) * np.mean(np.maximum(self.S - self.K,0.0))
        elif self.CP == Euro_Opt_Type.PUT:
            return np.exp(-self.r * self.T) * np.mean(np.maximum(self.K - self.S,0.0))

    def BS_Call_Put_Option_Price(self):
        K = np.array(self.K).reshape([len([self.K]),1])
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * np.power(self.sigma_J,2.0))
        * (self.T-self.t)) / (self.sigma_J * np.sqrt(self.T-self.t))
        d2 = d1 - self.sigma_J * np.sqrt(self.T-self.t)
        if self.CP == Euro_Opt_Type.CALL:
            value = st.norm.cdf(d1) * self.S0 - st.norm.cdf(d2) * K * np.exp(-self.r * (self.T-self.t))
        elif self.CP == Euro_Opt_Type.PUT:
            value = st.norm.cdf(-d2) * K * np.exp(-self.r * (self.T-self.t)) - st.norm.cdf(-d1)*self.S0
        return value
#
    def Euro_Option_Cond_Exp(self):
        
        # Jumps at time T
        J_i = self.J[:,-1]
        
        result = np.zeros([self.n_paths])
        for j in range(0,self.n_paths):
            sigma = J_i[j]
            result[j] = self.BS_Call_Put_Option_Price()
        return np.mean(result)
#
a = OptionPrice_MC_CondExp(n_paths = n_paths, n_steps = n_steps, S0 = S0, T = T, mu_J = mu_J, sigma_J = sigma_J, 
                           r = r, CP = CP, S = S, K = K, J = J, t = t)

Paths = a.Sim_Paths()
t_grid = Paths["time"]
X = Paths["X"]
S = Paths["S"]
J = Paths["J"]

plt.figure(1)
plt.plot(t_grid, np.transpose(X))   
plt.grid()
plt.xlabel("time")
plt.ylabel("X(t)")

plt.figure(2)
plt.plot(t_grid, np.transpose(S))   
plt.grid()
plt.xlabel("time")
plt.ylabel("S(t)")

plt.figure(3)
plt.plot(t_grid[:(len(t_grid) - 1)], np.transpose(J[0]))   
plt.grid()
plt.xlabel("time")
plt.ylabel("J(t)")
# Check the convergence for a given strike
grid = range(100,10000,1000)
n_runs = len(grid)

resultMC = np.zeros([n_runs])
resultCondExp = np.zeros([n_runs])
   
for (i,N) in enumerate(grid):
        print(N)
        b = OptionPrice_MC_CondExp(n_paths = N, n_steps = n_steps, S0 = S0, T = T, mu_J = mu_J, sigma_J = sigma_J, 
                           r = r, CP = CP, S = S, K = K, J = J, t = t)
        Paths = b.Sim_Paths()
        t_grid = Paths["time"]
        S = Paths["S"]
        c = OptionPrice_MC_CondExp(n_paths = N, n_steps = n_steps, S0 = S0, T = T, mu_J = mu_J, sigma_J = sigma_J, 
                           r = r, CP = CP, S = S[:,-1], K = K, J = J, t = t)
        resultMC[i] = c.Calc_Euro_Option_MC()
        
        J = Paths["J"]
        d = OptionPrice_MC_CondExp(n_paths = N, n_steps = n_steps, S0 = S0, T = T, mu_J = mu_J, sigma_J = sigma_J, 
                           r = r, CP = CP, S = S, K = K, J = J, t = t)
        resultCondExp[i]= d.Euro_Option_Cond_Exp()

plt.figure(4)
plt.plot(grid,resultMC)  
plt.plot(grid,resultCondExp)
plt.legend(['Monte Carlo','Conditional Expectation'])
if CP == Euro_Opt_Type.CALL:
    plt.title('Call Option Price- Convergence')
else:
    plt.title('Put Option Price- Convergence')
plt.xlabel('No Paths')
plt.ylabel('Option price, strike = K')
plt.grid()
                   
