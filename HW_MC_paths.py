#%%
"""
- Hull-White model
- MC simulation
- Try out different parameters (theta, eta) and plot the paths dependent on the parameter value
"""
import numpy as np
import matplotlib.pyplot as plt

##### INPUT
n_paths   = 1
n_steps   = 50000
T         = 100.0
r         = 0.03
# HW parameters
lambd     = 0.5
eta       = 0.01
lambdVec  = [-0.01, 0.01, 0.1, 0.2, 1.0, 2.0, 5.0]
etaVec    = [0.01, 0.1, 0.2, 0.3, 0.5, 1.0]

class HWPaths:
    def __init__(self, n_paths, n_steps, T, P0T, r):#, lambd, eta): # , lambd, eta
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.T = T
        self.P0T = P0T
        #self.lambd = lambd
        #self.eta = eta
        self.r = r
        
    def HW_path_generator(self, lambd, eta):  
        
        # time-step needed for differentiation
        dt = 0.0001    
        # Define the forward rate equation (lambda statement)
        f0T = lambda t: - (np.log(self.P0T(t + dt)) - np.log(self.P0T(t - dt))) / (2 * dt)
        
        # Initial interest rate is a forward rate(t0 --> 0)
        r0 = f0T(0.00001)
        # Define theta(t) for HW model such that
        # dr(t) = lambda * (theta(t) - r(t))dt + eta * dW(t)
        theta = lambda t: \
                1.0 / lambd * (f0T(t + dt) - f0T(t - dt))/(2.0 * dt) + f0T(t) + \
                eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))      
        # Generate standard normal random variables
        Z = np.random.normal(0.0,1.0,[self.n_paths,self.n_steps])
        # Initiate W
        W = np.zeros([self.n_paths,self. n_steps+1])
        # Initiate interest rate paths
        R = np.zeros([self.n_paths, self.n_steps+1])
        R[:,0] = r0
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
            time[i+1] = time[i] + dt
            
        # Outputs
        paths = {"time": time,"R":R}
        return paths
    def HW_path_plot_wrt_lamb(self, lambdVec, eta):  
        np.random.seed(0)
        # We define a ZCB curve (obtained from the market)
        P0T = lambda T: np.exp(-self.r * self.T) 
        # Effect of mean reversion lambda
        plt.figure(1) 
        legend = []
        for lambdTemp in lambdVec:    
            Paths = self.HW_path_generator(lambdTemp, eta)
            legend.append('lambda={0}'.format(lambdTemp))
            timeGrid = Paths["time"]
            R = Paths["R"]
            plt.plot(timeGrid, np.transpose(R))   
        plt.grid()
        plt.xlabel("time")
        plt.ylabel(str("R(t), eta = " + str(eta)))
        plt.legend(legend)
        
    def HW_path_plot_wrt_eta(self, etaVec, lambd):  
        np.random.seed(0)
        # We define a ZCB curve (obtained from the market)
        P0T = lambda T: np.exp(-self.r * self.T) 
        # Effect of the volatility
        plt.figure(2)    
        legend = []
        for etaTemp in etaVec:
            Paths = self.HW_path_generator(lambd, etaTemp)
            legend.append('eta={0}'.format(etaTemp))
            timeGrid = Paths["time"]
            R = Paths["R"]
            plt.plot(timeGrid, np.transpose(R))   
        plt.grid()
        plt.xlabel("time")
        plt.ylabel(str("R(t), lambd = " + str(lambd)))
        plt.legend(legend)

a = HWPaths(n_paths = n_paths, n_steps = n_steps, T = T, P0T = P0T, r = r)
a.HW_path_generator(lambd = lambd, eta = eta)
a.HW_path_plot_wrt_lamb(lambdVec = lambdVec, eta = eta)
a.HW_path_plot_wrt_eta(etaVec = etaVec, lambd = lambd)
