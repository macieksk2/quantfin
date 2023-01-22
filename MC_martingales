
"""
Simulation of E(W(t)|F(s)) = W(s) using nested Monte Carlo:
    1. Simulate W up till the moment t, store the simluation of W(t)
    2. Simulate W from s to t, calculate the expectation of the nested simulations
    3. Check if 1. and 2. are equal
"""
import numpy as np
import matplotlib.pyplot as plt
t = 10 
s = 5
no_paths = 100000
n_steps = 10
no_trials = 100

# Martingale A:
# 1. Calculate E(W(t)|F(0)) = W(0)
# 2. Check if it equals 0
class martingaleA:
    def __init__(self, no_paths):
        self.no_paths = no_paths
    def E_W_t(self):
        self.W_t = np.random.normal(0.0, pow(t,0.5), [self.no_paths,1])
        self.E_W_t = np.mean(self.W_t)
        print("mean value =: %.2f while the expected value is W(0) =%0.2f " % (self.E_W_t, 0.0))
        return self.E_W_t

# Repeat the exercise and check the distirbution of errors
Errors = []
for i in range(no_trials):
    A = martingaleA(no_paths)
    err = A.E_W_t()
    Errors.append(err)

plt.hist(Errors, density=True, bins=30)  
plt.ylabel('Probability')
plt.xlabel('Errors')
plt.show()

# Martingale B:
# 1. Perform nested MC sim
# 2. Check if E(W(t)|F(s)) = W(s)
class martingaleB:
    def __init__(self, no_paths, n_steps, t, s):
        self.no_paths = no_paths 
        self.n_steps = n_steps 
        self.t = t 
        self.s = s 
    def is_E_Wt_Fs_eq_Ws(self):
        # Draw random variables from a normal distribution
        # Save in a matrix nxm
        # where:
        # n - no of paths
        # m - no of steps
        Z = np.random.normal(0.0, 1.0, [self.no_paths, self.n_steps])
        # Initialize matrix W
        W = np.zeros([self.no_paths,self.n_steps+1])
            
        # Determine the time step in the interval [t0, s]
        dt_1 = s / float(self.n_steps)
        for i in range(0, self.n_steps):
            # normalize the samples -> mean = 0, variance = 1
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + pow(dt_1,0.5) * Z[:,i]
                
        # W_s - last column of W
        W_s = W[:,-1]
        # 1. For every path W(s) perform sub-simulation until time t and calculate
        # 2. Calculate the expectation
        # time-step from [s,t]
        dt_2 = (self.t - self.s) / float(self.n_steps);
        W_t  = np.zeros([self.no_paths, self.n_steps + 1]);
        
        # Store the results
        E_W_t = np.zeros([self.no_paths])
        Error = []
        for i in range(0,self.no_paths):
            # Simulate from s to t
            W_t[:,0] = W_s[i];
            Z = np.random.normal(0.0, 1.0,[self.no_paths, self.n_steps])
            for j in range(0, self.n_steps):
                # normalize the samples -> mean = 0, variance = 1
                Z[:,j] = (Z[:,j] - np.mean(Z[:,j])) / np.std(Z[:,j]);
                # Simulate from s to t
                W_t[:,j+1] = W_t[:,j] + pow(dt_2, 0.5)*Z[:,j];        
                
            E_W_t[i] = np.mean(W_t[:,-1])
            Error.append(E_W_t[i] - W_s[i])
            
            # Plot the paths
            if i == 0:
                plt.plot(np.linspace(0, s, n_steps + 1),W[0,:])
                for j in range(0,self.no_paths):
                    plt.plot(np.linspace(s,t,n_steps + 1),W_t[j,:])
                plt.xlabel("t")
                plt.ylabel("W(t)")
                plt.grid()
                plt.show()
            
        error = np.max(np.abs(E_W_t-W_s))
        return "The error =: %.18f" % (error)
    
B = martingaleB(no_paths = no_paths, n_steps = n_steps, t = t, s = s)
B.is_E_Wt_Fs_eq_Ws()
