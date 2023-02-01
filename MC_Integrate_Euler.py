import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

class MCIntegral:
    def __init__(self, n_samples, n_paths, n_steps, a, b, c, d, g, T): 
        """
        g - a function to be applied to randomly drawn xs
        a, b - bounds for x
        c, d - bounds for y
        """
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.g = g
        self.T = T
    def ComputeIntegral1(self):  
        """
        Suppose an x − y plane,
        draw random numbers (dots in the graph). 
        The MC integral of a function is approximately given by the total area times the fraction of points that fall under the curve g(x).
        -> The greater the number of points the more accurate the evaluation of this area.
        -> Only competitive for complicated and/or multi-dimensional functions.
        """
        # Draw n random samples from unif distr for X and Y axis
        x_i = np.random.uniform(self.a,self.b,self.n_samples)
        y_i = np.random.uniform(self.c,self.d,self.n_samples)
        # Check which g(x) are bove ys
        p_i = g(x_i) > y_i
        # take the average of obs above 
        p = np.sum(p_i) / self.n_samples
        # Calc the integral
        integral = p * (self.b - self.a) * (self.d - self.c) 
        # Plot
        plt.figure(1)
        plt.plot(x_i,y_i,'.r')
        plt.plot(x_i,g(x_i),'.b')
        return integral
    
    def ComputeIntegral2(self): 
        """
        Monte Carlo integration defined as follows:
        1. draw N random numbers, xi, i = 1, 2, . . . , N
        from uniform distribution U[a, b];
        2. calculating the following approximation:
            int_{a,b} g(x) dx ~= (b - a) / N * sum g(x_i)
    
        """
        x_i = np.random.uniform(self.a,self.b,self.n_samples)
        p = (self.b - self.a) * np.mean(g(x_i))
        
        return p
    def ComputeIntegrals(self):    
        """
        Integrated BM
        """
        Z = np.random.normal(0.0, 1.0, [self.n_paths,self.n_steps])
        W = np.zeros([self.n_paths, self.n_steps + 1])
        Integrated_1 = np.zeros([self.n_paths, self.n_steps + 1])
        time = np.zeros([self.n_steps + 1])
        
        dt = self.T / float(self.n_steps)
        for i in range(0,self.n_steps):
            # Normalization
            if n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
            # Calculate the integer as sum g(W(i)) * (W(i + 1) - W(i))
            Integrated_1[:,i+1] = Integrated_1[:,i] + self.g(W[:,i]) * (W[:,i+1] - W[:,i]) 
            time[i+1] = time[i] + dt
            
        paths = {"time": time,"W": W,"Integrated_1": Integrated_1}
        timeGrid = W_t["time"]
        Ws = W_t["W"]
        intWsds = W_t["Integrated_1"]

        plt.figure(1)
        plt.plot(timeGrid, np.transpose(Ws))
        plt.plot(timeGrid, np.transpose(intWsds),'r')
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title("Integrated Brownian Motion paths")
        return paths
    def ComputeIntegrals3cases(self):    
        """
        Integrated Brownian motion- three cases, W(t), and I(t)
        """
        Z = np.random.normal(0.0,1.0,[self.n_paths,self.n_steps])
        W = np.zeros([self.n_paths, self.n_steps+1])
        Integrated_1 = np.zeros([self.n_paths, self.n_steps+1])
        Integrated_2 =  np.zeros([self.n_paths, self.n_steps+1])
        time = np.zeros([self.n_steps + 1])
        
        dt = self.T / float(self.n_steps)
        for i in range(0, self.n_steps):
            # making sure that samples from normal have mean 0 and variance 1
            if n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
            # Calculate the integer as sum W(i) * dt
            Integrated_1[:,i + 1] = Integrated_1[:,i] + W[:,i]*dt
            # Calculate the integer as sum W(i) * (W(i + 1) - W(i))
            Integrated_2[:,i + 1] = Integrated_2[:,i] + W[:,i]*(W[:,i + 1]-W[:, i])
            time[i+1] = time[i] + dt
            
        paths = {"time":time,"W":W,"Integrated_1":Integrated_1,"Integrated_2":Integrated_2}
        plt.figure(1)
        plt.plot(paths["time"], np.transpose(Integrated_1))
        plt.plot(paths["time"], np.transpose(Integrated_2),'r')
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("value")
        plt.title("Integrated Brownian Motion paths")
        return paths

### INPUT 
n_samples = 10000
n_paths = 100
n_steps = 100
a = 0.0
b = 1.0
c = 0.0
d = 3.0 
T = 1

### INTEGRATION
g = lambda x: np.exp(x)
# Methods:
# 1. Count the number of occurences of points under the curve
MC_Ints = MCIntegral(n_samples = n_samples, n_paths = n_paths, n_steps = n_steps, a = a, b = b, c = c, d = d, g = g, T = T)
out_ = MC_Ints.ComputeIntegral1()
print('Integral from Monte Carlo 1 is {0}'.format(out_))
# 2. (b - a) * mean
out_2_ =  MC_Ints.ComputeIntegral2()
print('Integral from Monte Carlo 2 is {0}'.format(out_2_))
# 3. Analytical
print('Integral computed analytically = {0}'.format(np.exp(b) - np.exp(a)))

# Quadatic function
g = lambda x: x ** 2
d = 1
# 1. Count the number of occurences of points under the curve
MC_Ints_Q = MCIntegral(n_samples = n_samples, n_paths = n_paths, n_steps = n_steps, a = a, b = b, c = c, d = d, g = g, T = T)
out_ = MC_Ints_Q.ComputeIntegral1()
print('Integral from Monte Carlo 1 is {0}'.format(out_))
# 2. (b - a) * mean
out_2_ =  MC_Ints_Q.ComputeIntegral2()
print('Integral from Monte Carlo 2 is {0}'.format(out_2_))
# 3. Analytical
print('Integral computed analytically = {0}'.format((b ** 3 - a ** 3) / 3))

# integration - single
n_paths = 1
n_steps = 1000
T = 1
MC_Int_single = MCIntegral(n_samples = n_samples, n_paths = n_paths, n_steps = n_steps, a = a, b = b, c = c, d = d, g = g, T = T)
W_t = MC_Ints.ComputeIntegrals()

# linear function
n_paths = 100000
n_steps = 1000
T = 2
g = lambda t: t
MC_Int_single_2 = MCIntegral(n_samples = n_samples, n_paths = n_paths, n_steps = n_steps, a = a, b = b, c = c, d = d, g = g, T = T)
out_2 = MC_Int_single_2.ComputeIntegrals()
G_T = out_2["Integrated_1"]
EX = np.mean(G_T[:,-1])
Var = np.var(G_T[:,-1])
print('Mean = {0} and variance ={1}'.format(EX,Var))

out_2_1 = MC_Int_single_2.ComputeIntegrals3cases()
G_T_1 = out_2_1["Integrated_1"]
G_T_2 = out_2_1["Integrated_2"]


### Generate Paths with Euler / Milstein discretization of GBM
class PathGenerator:
    def __init__(self, n_samples, n_paths, n_steps, T, r, sigma, S_0): 
        """
        Euler / Milstein discretization of GBM
        """
        self.n_samples = n_samples
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.r = r
        self.sigma = sigma
        self.S_0 = S_0
        self.d = d
        self.g = g
        self.T = T
    def GeneratePathsGBMEuler(self):  
        """
        Euler discretization of the GBM
        """
        Z = np.random.normal(0.0, 1.0, [self.n_paths,self.n_steps])
        W = np.zeros([self.n_paths, self.n_steps + 1])
       
        # Euler approximation
        S1 = np.zeros([self.n_paths, self.n_steps + 1])
        S1[:, 0] = self.S_0
        
        # Analytical method (exact)
        S2 = np.zeros([self.n_paths, self.n_steps + 1])
        S2[:, 0] = self.S_0
        
        time = np.zeros([self.n_steps + 1])
            
        dt = self.T / float(self.n_steps)
        for i in range(0,self.n_steps):
            # Normalization
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5) * Z[:,i]
            # generate path based on Euler discr
            S1[:,i+1] = S1[:,i] + self.r * S1[:,i] * dt                          + self.sigma * S1[:,i] * (W[:,i+1] - W[:,i])
            # generate path based on analytical solution of SDE
            S2[:,i+1] = S2[:,i] * np.exp((self.r - 0.5 * self.sigma ** 2.0) * dt + self.sigma * (W[:,i+1] - W[:,i]))
            time[i+1] = time[i] + dt
            
        # Retun S1 and S2
        paths = {"time" : time,"S1" : S1,"S2" : S2}
        timeGrid = paths["time"]
        S1 = paths["S1"]
        S2 = paths["S2"]

        plt.figure(1)
        plt.plot(timeGrid, np.transpose(S1),'k')   
        plt.plot(timeGrid, np.transpose(S2),'--r')   
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("S(t)")
        
        return paths
    
    def GeneratePathsGBMMilstein(self):
        """
        Milstein discretization of the GBM
        
        In the case of deterministic differential equations, one may employ
        the Taylor expansion to define discretizations by which we may
        obtain a higher order of convergence. For stochastic differential
        equations a similar approach is available, which is based on the
        stochastic Taylor expansion, or the so-called Itˆo-Taylor expansion.
        The stochastic Euler approximation is based on the first two terms
        of this expansion.
    
        """    
        Z = np.random.normal(0.0, 1.0,[self.n_paths, self.n_steps])
        W = np.zeros([self.n_paths, self.n_steps + 1])
       
        # Approximation
        S1 = np.zeros([self.n_paths, self.n_steps + 1])
        S1[:,0] = self.S_0
        
        # Exact
        S2 = np.zeros([self.n_paths, self.n_steps + 1])
        S2[:,0] = self.S_0
        
        time = np.zeros([self.n_steps + 1])
            
        dt = self.T / float(self.n_steps)
        for i in range(0, self.n_steps):
            # making sure that samples from normal have mean 0 and variance 1
            if self.n_paths > 1:
                Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            W[:,i+1] = W[:,i] + np.power(dt, 0.5)*Z[:,i]
            # generate path based on Milstein discr            
            S1[:,i+1] = S1[:,i] + self.r * S1[:,i]* dt + self.sigma * S1[:,i] * (W[:,i+1] - W[:,i]) \
                        + 0.5 * self.sigma ** 2 * S1[:,i] * (np.power((W[:,i+1] - W[:,i]),2) - dt)
             # generate path based on analytical solution of SDE                       
            S2[:,i+1] = S2[:,i] * np.exp((self.r - 0.5*self.sigma*self.sigma) * dt + self.sigma * (W[:,i+1] - W[:,i]))
            time[i+1] = time[i] + dt
            
        # Retun S1 and S2
        paths = {"time":time,"S1":S1,"S2":S2}
        timeGrid = paths["time"]
        S1 = paths["S1"]
        S2 = paths["S2"]

        plt.figure(1)
        plt.plot(timeGrid, np.transpose(S1),'k')   
        plt.plot(timeGrid, np.transpose(S2),'--r')   
        plt.grid()
        plt.xlabel("time")
        plt.ylabel("S(t)")
        
        return paths

### INPUT 
n_paths = 25
n_steps = 25
T = 1
r = 0.05
sigma = 0.2
S_0 = 100

### Simulated paths - Euler
Generator_ = PathGenerator(n_samples, n_paths, n_steps, T, r, sigma, S_0)
Paths = Generator_.GeneratePathsGBMEuler()

### Weak and strong convergence
n_stepsV = range(1,500,1)
n_paths = 250
errorWeak = np.zeros([len(n_stepsV),1])
errorStrong = np.zeros([len(n_stepsV),1])
dtV = np.zeros([len(n_stepsV),1])
for idx, n_steps in enumerate(n_stepsV):
    Generator_ = PathGenerator(n_samples, n_paths, n_steps, T, r, sigma, S_0)
    Paths = Generator_.GeneratePathsGBMEuler()
    # Get the paths at T
    S1_atT = Paths["S1"][:,-1]
    S2_atT = Paths["S2"][:,-1]
    
    errorWeak[idx] = np.abs(np.mean(S1_atT)-np.mean(S2_atT))
    
    errorStrong[idx] = np.mean(np.abs(S1_atT-S2_atT))
    dtV[idx] = T/n_steps
    
print(errorStrong)    
plt.figure(2)
plt.plot(dtV,errorWeak)
plt.plot(dtV,errorStrong,'--r')
plt.grid()
plt.legend(['weak conv.','strong conv.'])
     
### Simulated paths - Milstein
Generator_M = PathGenerator(n_samples, n_paths, n_steps, T, r, sigma, S_0)
Paths = Generator_M.GeneratePathsGBMMilstein()

# Weak and strong convergence
n_stepsV = range(1,500,1)
n_paths = 100
errorWeak = np.zeros([len(n_stepsV),1])
errorStrong = np.zeros([len(n_stepsV),1])
dtV = np.zeros([len(n_stepsV),1])
for idx, n_steps in enumerate(n_stepsV):
    Generator_M = PathGenerator(n_samples, n_paths, n_steps, T, r, sigma, S_0)
    Paths = Generator_M.GeneratePathsGBMMilstein()
    # Get the paths at T
    S1_atT = Paths["S1"][:,-1]
    S2_atT = Paths["S2"][:,-1]
    errorWeak[idx] = np.abs(np.mean(S1_atT)-np.mean(S2_atT))
    errorStrong[idx] = np.mean(np.abs(S1_atT-S2_atT))
    dtV[idx] = T/n_steps
    
print(errorStrong)    
plt.figure(2)
plt.plot(dtV,errorWeak)
plt.plot(dtV,errorStrong,'--r')
plt.grid()
plt.legend(['weak conv.','strong conv.'])
    
