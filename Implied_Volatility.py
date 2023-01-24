#%%
"""
Calculate Black Scholes Implied Volatility
Compare implied volatility from market prices to BS model (Vol Smile / Hump)
"""
import numpy as np
from numpy import arange
import scipy.stats as st
import enum
import matplotlib.pyplot as plt
################################################ INPUT ###################################################
# Define option type
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

V_price_     = 15             # np.arange(1.0, 3.1, 0.1)    # market call option price
K            = 90             # Strike
T_           = 1              # Time to Maturity
r            = 0.05           # Risk-free rate
S_init       = 100            # Stock price at moment 0
sigmaInit    = 0.25           # Start impliedvol
CP           = OptionType.CALL #CALL = call, PUT = put European equity option
################################################ CLASS INITIATION ###################################################
class ImplVolCalc:
    def __init__(self, V_price_, K , S_init, T_, r, sigmaInit, CP): 
        self.V_price_ = V_price_
        self.K = K
        self.S_init = S_init
        self.T_ = T_
        self.r = r
        self.sigmaInit = sigmaInit
        self.CP = CP
    # Black Scholes European Call Option Price
    def Eur_Call_BS_Price(self, sigma):
        d1    = (np.log(self.S_init / float(self.K)) + (r + 0.5 * np.power(sigma,2.0)) * self.T_) / float(sigma * np.sqrt(self.T_))
        d2    = d1 - sigma * np.sqrt(self.T_)
        if self.CP == OptionType.CALL:
            val_ = st.norm.cdf(d1) * S_init - st.norm.cdf(d2) * self.K * np.exp(-self.r * self.T_)
        elif self.CP == OptionType.PUT:
            val_ = st.norm.cdf(-d2) * self.K * np.exp(-self.r * self.T_) - st.norm.cdf(-d1) * self.S_init
        return val_
    # Calculate Vega
    def Eur_vega(self, sigma):
        #parameters and val_ of Vega
        d2   = (np.log(self.S_init / float(self.K)) + (self.r - 0.5 * np.power(sigma,2.0)) * self.T_) / float(sigma * np.sqrt(self.T_))
        val_ = self.K * np.exp(-self.r * self.T_) * st.norm.pdf(d2) * np.sqrt(self.T_)
        return val_
    # Calculate implied volatiltiy with Newton method
    def ImplVol(self, sigma):
        e_    = 1e10; # initial e_
        #Handy lambda expressions
        bsEurOptPrice = lambda sigma: self.Eur_Call_BS_Price(sigma)
        vega= lambda sigma: self.Eur_vega(sigma)
        
        # While the difference between the model and the market price is large
        # follow the iteration
        n = 1.0 
        while e_ > 10e-12:
            #print("opt price = ", bsEurOptPrice(sigma))
            g         = bsEurOptPrice(sigma) - self.V_price_
            #print("g = ", g)
            g_    = vega(sigma)
            #print("g_ = ", g_)
            sigma_new = sigma - g / g_
            #print("sigma_new = ", sigma_new)
            #e_=abs(sigma_new-sigma)
            e_ = abs(g)
            sigma = sigma_new;
            
            print('At iteration {0} the e_ value equals {1}'.format(n, e_))
            
            n = n + 1
            
        m = '''Implied volatility for OptionPrice= {} with a 
              Strike K={}, 
              Time to Maturity T= {}, 
              Risk-free rate r= {} 
              and Stock price at 0 S_init={} 
              equals sigma_imp = {:.7f}'''.format(self.V_price_, self.K, self.T_, self.r, self.S_init,sigma)
                    
        print(m)
        return sigma

################################################ ANALYSIS ###################################################

a = ImplVolCalc(V_price_ = V_price_, K = K, S_init = S_init, T_ = T_, r = r, sigmaInit = sigmaInit, CP = CP)
a.Eur_Call_BS_Price(sigmaInit)
a.Eur_vega(sigmaInit)


###### Verify correctness of calculation
sigma_imp = a.ImplVol(sigmaInit)
b = ImplVolCalc(V_price_ = V_price_, K = K, S_init = S_init, T_ = T_, r = r, sigmaInit = sigma_imp, CP = CP)
val = b.Eur_Call_BS_Price(sigma_imp)
print('European Option Price for implied volatility of {0} is equal to {1}'.format(sigma_imp, val))


################################################ CREATE A SMILE ###################################################
# Verify the CALL option prices for different strikes
K_list = np.arange(50, 210, 10)
Eur_Call_BS_Prices = []
for k in K_list:
    c = ImplVolCalc(V_price_ = V_price_, K = k, S_init = S_init, T_ = T_, r = r, sigmaInit = sigmaInit, CP = CP)
    Eur_Call_BS_Prices.append(c.Eur_Call_BS_Price(sigmaInit))
    
# Input market prices for European call option with different strikes
# ! The order of prices is sensible for a Call Option
# In case of a change to a PUT, a reordering would make more sense
V_price__list = [54, 50, 40, 31, 22.5, 15.5, 9.6, 5, 2.5, 1.2, 0.55, 0.285, 0.15, 0.09, 0.06, 0.045]
# Calc implied volatilities for each strike
Eur_Call_BS_Prices = dict(zip(K_list, Eur_Call_BS_Prices))
V_price__Prices = dict(zip(K_list, V_price__list))
Impl_Vols = []
for k in K_list:
    d = ImplVolCalc(V_price_ = V_price__Prices[k], K = k, S_init = S_init, T_ = T_, r = r, sigmaInit = sigmaInit, CP = CP)
    Impl_Vols.append(d.ImplVol(sigmaInit))
# Calculate Vols from BS model prices (check)
BS_Vols = []
for k in K_list:
    e = ImplVolCalc(V_price_ = Eur_Call_BS_Prices[k], K = k, S_init = S_init, T_ = T_, r = r, sigmaInit = sigmaInit, CP = CP)
    BS_Vols.append(e.ImplVol(sigmaInit))
    
# Plot Implied vs BS volatility
plt.plot(K_list, Impl_Vols)   
plt.plot(K_list, BS_Vols)   
plt.grid()
plt.xlabel("Strikes")
plt.ylabel("Implied Volatility from market prices vs BS model")
plt.legend(["Market", "BS model"])
    
