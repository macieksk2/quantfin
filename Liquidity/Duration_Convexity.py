# -*- coding: utf-8 -*-
"""
-> Duration, Modified Duration
-> Convexity

Calculated for three cases:
    1. Bond with fixed coupons
    2. Loan with equal principal
    3. Loan with equal CFs
    
Also, compare the change in FV of a bonf after a shift in interest rate:
    1. 1st order approximation (Modified Duration)
    2. 2nd order approximation (Modified Duration + Convexity)
    
"""
import copy as copylib
import numpy as np
import os
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import enum
import scipy.stats as st
import datetime
from datetime import datetime

### INPUTS
p_ = 1000
r_ = 0.06
d_ = 0.06
start_date = "31-12-2014"
mat_date = "31-12-2019"

# CLASSES
class DebtCFs:
    def __init__(self, p_, r_, d_, start_date, mat_date):
        self.p_ = p_
        self.r_ = r_
        self.d_ = d_
        self.start_date = start_date
        self.mat_date = mat_date
        
    def Duration(self, dcf, dcf_t):
        return sum(dcf_t) / sum(dcf)

    def Mod_Duration(self, dcf, dcf_t):
        return sum(dcf_t) / sum(dcf) / (1 + self.d_)

    def Convexity(self, dcf, dcf_t_t_1):
        return sum(dcf_t_t_1) / sum(dcf) / (1 + self.d_) ** 2
    
    def PreProcess(self):
        col_names = ["Outstanding principal", "Interest charged", "Principal paid", "CF", "DCF", "DCF*t", "DCF*t*(t+1)"]
        mat_yr = datetime.strptime(self.mat_date, '%d-%m-%Y').date().year
        start_yr = datetime.strptime(self.start_date, '%d-%m-%Y').date().year   
        n_rows = mat_yr - start_yr
        return col_names, mat_yr, start_yr, n_rows

    def PostProcess(self, data, col_names):
        out = pd.DataFrame(data)
        out.columns = col_names
        durr = self.Duration(out["DCF"], out["DCF*t"])
        mod_durr = self.Mod_Duration(out["DCF"], out["DCF*t"])
        conv = self.Convexity(out["DCF"], out["DCF*t*(t+1)"])
        return out, durr, mod_durr, conv
    
    def DCF(self, cf, row):
        dcf = cf / (1 + self.d_) ** (row + 1)
        dcf_t = dcf * (row + 1)
        dcf_t_t_1 = dcf * (row + 1) * (row + 2)
        return dcf, dcf_t, dcf_t_t_1
    
    def Bond(self):
        """
        Fixed coupon bond
        """
        data = []
        col_names, mat_yr, start_yr, n_rows = self.PreProcess()
        for row in range(n_rows):
            if row == 0:
                out_princip = self.p_
            else:
                out_princip = out_princip
            int_chrg = out_princip * self.r_
            if row + 1 == n_rows:
                princip_pd = out_princip
            else:
                princip_pd = 0
            cf = int_chrg + princip_pd
            dcf, dcf_t, dcf_t_t_1 = self.DCF(cf, row)
            row = [out_princip, int_chrg, princip_pd, cf, dcf, dcf_t, dcf_t_t_1]
            data.append(row)
        
        return self.PostProcess(data, col_names)

    def Loan_Eq_Princip(self):
        """
        Loan with the same principal paid in each period 
        """
        data = []
        col_names, mat_yr, start_yr, n_rows = self.PreProcess()
        princip_pd = self.p_ / n_rows
        for row in range(n_rows):
            if row == 0:
                out_princip = self.p_
            else:
                out_princip = out_princip - princip_pd
            int_chrg = out_princip * self.r_
            cf = int_chrg + princip_pd
            dcf, dcf_t, dcf_t_t_1 = self.DCF(cf, row)
            row = [out_princip, int_chrg, princip_pd, cf, dcf, dcf_t, dcf_t_t_1]
            data.append(row)
        
        return self.PostProcess(data, col_names)

    def Loan_Eq_CFs(self):
        """
        Loan with the same CF in each period 
        """
        data = []
        col_names, mat_yr, start_yr, n_rows = self.PreProcess()
        cf = self.p_ * self.r_ / (1 - (1 + self.r_) ** (-n_rows))
        for row in range(n_rows):
            princip_pd = self.p_ / n_rows
            if row == 0:
                out_princip = self.p_
            else:
                out_princip = out_princip - princip_pd
            int_chrg = out_princip * self.r_
            princip_pd = cf - int_chrg
            dcf, dcf_t, dcf_t_t_1 = self.DCF(cf, row)
            row = [out_princip, int_chrg, princip_pd, cf, dcf, dcf_t, dcf_t_t_1]
            data.append(row)
        
        return self.PostProcess(data, col_names)

# INITATE CLASSES
b_ = DebtCFs(p_, r_, d_, start_date, mat_date)
bond_cfs, bond_durr, bond_mod_durr, bond_conv = b_.Bond()
loan_eq_princip_cfs, loan_eq_princip_durr, loan_eq_princip_mod_durr, loan_eq_princip_conv = b_.Loan_Eq_Princip()
loan_eq_cf_cfs, loan_eq_cf_cfs_durr, loan_eq_cf_cfs_mod_durr, loan_eq_cf_cfs_conv = b_.Loan_Eq_CFs()

# COMPARE DURATION AND CONVEXITY APPROXIMATION OF BOND FAIR VALUE AFTER A CHANGE IN INTEREST RATE
p_ = 1000
r_ = 0.06
d_ = 0.05
start_date = "31-12-2014"
mat_date = "31-12-2019"
int_chg = 0.01

c_ = DebtCFs(p_, r_, d_, start_date, mat_date)
bond_cfs, bond_durr, bond_mod_durr, bond_conv = c_.Bond()

fst_order_approx = - int_chg * bond_mod_durr
snd_order_approx = - int_chg * bond_mod_durr + int_chg ** 2 * bond_conv / 2

fair_val_fst_order = sum( bond_cfs["DCF"]) * (1 + fst_order_approx)
fair_val_snd_order = sum( bond_cfs["DCF"]) * (1 + snd_order_approx)
fair_val_true =      sum([bond_cfs["CF"][x] / (1 + d_ + int_chg) ** (x + 1) for x in range(len(bond_cfs["CF"]))])

print("FV, 1st order approximation", fair_val_fst_order)
print("FV, 2nd order approximation", fair_val_snd_order)
print("True FV", fair_val_true)
