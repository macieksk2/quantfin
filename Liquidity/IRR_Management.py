# -*- coding: utf-8 -*-
"""
-> Interest Rate Risk Management (Balance Sheet, ALM)
-> Hedge bank's exposure to interest rate movement either:
    1. Externally (e.g. via IRS)
    2. Internally (change a mix of rate sens / insensitive assets / liabilities)
    
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
from scipy.optimize import minimize

### INPUTS
items = pd.read_excel("...\\IRR_Management.xlsx")

# Create ALM Assets / Liab columns
items["Int_Hedg_ALM_Ass"] = 0
items["Int_Hedg_ALM_Liab"] = 0

# Add calc rows
items.loc[len(items)] = ["Repricing gap (net interest income exposure), $ mln", 0, 0, 0]
items.loc[len(items)] = ["Net interest income change, $ mln", 0, 0, 0]
items.loc[len(items)] = ["Asset fair value exposure, $ mln", 0, 0, 0]
items.loc[len(items)] = ["Liabilities fair value exposure, $ mln", 0, 0, 0]
items.loc[len(items)] = ["Economic value of equity exposure, $ mln", 0, 0, 0]
items.loc[len(items)] = ["Economic value of equity change, $ mln", 0, 0, 0]
items.loc[len(items)] = ["Total exposure", 0, 0, 0]
items.loc[len(items)] = ["Total assets", 0, 0, 0]
items.loc[len(items)] = ["Total liabilities", 0, 0, 0]
items.loc[len(items)] = ["Interest rate swap notional principal", 0, "-", "-"]
items.loc[len(items)] = ["Position", 0, "-", "-"]

# Perform calculations
### External Hedging
# Repricing gap
items['External hedging'].loc[items["Items"] == "Repricing gap (net interest income exposure), $ mln"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['External hedging']) - \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['External hedging'])
# Net interest income change, $ mln
items['External hedging'].loc[items["Items"] == "Net interest income change, $ mln"] = \
    float(items[items["Items"] == "Interest rate change, p.p."]['External hedging']) * \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['External hedging'])
# Asset fair value exposure, $ mln
items['External hedging'].loc[items["Items"] == "Asset fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive assets, $ mln"]['External hedging']) * \
      float(items[items["Items"] == "Modified duration, assets"]['External hedging'])
# Liabilities fair value exposure, $ mln
items['External hedging'].loc[items["Items"] == "Liabilities fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive liabilities, $ mln"]['External hedging']) * \
      float(items[items["Items"] == "Modified duration, liabilities"]['External hedging'])
# Economic value of equity exposure, $ mln
items['External hedging'].loc[items["Items"] == "Economic value of equity exposure, $ mln"] = \
    float(items[items["Items"] == "Asset fair value exposure, $ mln"]['External hedging']) - \
    float(items[items["Items"] == "Liabilities fair value exposure, $ mln"]['External hedging'])
# Liabilities fair value exposure, $ mln
items['External hedging'].loc[items["Items"] == "Economic value of equity change, $ mln"] = \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['External hedging']) * \
    float(items[items["Items"] == "Interest rate change, p.p."]['External hedging'])
# Total exposure
items['External hedging'].loc[items["Items"] == "Total exposure"] = \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['External hedging']) + \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['External hedging'])
# Total assets
items['External hedging'].loc[items["Items"] == "Total assets"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['External hedging']) + \
    float(items[items["Items"] == "Rate insensitive assets, $ mln"]['External hedging'])
# Total liabilities
items['External hedging'].loc[items["Items"] == "Total liabilities"] = \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['External hedging']) + \
    float(items[items["Items"] == "Rate insensitive liabilities, $ mln"]['External hedging'])
# Interest rate swap notional principal
items['External hedging'].loc[items["Items"] == "Interest rate swap notional principal"] = \
    abs(float(items[items["Items"] == "Total exposure"]['External hedging']))
# Position
# For external hedging:
# if exposure > 0 -> pay floating and receive fixed
# if exposure < 0 -> pay fixed and receive floating
# notional principal = exposure
# Combining NII and EVE exposures can immunise against both!

if(float(items[items["Items"] == "Total exposure"]['External hedging'] < 0)):
   items['External hedging'].loc[items["Items"] == "Position"] = "pay fixed, receive floating" 
else:
   items['External hedging'].loc[items["Items"] == "Position"] = "pay floating, receive fixed"

### Internal Hedging
# For Internal Hedging:
# Delta RSA = - Total Exposure / (1 + MD Assets)
# Delta RSL = - Total Exposure / (1 + MD Liabilities)

# First copy the same cells from External Hedging
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Total assets"] = \
   items['External hedging'].loc[items["Items"] == "Total assets"] 
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Total assets"] = \
   items['External hedging'].loc[items["Items"] == "Total assets"] 
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Total liabilities"] = \
   items['External hedging'].loc[items["Items"] == "Total liabilities"] 
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Total liabilities"] = \
   items['External hedging'].loc[items["Items"] == "Total liabilities"] 
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Modified duration, assets"] = \
   items['External hedging'].loc[items["Items"] == "Modified duration, assets"] 
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Modified duration, assets"] = \
   items['External hedging'].loc[items["Items"] == "Modified duration, assets"] 
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Modified duration, liabilities"] = \
   items['External hedging'].loc[items["Items"] == "Modified duration, liabilities"] 
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Modified duration, liabilities"] = \
   items['External hedging'].loc[items["Items"] == "Modified duration, liabilities"] 

# Internal Hedging calculations   
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Rate sensitive assets, $ mln"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['External hedging']) - \
    float(items[items["Items"] == "Total exposure"]['External hedging']) / (1 + float(items[items["Items"] == "Modified duration, assets"]['External hedging']))
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Rate sensitive assets, $ mln"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['External hedging'])
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Rate sensitive liabilities, $ mln"] = \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['External hedging'])
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Rate sensitive liabilities, $ mln"] = \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['External hedging']) + \
    float(items[items["Items"] == "Total exposure"]['External hedging']) / (1 + float(items[items["Items"] == "Modified duration, liabilities"]['External hedging']))

items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Repricing gap (net interest income exposure), $ mln"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['Int_Hedg_ALM_Ass']) - \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['Int_Hedg_ALM_Ass'])
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Repricing gap (net interest income exposure), $ mln"] = \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['Int_Hedg_ALM_Liab']) - \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['Int_Hedg_ALM_Liab'])
    
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Interest rate change, p.p."] = items['External hedging'].loc[items["Items"] == "Interest rate change, p.p."]
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Interest rate change, p.p."] = items['External hedging'].loc[items["Items"] == "Interest rate change, p.p."]

items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Net interest income change, $ mln"] = \
    float(items[items["Items"] == "Interest rate change, p.p."]['Int_Hedg_ALM_Ass']) * \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['Int_Hedg_ALM_Ass'])   
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Net interest income change, $ mln"] = \
    float(items[items["Items"] == "Interest rate change, p.p."]['Int_Hedg_ALM_Liab']) * \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['Int_Hedg_ALM_Liab'])
    
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Rate insensitive assets, $ mln"] = \
    float(items[items["Items"] == "Total assets"]['Int_Hedg_ALM_Ass']) - \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['Int_Hedg_ALM_Ass']) 
          
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Rate insensitive assets, $ mln"] = \
    float(items[items["Items"] == "Total assets"]['Int_Hedg_ALM_Liab']) - \
    float(items[items["Items"] == "Rate sensitive assets, $ mln"]['Int_Hedg_ALM_Liab'])
                                                                  
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Rate insensitive liabilities, $ mln"] = \
    float(items[items["Items"] == "Total liabilities"]['Int_Hedg_ALM_Ass']) - \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['Int_Hedg_ALM_Ass']) 
          
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Rate insensitive liabilities, $ mln"] = \
    float(items[items["Items"] == "Total liabilities"]['Int_Hedg_ALM_Liab']) - \
    float(items[items["Items"] == "Rate sensitive liabilities, $ mln"]['Int_Hedg_ALM_Liab'])
    
# Asset fair value exposure, $ mln
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Asset fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive assets, $ mln"]['Int_Hedg_ALM_Ass']) * \
      float(items[items["Items"] == "Modified duration, assets"]['Int_Hedg_ALM_Ass'])
# Liabilities fair value exposure, $ mln
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Liabilities fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive liabilities, $ mln"]['Int_Hedg_ALM_Ass']) * \
      float(items[items["Items"] == "Modified duration, liabilities"]['Int_Hedg_ALM_Ass'])
# Economic value of equity exposure, $ mln
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Economic value of equity exposure, $ mln"] = \
    float(items[items["Items"] == "Asset fair value exposure, $ mln"]['Int_Hedg_ALM_Ass']) - \
    float(items[items["Items"] == "Liabilities fair value exposure, $ mln"]['Int_Hedg_ALM_Ass'])
    
# Asset fair value exposure, $ mln
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Asset fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive assets, $ mln"]['Int_Hedg_ALM_Liab']) * \
      float(items[items["Items"] == "Modified duration, assets"]['Int_Hedg_ALM_Liab'])
# Liabilities fair value exposure, $ mln
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Liabilities fair value exposure, $ mln"] = \
    - float(items[items["Items"] == "Rate insensitive liabilities, $ mln"]['Int_Hedg_ALM_Liab']) * \
      float(items[items["Items"] == "Modified duration, liabilities"]['Int_Hedg_ALM_Liab'])
# Economic value of equity exposure, $ mln
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Economic value of equity exposure, $ mln"] = \
    float(items[items["Items"] == "Asset fair value exposure, $ mln"]['Int_Hedg_ALM_Liab']) - \
    float(items[items["Items"] == "Liabilities fair value exposure, $ mln"]['Int_Hedg_ALM_Liab'])
    
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Economic value of equity change, $ mln"] = \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['Int_Hedg_ALM_Ass']) * \
    float(items[items["Items"] == "Interest rate change, p.p."]['Int_Hedg_ALM_Ass'])
# Total exposure
items['Int_Hedg_ALM_Ass'].loc[items["Items"] == "Total exposure"] = \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['Int_Hedg_ALM_Ass']) + \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['Int_Hedg_ALM_Ass'])
    
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Economic value of equity change, $ mln"] = \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['Int_Hedg_ALM_Liab']) * \
    float(items[items["Items"] == "Interest rate change, p.p."]['Int_Hedg_ALM_Liab'])
# Total exposure
items['Int_Hedg_ALM_Liab'].loc[items["Items"] == "Total exposure"] = \
    float(items[items["Items"] == "Repricing gap (net interest income exposure), $ mln"]['Int_Hedg_ALM_Liab']) + \
    float(items[items["Items"] == "Economic value of equity exposure, $ mln"]['Int_Hedg_ALM_Liab'])
