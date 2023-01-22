# Principal Component Analysis
# Decomposition of US TY Yield Curve (maturities 1M - 20Y)
# Data frequency: daily
# Sample: 2001 - 2018

############################################### LIBRARIES ############################################################
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import math
from functools import reduce
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import scale, StandardScaler
############################################### FUNCTIONS ############################################################
# remove timestamp
def rem_time(d):
    """
    Remove time stamp from a date in the index
    """
    s = ''
    s = str(d.year) + '-' + str(d.month) + '-' + str(d.day)
    return s

def plot_two_pcs(pc_1_ = 0, pc_2_ = 1, no_years_ = 1):
    """
    Plot two selected Principal Components against each other, split by year
    pc_1_ - index of the 1st PC (0 stands for the 1st, 1 for the 2nd etc)
    pc_2_ - index of the 2nd PC (0 stands for the 1st, 1 for the 2nd etc)
    """
    no_years = no_years_
    pc1 = pc_1_
    pc2 = pc_2_
    fig = plt.figure(figsize=(16,12))
    plt.title('Projection on {0}. and {1}. PC'.format(pc1 + 1,pc2 + 1))
    plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
    plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
        
    sc = plt.scatter(pca.loc[:,pc1],pca.loc[:,pc2], c=[d.year for d in pca.index], cmap='rainbow')
    cb = plt.colorbar(sc)
    cb.set_ticks(ticks=np.unique([d.year for d in pca.index])[::1])
    cb.set_ticklabels(np.unique([d.year for d in pca.index])[::1])

    for year in range(df.index.min().year,df.index.max().year,no_years):
        year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
        year_end = df.index[df.index.searchsorted(pd.datetime(year + no_years - 1,12,31))]
        
        plt.annotate('{0}'.format(year), xy=(pca.loc[year_start,pc1],pca.loc[year_start,pc2]), xytext=(pca.loc[year_start,pc1],pca.loc[year_start,pc2]))

    plt.show()

def plot_pcs_mrg_yrs(no_years_ = 3):
    """
    Plot Principal Components merged into year clusters, by TY tenor
    no_years_ - no of years in one plot
    """
    no_years = no_years_
    cols = 3
    num_years = df.index.max().year - df.index.min().year
    rows = math.ceil(num_years / cols)

    plt.figure(figsize=(24,(24/cols)*rows))

    colnum = 0
    rownum = 0
    for year in range(df.index.min().year,df.index.max().year + 1,no_years):
        year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
        year_end = df.index[min(df.index.searchsorted(pd.datetime(year+no_years-1,12,31)),len(df) - 1)]
        
        pca.fit(df.loc[year_start:year_end,:].values)
        pca_components = np.transpose(pca.components_)

        plt.subplot2grid((rows,cols), (rownum,colnum), colspan=1, rowspan=1)
        plt.title('{0} - {1}'.format(year_start.year, year_end.year))
        plt.xlim(0, len(pca_components)-1)
        plt.ylim(-0.5, 0.6)
        plt.xticks(range(len(pca_components)), dt.index, size='small')
        
        for i, comp in enumerate(pca.components_):
            plt.plot(pca_thr_comp.components_[i], label='{0}. PC'.format(i+1), color='#dddddd')
            plt.plot(comp, label='{0}. PC'.format(i+1))
        plt.legend(loc='upper right')
        
        if colnum != cols-1:
            colnum += 1
        else:
            colnum = 0
            rownum += 1
    plt.show()

def plot_kernel_pca_mrg_years(no_years_ = 1, pc_1_ = 0, pc_2_ = 1):
    """
    Plot Kernel Principal Components merged into year clusters, by TY tenor
    no_years_ - no of years in one plot
    """
    no_years = no_years_
    pc1 = pc_1_
    pc2 = pc_2_
    fig = plt.figure(figsize=(16,12))
    plt.title('Projection on {0}. and {1}. PC'.format(pc1+1,pc2+1))
    plt.axhline(y=0,c="grey",linewidth=1.0,zorder=0)
    plt.axvline(x=0,c="grey",linewidth=1.0,zorder=0)
        
    sc = plt.scatter(pca.loc[:,pc1],pca.loc[:,pc2], c=[d.year for d in pca.index], cmap='rainbow')
    cb = plt.colorbar(sc)
    cb.set_ticks(ticks=np.unique([d.year for d in pca.index])[::1])
    cb.set_ticklabels(np.unique([d.year for d in pca.index])[::1])

    for year in range(df.index.min().year,df.index.max().year + 1,no_years):
        year_start = df.index[df.index.searchsorted(pd.datetime(year,1,1))]
        year_end = df.index[min(df.index.searchsorted(pd.datetime(year+no_years - 1,12,31)), len(df) - 1)]
        
        plt.annotate('{0}'.format(year), xy=(pca.loc[year_start,pc1],pca.loc[year_start,pc2]), xytext=(pca.loc[year_start,pc1],pca.loc[year_start,pc2]))

    plt.show()
############################################### INPUT ############################################################
# Load the historical yeilds per tenor
df_1MO = pd.read_csv('...\\DGS1MO.csv', sep=',')
df_3MO = pd.read_csv('...\\DGS3MO.csv', sep=',')
df_6MO = pd.read_csv('...\\DGS6MO.csv', sep=',')
df_1 = pd.read_csv('...\\DGS1.csv', sep=',')
df_2 = pd.read_csv('...\\DGS2.csv', sep=',')
df_5 = pd.read_csv('...\\DGS5.csv', sep=',')
df_7 = pd.read_csv('...\\DGS7.csv', sep=',')
df_10 = pd.read_csv('...\\DGS10.csv', sep=',')
df_20 = pd.read_csv('...\\DGS20.csv', sep=',')
# Merge the time series into a historical Yield Curve
dfs = [df_1MO, df_3MO, df_6MO, df_1, df_2, df_5, df_7, df_10, df_20]
df = reduce(lambda left, right: pd.merge(left,right,on=['DATE'], how='left'), dfs)
df.columns = ["DATE", "1M", "3M", "6M", "1Y", "2Y", "5Y", "7Y", "10Y", "20Y"]
df['DATE'] = pd.to_datetime(df['DATE'],infer_datetime_format=True)
df.set_index('DATE', drop=True, inplace=True)
df.index.names = [None]
df.index = pd.to_datetime(df.index)
# Convert strings to floats
for x in df.columns:
    df[x] = pd.to_numeric(df[x], errors='coerce') / 100
# Drop all rows with NAs
df = df.dropna()
dt = df.transpose()
############################################### PLOTS ############################################################
# Visualizing the Dataset
plt.figure(figsize=(20,15))
plt.plot(df.index, df)
plt.xlim(df.index.min(), df.index.max())
# plt.ylim(0, 0.1)
plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
for i in range(df.index.min().year, df.index.max().year + 1):
    plt.axvline(x = df.index[df.index.searchsorted(pd.datetime(i,1,1)) - 1],
                c="grey", linewidth=0.5, zorder=0)
plt.show()

# Visualize Yield Curve per Years
cols = 6
num_years = df.index.max().year - df.index.min().year
rows = math.ceil(num_years/cols)
plt.figure(figsize = (24,(24 / cols)*rows))
plt.subplot2grid((rows,cols), (0,0), colspan=cols, rowspan=rows)
colnum = 0
rownum = 0
for y in range(df.index.min().year,df.index.max().year):
    year_start = df.index[df.index.searchsorted(pd.datetime(y,1,1))]
    year_end = df.index[df.index.searchsorted(pd.datetime(y,12,31))]
    year_start = rem_time(year_start)
    year_end = rem_time(year_end)
    
    plt.subplot2grid((rows,cols), (rownum,colnum), colspan=1, rowspan=1)
    plt.title('{0}'.format(y))
    plt.xlim(0, len(dt.index)-1)
    plt.ylim(np.nanmin(dt.values), np.nanmax(dt.values))
    plt.xticks(range(len(dt.index)), dt.index, size='small')
    
    plt.plot(dt.loc[:,year_start:year_end].values)
    
    if colnum != cols-1:
        colnum += 1
    else:
        colnum = 0
        rownum += 1
plt.show()
############################################### PCA ############################################################
# calculate the PCA (eigenvectors of the covariance matrix)
pca_thr_comp = PCA(n_components=3, copy=True, whiten=False)
# transform the dataset onto the first two eigenvectors
pca_thr_comp.fit(df)
pca = pd.DataFrame(pca_thr_comp.transform(df))
pca.index = df.index
for i,pc in enumerate(pca_thr_comp.explained_variance_ratio_):
    print('{0}.\t{1:2.2f}%'.format(i + 1,pc * 100.0))

fig = plt.figure(figsize=(16,10))
plt.title('First {0} PCA components'.format(np.shape(np.transpose(pca_thr_comp.components_))[-1]))

plt.plot(np.transpose(pca_thr_comp.components_), label=['1. PC', '2. PC', '3. PC'])
plt.legend(['1. PC', '2. PC', '3. PC'])
plt.show()

# plot the outcome
# PC 1 vs PC2, per year
plot_two_pcs(0, 1)
# PC 1 vs PC3, per year
plot_two_pcs(0, 2)
# PC 2 vs PC3, per year
plot_two_pcs(1, 2)

# # Principal Components, combine years into clusters
pca = PCA(n_components=2, copy=True, whiten=False)
# no years = 3
plot_pcs_mrg_yrs(no_years_ = 3)
# no years = 4
plot_pcs_mrg_yrs(no_years_ = 4)
# no years = 5
plot_pcs_mrg_yrs(no_years_ = 5)
# no years = 6
plot_pcs_mrg_yrs(no_years_ = 6)

# FIT
model_fit = pd.DataFrame(0, index=np.arange(len(df)), columns=df.columns)
model_fit.index = df.index
pca_comps = pca_thr_comp.components_[:3,:]
pca_comps = pd.DataFrame(pca_comps)
pca_comps.columns = df.columns
for i in range(df.shape[0]):
    for j in df.columns:
        model_fit[j].iloc[i] = pca.iloc[i][0] * pca_comps.iloc[0][j] + \
                               pca.iloc[i][1] * pca_comps.iloc[1][j] + \
                               pca.iloc[i][2] * pca_comps.iloc[2][j]

# Visualizing the values constructed from PCA
plt.figure(figsize=(20,15))
plt.plot(model_fit.index, df)
plt.xlim(model_fit.index.min(), model_fit.index.max())
# plt.ylim(0, 0.1)
plt.axhline(y=0,c="grey",linewidth=0.5,zorder=0)
for i in range(model_fit.index.min().year, model_fit.index.max().year + 1):
    plt.axvline(x = model_fit.index[model_fit.index.searchsorted(pd.datetime(i,1,1)) - 1],
                c="grey", linewidth=0.5, zorder=0)
plt.show()
############################################### KERNEL PCA ############################################################
# calculate the PCA (Eigenvectors & Eigenvalues of the covariance matrix)
kern_pca = KernelPCA(n_components=3,
                 kernel='rbf',
                 gamma = 4, # default 1/n_features
                 kernel_params=None,
                 fit_inverse_transform=True,
                 eigen_solver='auto',
                 tol=0,
                 max_iter=None)

# transform the dataset onto the first two eigenvectors
kern_pca.fit(df)
pca = pd.DataFrame(kern_pca.transform(df))
pca.index = df.index

# plot the result
plot_kernel_pca_mrg_years(no_years_ = 1, pc_1_ = 0, pc_2_ = 1)
plot_kernel_pca_mrg_years(no_years_ = 1, pc_1_ = 0, pc_2_ = 2)
plot_kernel_pca_mrg_years(no_years_ = 1, pc_1_ = 1, pc_2_ = 2)


############################################### CHECK THE FIT ############################################################
trans_fit = pd.DataFrame(kern_pca.X_transformed_fit_)
trans_fit.index = df.index

# PC1: Level
fig = plt.figure(figsize=(16,10))
plt.title('First PCA component vs level of 1M US TY times 5')
plt.plot(trans_fit[0])
plt.plot(df["1M"] * 5)
plt.show()
# PC2: Slope
fig = plt.figure(figsize=(16,10))
plt.title('Second PCA component vs 20Y / 3M US TY Spread times 3')
plt.plot(trans_fit[1])
plt.plot((df["20Y"] - df["1M"]) * 2)
plt.show()
# PC3: Curvature
fig = plt.figure(figsize=(16,10))
plt.title('Thrid PCA component vs 2 * 5Y US TY - 20Y US TY - 6M US TY')
plt.plot(trans_fit[2])
plt.plot(- (df["20Y"] + df["6M"] - 2 * df["5Y"]))
plt.show()
