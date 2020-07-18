#!/usr/bin/env python
# coding: utf-8

# # SECOND ASSIGNMENT
#
# **GROUP: EARTHQUAKES**
#
# Guadagnini Michele - 1230663
#
# Lambertini Alessandro - 1242885
#
# Pagano Alice - 1236916
#
# Puppin Michele - 1227474

# Import main packages 
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh.palettes as palette # For palette of colors

import time
import os
import glob
from joblib import Parallel, delayed
import dask.dataframe as dd

from class_read_parallel import read_parallel
from class_plot_hist import plot_hist

read_parallel = read_parallel()
plot_hist = plot_hist()

start_total_time = time.time()

fMT = ['01','02','03','04','05','07','1']
metal = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunk = ['0','1','2','3','4']

start_time = time.time()

# Read all chunks in parallel
single_chunk = Parallel(n_jobs=64)(delayed(read_parallel.read_chunks)(i, j, m) for i in fMT for j in metal for m in chunk)

end_time = time.time()
print("Times in seconds: "+str(end_time-start_time))


start_time = time.time()

# Filter data
start_time = time.time()

data_BH = [read_parallel.filter_chunks_BH(data) for data in single_chunk]
data_CE = [read_parallel.filter_chunks_CE(data) for data in data_BH]
data_nCE = [read_parallel.filter_chunks_nCE(data) for data in data_BH]

data_CE_initial = [read_parallel.filter_chunks_initial(data) for data in data_CE]
data_nCE_initial = [read_parallel.filter_chunks_initial(data) for data in data_nCE]

data_CE_final = [read_parallel.filter_chunks_final(data) for data in data_CE]
data_nCE_final = [read_parallel.filter_chunks_final(data) for data in data_nCE]

data_CE_dtimes = [read_parallel.filter_chunks_dtimes(data) for data in data_CE]
data_nCE_dtimes = [read_parallel.filter_chunks_dtimes(data) for data in data_nCE]

end_time = time.time()
print("Times in seconds: "+str(end_time-start_time))



# Organization of data in two new dataframe
data_CE_filtered = []
data_nCE_filtered = []

for i in range(len(data_CE)):
    frame_CE = {'m1_init' : np.array(data_CE_initial[i]['mt1[4]']),
                'm2_init' : np.array(data_CE_initial[i]['mt2[18]']),
                'm1_fin' : np.array(data_CE_final[i]['mt1[4]']),
                'm2_fin' : np.array(data_CE_final[i]['mt2[18]']),
                'dtimes' : np.array(data_CE_dtimes[i]['t_step[1]'])}
    dataframe_CE = pd.DataFrame(frame_CE) 
    
    frame_nCE = {'m1_init' : np.array(data_nCE_initial[i]['mt1[4]']),
                 'm2_init' : np.array(data_nCE_initial[i]['mt2[18]']),
                 'm1_fin' : np.array(data_nCE_final[i]['mt1[4]']),
                 'm2_fin' : np.array(data_nCE_final[i]['mt2[18]']),
                 'dtimes' : np.array(data_nCE_dtimes[i]['t_step[1]'])}
    dataframe_nCE = pd.DataFrame(frame_nCE) 
    
    data_CE_filtered.append(dataframe_CE)
    data_nCE_filtered.append(dataframe_nCE)



# Organization of data 
start_time = time.time()
df_fMT_CE, df_fMT_nCE = read_parallel.organize_data(data_CE_filtered, data_nCE_filtered,fMT,metal,chunk)
end_time = time.time()
print("Times in seconds: "+str(end_time-start_time))


# Plot of the histograms

start_time = time.time()

# CE e nCE in the same plot for metallicities until 0.002 (included)
plot_hist.mass(df_fMT_CE,df_fMT_nCE,fMT,'m1_init')
plot_hist.mass(df_fMT_CE,df_fMT_nCE,fMT,'m2_init')
plot_hist.mass(df_fMT_CE,df_fMT_nCE,fMT,'m1_fin')
plot_hist.mass(df_fMT_CE,df_fMT_nCE,fMT,'m2_fin')
plot_hist.dtimes(df_fMT_CE,df_fMT_nCE,fMT)
plot_hist.mass_ratio(df_fMT_CE,df_fMT_nCE,fMT)

# Plot of final masses flipped with m1>m2
plot_hist.mass_final_flip(df_fMT_CE,df_fMT_nCE,fMT,'m1_fin')
plot_hist.mass_final_flip(df_fMT_CE,df_fMT_nCE,fMT,'m2_fin')

end_time = time.time()
print("Times in seconds: "+str(end_time-start_time))

# Total time in second 
end_total_time = time.time()
print("Total time in seconds: "+str(end_total_time-start_total_time))