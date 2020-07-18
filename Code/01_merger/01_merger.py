#!/usr/bin/env python
# coding: utf-8

# # FIRST ASSIGNMENT
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

from class_plot_hist_fMT import plot_hist_fMT
from class_plot_hist_metal import plot_hist_metal
from class_fit_hist import fit_hist

plot_hist_fMT = plot_hist_fMT()
plot_hist_metal = plot_hist_metal()
fit_hist = fit_hist()

fMT = ['01','02','03','04','05','07','1']
metal = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012','0.016','0.02']
chunk = ['0','1','2','3','4']

df_fMT = []


for i in fMT:

    df_metal = []

    for j in metal:

        df_chunk = []

        for m in chunk:
            path = '/data1/collaborator/modB/simulations_fMT'+i+'/A5/'+j+'/chunk'+m
            #path = 'simulations_fMT'+i+'/A5/'+j+'/chunk'+m
            cols = pd.read_csv(path+"/mergers.out",sep="\s+",nrows=1,header=None).iloc[0].values[1:]
            df = pd.read_csv(path+"/mergers.out",sep="\s+",skiprows=1,header=None)
            df.columns = cols
            df = df.apply(pd.to_numeric, errors='ignore')
            df = df[df['k1form[7]']==14]
            df = df[df['k2form[9]']==14]
            # Ordering by m1>m2
            mask = df['m2form[10]']>df['m1form[8]']
            df['flip_mask'] = mask
            df['m1form[8]'] = np.where(mask, df['m2form[10]'], df['m1form[8]'])
            df['m2form[10]'] = np.where(mask, df['m1form[8]'], df['m2form[10]'])

            df_chunk.append(df)

        df_metal.append(pd.concat(df_chunk, ignore_index=True))

    # Concatenation of dataframes with metallicity 0.012, 0.016, 0.02
    df_metal_u = pd.concat(df_metal[-3:],ignore_index=True)
    df_metal = df_metal[:-3]
    
    df_metal.append(df_metal_u)

    df_fMT.append(df_metal)

#######################

metal_red = ['0.0002','0.0004','0.0008','0.0012','0.0016','0.002','0.004','0.006','0.008','0.012-0.016-0.02']

# Plot for different fMT
plot_hist_fMT.mass_1(df_fMT,fMT,metal_red)
plot_hist_fMT.mass_2(df_fMT,fMT,metal_red)
plot_hist_fMT.mass_chirp(df_fMT,fMT,metal_red)
plot_hist_fMT.mass_tot(df_fMT,fMT,metal_red)
plot_hist_fMT.q(df_fMT,fMT,metal_red)
plot_hist_fMT.t_merg(df_fMT,fMT,metal_red)

plot_hist_fMT.flip_fraction(df_fMT,fMT,metal_red)


# Plot for different metallicity
plot_hist_metal.mass_1(df_fMT,fMT,metal_red)
plot_hist_metal.mass_2(df_fMT,fMT,metal_red)
plot_hist_metal.mass_chirp(df_fMT,fMT,metal_red)
plot_hist_metal.mass_tot(df_fMT,fMT,metal_red)
plot_hist_metal.q(df_fMT,fMT,metal_red)
plot_hist_metal.t_merg(df_fMT,fMT,metal_red)

plot_hist_metal.flip_fraction(df_fMT,fMT,metal_red)

# Fit tmerg distribution with a straight line
metal = ['0.0002','0.0004']
fit_hist.t_merg(df_fMT,fMT,metal)