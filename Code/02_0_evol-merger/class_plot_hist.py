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



class plot_hist:


        def mass(self,df,fMT,metal,ce_str,m_str): #(dataframe,fMT,metal)
    
                fig = plt.figure(figsize=(30,30))
                ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of mass '+m_str+' '+ce_str, fontsize=30)        
                col = sns.color_palette(palette.viridis(len(metal)))

                for i in range(0,len(fMT)):
                        for j in range(0,len(metal)):               
                                #edge = np.logspace(np.log10(df[i][j]['mt1[4]'].min()), np.log10(df[i][j]['mt1[4]'].max()), 30)
                                sns.distplot(df[i][j][m_str],kde=False,norm_hist=False,label=metal[j], 
                                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color=col[j])

                        ax[i].set_title('Distribution of mass '+m_str+' for fMT={}'.format(fMT[i]),fontsize=18)
                        ax[i].set_xlabel('m $[M_\odot]$',fontsize=18)
                        ax[i].set_ylabel('Counts',fontsize=18)
                        #ax[i].set_xscale('log')
                        ax[i].legend(title='Metallicity',fontsize=14,title_fontsize=16,loc='upper right')
                        ax[i].tick_params(axis="x", labelsize=16)
                        ax[i].tick_params(axis="y", labelsize=16)
                        ax[i].grid(True, which="both", ls="-",color='0.93')
                        ax[i].set_axisbelow(True)

                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')

                plt.savefig('hist_fMT/hist_'+m_str+'_fMT_'+ce_str+'.pdf', format='pdf')
                plt.close()
        
        def dtimes(self,df,fMT,metal,ce_str): #(dataframe,fMT,metal)

                fig = plt.figure(figsize=(30,30))
                ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of delay times ' + ce_str,fontsize=30)        
                col = sns.color_palette(palette.viridis(len(metal)))

                for i in range(0,len(fMT)):
                        for j in range(0,len(metal)):               
                                #edge = np.logspace(np.log10(df[i][j]['t_step[1]'].min()), np.log10(df[i][j]['t_step[1]'].max()), 30)
                                sns.distplot(df[i][j]['dtimes'],kde=False,norm_hist=False,label=metal[j], 
                                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color=col[j])

                        ax[i].set_title('Distribution of delay times for fMT={}'.format(fMT[i]),fontsize=18)
                        ax[i].set_xlabel('t $[Myr]$',fontsize=18)
                        ax[i].set_ylabel('Counts',fontsize=18)
                        #ax[i].set_xscale('log')
                        #ax[i].set_yscale('log')
                        ax[i].legend(title='Metallicity',fontsize=14,title_fontsize=16,loc='upper right')
                        ax[i].tick_params(axis="x", labelsize=16)
                        ax[i].tick_params(axis="y", labelsize=16)
                        ax[i].grid(True, which="both", ls="-",color='0.93')
                        ax[i].set_axisbelow(True)
        
                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')

                plt.savefig('hist_fMT/hist_dtimes_fMT_'+ce_str+'.pdf', format='pdf')
                plt.close()



        def frac_CE(self,df_CE,df_nCE,fMT,metal): #(dataframe,fMT,metal)

                num_fMT_CE = []
                num_fMT_nCE = []

                for i in range(len(fMT)):
                        num_CE = 0
                        num_nCE = 0
                        for j in range(len(metal)):
                                num_CE += df_CE[i][j].shape[0]
                                num_nCE += df_nCE[i][j].shape[0]
                        num_fMT_CE.append(num_CE)
                        num_fMT_nCE.append(num_nCE)

                frac_fMT_CE = [a/(a+b) for a,b in zip(num_fMT_CE,num_fMT_nCE)]
                frac_fMT_nCE = [b/(a+b) for a,b in zip(num_fMT_CE,num_fMT_nCE)]

                fMT = [0.1,0.2,0.3,0.4,0.5,0.7,1]
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                fig.suptitle('Plots of fraction of CE BH system',fontsize=30)

                ax.plot(fMT,frac_fMT_CE,'.',color='#440154')
                ax.set_xticks([i/10 for i in range (1,11)])
                ax.set_title('Fraction of CE BH system')
                ax.set_xlabel('fMT')
                ax.set_ylabel('Fraction')
                ax.grid(color='0.93')
        
                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')

                plt.savefig('hist_fMT/frac_CE.pdf', format='pdf')
                plt.close()