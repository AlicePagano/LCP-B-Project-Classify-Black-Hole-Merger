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


        def mass(self,df_CE,df_nCE,fMT,m_str): #(dataframe,fMT,metal)
    
                fig = plt.figure(figsize=(30,30))
                ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of mass '+m_str+'for metallicities <= 0.002', fontsize=30)        

                for i in range(0,len(fMT)):
                            
                    sns.distplot(df_CE[i][0][m_str],kde=False,norm_hist=True,label='CE', 
                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#440154')
                    
                    sns.distplot(df_nCE[i][0][m_str],kde=False,norm_hist=True,label='nCE', 
                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#29788E')
                    
                    ax[i].set_title('Distribution of mass '+m_str+' for fMT={}'.format(fMT[i]),fontsize=18)
                    ax[i].set_xlabel('m $[M_\odot]$',fontsize=18)
                    ax[i].set_ylabel('Density',fontsize=18)
                    #ax[i].set_xscale('log')
                    ax[i].legend(title='Mass',fontsize=14,title_fontsize=16,loc='upper right')
                    ax[i].tick_params(axis="x", labelsize=16)
                    ax[i].tick_params(axis="y", labelsize=16)
                    ax[i].grid(True, which="both", ls="-",color='0.93')
                    ax[i].set_axisbelow(True)
                    
                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')
                plt.savefig('hist_fMT/hist_'+m_str+'_fMT.pdf', format='pdf')
                plt.close()
        

        def mass_final_flip(self,df_CE,df_nCE,fMT,m_str): #(dataframe,fMT,metal)
    
                fig = plt.figure(figsize=(30,30))
                ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of mass '+m_str+'for metallicities <= 0.002', fontsize=30)        

                for i in range(0,len(fMT)):
                
                    # Ordering by m1>m2
                    mask_CE = df_CE[i][0]['m2_fin']>df_CE[i][0]['m1_fin']
                    mask_nCE = df_nCE[i][0]['m2_fin']>df_nCE[i][0]['m1_fin']
                    
                    if(m_str=='m1_fin'):
                        flip_CE = np.where(mask_CE, df_CE[i][0]['m2_fin'], df_CE[i][0]['m1_fin'])
                        flip_nCE = np.where(mask_nCE, df_nCE[i][0]['m2_fin'], df_nCE[i][0]['m1_fin'])

                    if(m_str=='m2_fin'):
                        flip_CE  = np.where(mask_CE, df_CE[i][0]['m1_fin'], df_CE[i][0]['m2_fin'])
                        flip_nCE  = np.where(mask_nCE, df_nCE[i][0]['m1_fin'], df_nCE[i][0]['m2_fin'])


                    sns.distplot(flip_CE,kde=False,norm_hist=True,label='CE', 
                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#440154')
                    
                    sns.distplot(flip_nCE,kde=False,norm_hist=True,label='nCE', 
                                        hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#29788E')
                    
                    ax[i].set_title('Distribution of mass '+m_str+' for fMT={}'.format(fMT[i]),fontsize=18)
                    ax[i].set_xlabel('m $[M_\odot]$',fontsize=18)
                    ax[i].set_ylabel('Density',fontsize=18)
                    #ax[i].set_xscale('log')
                    ax[i].legend(title='Mass',fontsize=14,title_fontsize=16,loc='upper right')
                    ax[i].tick_params(axis="x", labelsize=16)
                    ax[i].tick_params(axis="y", labelsize=16)
                    ax[i].grid(True, which="both", ls="-",color='0.93')
                    ax[i].set_axisbelow(True)
                    
                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')
                plt.savefig('hist_fMT/hist_'+m_str+'_fMT_flip.pdf', format='pdf')
                plt.close()


        def dtimes(self,df_CE,df_nCE,fMT): #(dataframe,fMT,metal)

                fig = plt.figure(figsize=(30,30))
                ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of delay times for metallicities <= 0.002',fontsize=30)        

                for i in range(0,len(fMT)):              
                    #edge = np.logspace(np.log10(df[i][j]['t_step[1]'].min()), np.log10(df[i][j]['t_step[1]'].max()), 30)
                    sns.distplot(df_CE[i][0]['dtimes'],kde=False,norm_hist=True,label='CE', 
                                            hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#440154')
                    
                    sns.distplot(df_nCE[i][0]['dtimes'],kde=False,norm_hist=True,label='nCE', 
                                            hist_kws={"histtype": "step","lw": 2},ax=ax[i],color='#29788E')

                    ax[i].set_title('Distribution of delay times for fMT={}'.format(fMT[i]),fontsize=18)
                    ax[i].set_xlabel('t $[Myr]$',fontsize=18)
                    ax[i].set_ylabel('Density',fontsize=18)
                    #ax[i].set_xscale('log')
                    #ax[i].set_yscale('log')
                    ax[i].legend(title='Delay times',fontsize=14,title_fontsize=16,loc='upper right')
                    ax[i].tick_params(axis="x", labelsize=16)
                    ax[i].tick_params(axis="y", labelsize=16)
                    ax[i].grid(True, which="both", ls="-",color='0.93')
                    ax[i].set_axisbelow(True)
        
                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')

                plt.savefig('hist_fMT/hist_dtimes_fMT.pdf', format='pdf')
                plt.close()



        def mass_ratio(self,df_CE,df_nCE,fMT): #(dataframe,fMT,metal)
    
                fig = plt.figure(figsize=(20,70))
                ax = [fig.add_subplot(7,2,s) for s in range(1,2*len(fMT)+1)] #nrows,ncols,index
                fig.suptitle('Plots of distribution of mass ratio', fontsize=30, )        
                col = palette.viridis(12)
                fig.subplots_adjust(top=0.95)
                
                for i in range(0,len(fMT)):
                            
                    q_initial_CE = df_CE[i][0]['m2_init']/df_CE[i][0]['m1_init']
                    q_initial_nCE = df_nCE[i][0]['m2_init']/df_nCE[i][0]['m1_init']
                    q_final_CE = df_CE[i][0]['m2_fin']/df_CE[i][0]['m1_fin']
                    q_final_nCE = df_nCE[i][0]['m2_fin']/df_nCE[i][0]['m1_fin']
                    
                    sns.distplot(q_initial_CE,kde=False,norm_hist=True,label='initial CE n='+len(q_initial_CE), 
                            hist_kws={"histtype": "step","lw": 2},ax=ax[2*i],color=col[0])
                    
                    sns.distplot(q_final_CE,kde=False,norm_hist=True,label='final CE n='+len(q_final_CE), 
                            hist_kws={"histtype": "step","lw": 2},ax=ax[2*i],color=col[4])

                    ax[2*i].set_title('Distribution of initial and final mass ratio CE for fMT={}'.format(fMT[i]),fontsize=18)
                    ax[2*i].set_xlabel('q',fontsize=18)
                    ax[2*i].set_ylabel('Density',fontsize=18)
                    #ax[i].set_xscale('log')
                    ax[2*i].legend(title='Mass ratio',fontsize=14,title_fontsize=16,loc='upper right')
                    ax[2*i].tick_params(axis="x", labelsize=16)
                    ax[2*i].tick_params(axis="y", labelsize=16)
                    ax[2*i].grid(True, which="both", ls="-",color='0.93')
                    ax[2*i].set_axisbelow(True)
                    
                    
                    sns.distplot(q_initial_nCE,kde=False,norm_hist=True,label='initial nCE n='+len(q_initial_nCE), 
                            hist_kws={"histtype": "step","lw": 2},ax=ax[2*i+1],color=col[0])
                    
                    sns.distplot(q_final_nCE,kde=False,norm_hist=True,label='final nCE n='+len(q_final_nCE), 
                            hist_kws={"histtype": "step","lw": 2},ax=ax[2*i+1],color=col[4])
                    

                    ax[2*i+1].set_title('Distribution of initial and final mass nCE ratio for fMT={}'.format(fMT[i]),fontsize=18)
                    ax[2*i+1].set_xlabel('q',fontsize=18)
                    ax[2*i+1].set_ylabel('Density',fontsize=18)
                    #ax[i].set_xscale('log')
                    ax[2*i+1].legend(title='Mass ratio',fontsize=14,title_fontsize=16,loc='upper right')
                    ax[2*i+1].tick_params(axis="x", labelsize=16)
                    ax[2*i+1].tick_params(axis="y", labelsize=16)
                    ax[2*i+1].grid(True, which="both", ls="-",color='0.93')
                    ax[2*i+1].set_axisbelow(True)

                if not os.path.isdir('./hist_fMT'):
                        os.mkdir('./hist_fMT')

                plt.savefig('hist_fMT/hist_mass_ratio_fMT.pdf', format='pdf')
                plt.close()
