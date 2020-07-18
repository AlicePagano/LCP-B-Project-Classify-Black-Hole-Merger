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
import os
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import bokeh.palettes as palette # For palette of colors
from scipy.optimize import curve_fit

class fit_hist:

    def fit_noerr(self,x,y):
        
        model, cov = np.polyfit(x, y, deg=1, cov=True)

        intercept, intercept_err = model[1], np.sqrt(cov[1][1])
        slope, slope_err = model[0], np.sqrt(cov[0][0])
        pearson = sp.stats.pearsonr(x, y)[0]

        param = [intercept, intercept_err,slope, slope_err,pearson,chisq_dof]
        
        return(param)

    def fit(self,x,y,sigmay):
        
        def line(x, a, b):
            return a * x + b

        popt, pcov = curve_fit(line, x, y, sigma=sigmay)

        intercept, intercept_err = popt[1], pcov[1,1]**0.5
        slope, slope_err = popt[0], pcov[0,0]**0.5
        pearson = sp.stats.pearsonr(x, y)[0]
        chisq_dof = np.sum( np.power( (y -(slope*x+intercept))/sigmay , 2) )/(len(x)-3)

        param = [intercept, intercept_err,slope, slope_err,pearson,chisq_dof]
        
        return(param)


    def t_merg(self,df,fMT,metal): #(dataframe,fMT,metal)

        fig = plt.figure(figsize=(30,30))
        ax = [fig.add_subplot(3,3,s) for s in range(1,len(fMT)+1)] 
        fig.suptitle('Fit of distribution of time tmerg',fontsize=30)
        col = sns.color_palette(('#440154', '#22A784')) 
        
        for i in range(0,len(fMT)):
            for j in range(0,len(metal)):

                edge = 50*np.logspace(np.log10(df[i][j]['tmerg[11]'].min()), np.log10(df[i][j]['tmerg[11]'].max()), 30) 
                
                # Compute non-normalized histogram
                n, bins = np.histogram(df[i][j]['tmerg[11]'],bins=edge,density=False) # histvals, binedges
                
                # Normalize histogram            
                yerrs = np.sqrt(n)
                binwidth = (bins[1:]-bins[:-1])
                constant = float(sum(n))*binwidth                
                n = n/constant
                yerrs = yerrs/constant

                x = (bins[1:]+bins[:-1])/2 # bincenters
                y = n # normalized counts

                # Remove zero count bins
                xy = pd.DataFrame({'x':x, 'y':y, 'yerrs':yerrs})
                xy = xy[ xy.y != 0 ] 
                
                # Take the log10 of x and y
                x_reg = np.log10(np.array(xy.x))
                y_reg = np.log10(np.array(xy.y))
                
                # Plot with error bar
                yerrs = np.divide(0.434*xy.yerrs,xy.y) # Error bar for logarithmic binning
                ax[i].errorbar(x_reg,y_reg,yerr=yerrs,fmt='.',color=col[j],label=metal[j])
                
                # Linear regression
                x_reg = x_reg[2:-1]
                y_reg = y_reg[2:-1]
                fit_param = self.fit(x_reg,y_reg,yerrs[2:-1])

                l = np.linspace(x_reg.min(),x_reg.max(),len(x_reg))
                ax[i].plot(l,fit_param[0]+fit_param[2]*l,color=col[j],
                           label='y ={0:01.2f}x{1:01.2f} \n $\chi^2$ ={2:01.4f} \n $r$ ={3:01.3f}'.format(fit_param[2],fit_param[0],fit_param[5],fit_param[4]))
                
            ax[i].set_title('Distribution of time tmerg for fMT={}'.format(fMT[i]),fontsize=18)
            ax[i].set_xlabel('$log_{10}$ (tmerg) [Myr]',fontsize=18)
            ax[i].set_ylabel('Density',fontsize=18)
            ax[i].legend(title='Metallicity',fontsize=14,title_fontsize=16)
            ax[i].tick_params(axis="x", labelsize=16)
            ax[i].tick_params(axis="y", labelsize=16)
            ax[i].grid(True, which="both", ls="-",color='0.93')
            ax[i].set_axisbelow(True)           
            
        if not os.path.isdir('./fit_hist'):
            os.mkdir('./fit_hist')

        plt.savefig('fit_hist/fit_hist_time_tmerg.pdf', format='pdf')
        plt.close()
