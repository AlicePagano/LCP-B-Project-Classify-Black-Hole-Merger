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



class read_parallel:


        # Read a single chunk of data with dask and joblib
        def read_chunks(self,i, j, m):
                cols = pd.read_csv('/data1/collaborator/modB/simulations_fMT01/A5/0.002/chunk0/evol_mergers.out',sep="\s+",nrows=1,header=None).iloc[0].values[1:]
                path = '/data1/collaborator/modB/simulations_fMT'+i+'/A5/'+j+'/chunk'+m+'/evol_mergers.out'
                columns_to_read = [cols[0],cols[1],cols[2],cols[4],cols[16],cols[18],cols[33]]   
                dataframe = pd.read_csv(path,sep="\s+",skiprows=1,header=None,usecols=columns_to_read,names=cols)
                return dataframe



        # Filter a single chunk of data for black holes
        def filter_chunks_BH(self,data):           
                data_BH = data[ data['k1[2]']==14 ]
                data_BH = data_BH[ data_BH['k2[16]']==14 ]
                index_BH = list(data_BH['ID[0]'].unique()) 
                data_BH = data[ data['ID[0]'].isin(index_BH) ]       
                return data_BH



        # Filter a single chunk of data for common envelope
        def filter_chunks_CE(self,data):      
                data_CE = data[ data['label[33]']=='COMENV' ]
                index_CE = list(data_CE['ID[0]'].unique()) 
                data_CE = data[ data['ID[0]'].isin(index_CE) ]     
                return data_CE



        # Filter a single chunk of data for no common envelope
        def filter_chunks_nCE(self,data):
                data_CE = data[ data['label[33]']=='COMENV' ]
                index_CE = list(data_CE['ID[0]'].unique()) 
                data_nCE = data[ ~data['ID[0]'].isin(index_CE) ]
                return data_nCE
        


        # Filter a single chunk of data for a initial mass
        def filter_chunks_initial(self,data):
                data = data[ data['label[33]']=='INITIAL' ]
                return data
        


        # Filter a single chunk of data for final mass
        def filter_chunks_final(self, data):
                data = data[ data['k1[2]']==14 ]
                data = data[ data['k2[16]']==14 ]
                data = data.groupby('ID[0]').first().reset_index(drop=False)           
                return data



        # Filter a single chunk of data for delay times
        def filter_chunks_dtimes(self, data):
                data = data[ data['label[33]']=='COELESCE' ]
                return data
            
            

        # Organize data by reordering them
        def organize_data(self,data_CE,data_nCE,fMT,metal,chunk):
                df_fMT_CE = []
                df_fMT_nCE = []
                for i in fMT:
                        df_metal_CE = []
                        df_metal_nCE = []
                        for j in metal:
                                df_chunk_CE = []
                                df_chunk_nCE = []
                                for m in chunk:
                                        # Read chunk with/without CE
                                        df_CE = data_CE[fMT.index(i)*len(metal)*len(chunk)+metal.index(j)*len(chunk)+chunk.index(m)]
                                        df_nCE = data_nCE[fMT.index(i)*len(metal)*len(chunk)+metal.index(j)*len(chunk)+chunk.index(m)]

                                        df_chunk_CE.append(df_CE)
                                        df_chunk_nCE.append(df_nCE)

                                df_metal_CE.append(pd.concat(df_chunk_CE,ignore_index=True)) 
                                df_metal_nCE.append(pd.concat(df_chunk_nCE,ignore_index=True))
                        
                         # Concatenate all the metallicities
                        df_metal_u_CE = pd.concat(df_metal_CE[:5],ignore_index=True)
                        df_metal_CE.append(df_metal_u_CE)
                        df_metal_u_nCE = pd.concat(df_metal_nCE[:5],ignore_index=True)
                        df_metal_nCE.append(df_metal_u_nCE)
                        
                        df_fMT_CE.append(df_metal_CE)
                        df_fMT_nCE.append(df_metal_nCE)

                return df_fMT_CE, df_fMT_nCE