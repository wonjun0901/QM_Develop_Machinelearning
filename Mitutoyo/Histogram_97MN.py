# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
import pandas as pd

def makinghistrogram(dataname, hist_bins=20, xaxis_range=(720,820), yaxis_top=25, Num_sample=1):
    
    if Num_sample == 1:
        # read excel data by pandas
        data = pd.read_csv(dataname, engine='python', index_col=False)
    
        # only save information about height of individual microneedle into variable
        height_tiptobase = data['Nominal']*1000
    
        # Plot the heatmap.
        fig, ax = plt.subplots(tight_layout=True)
        plt.ylim(top=yaxis_top)
        im = plt.hist(height_tiptobase, bins=hist_bins, range=xaxis_range )
    
        
        return im
    
    else : 
        makingnewlist = []
#        sample_all_heatmap = np.zeros((1,97))
        for i in range(Num_sample) :
            makingnewlist.append(dataname[i]['Nominal']*1000)
            
        all_data = pd.concat(makingnewlist,ignore_index=True)
        
        fig, ax = plt.subplots(tight_layout=True)
        plt.ylim(top=yaxis_top)
        im = plt.hist(all_data, bins=hist_bins, range=xaxis_range )
        
        return im
    

# 한 개만 보고 싶을 때 코드

dataname = "data/NT7_InjMol_Monument800/20190724_data1.csv"
makinghistrogram(dataname, hist_bins=20, xaxis_range=(720,820),
                 yaxis_top=25)


## 샘플 여러개를 보고 싶을 때 코드

#fname = []
## data 중 가장 첫번째 번호
#sample_num1 = 1
## data 중 가장 마지막 번호
#sample_num2 = 2
#for i in range(sample_num1,sample_num2 + 1):
#    fname.append(f"data/NT7_InjMol_Monument800/20190724_data{i}.csv")
#    
#dataname = []
## 총 측정할 샘플 갯수
#Num_sample = 2
#
#for makingdata in range(Num_sample):
#    dataname.append(pd.read_csv(fname[makingdata], 
#                                engine='python', index_col=False))
#
#
#makinghistrogram(dataname, hist_bins=20, xaxis_range=(700,820),
#                 yaxis_top=100, Num_sample=Num_sample)



