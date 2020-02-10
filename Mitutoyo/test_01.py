# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
import pandas as pd

# 샘플 여러개를 보고 싶을 때 코드

fname = []
for i in range(2):
    
    fname.append(f"data/NT7_InjMol_Monument800/20190724_data{i}.csv")
    
#print(fname[1])
#print(type(fname[1]))
data = pd.read_csv(fname[1], engine='python', index_col=False)
#dataname.append(pd.read_excel("data/NT7_InjMol_Monument800/20190724_data2.csv", 
#                              engine='python', index_col=False))