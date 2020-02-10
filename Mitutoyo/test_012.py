# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 2
import pandas as pd
import os
# 샘플 여러개를 보고 싶을 때 코드

fname = "QuadMedicine/Mitutoyo/data/NT7_InjMol_Monument800/20190724_data1.csv"
# print(fname[1])
# print(type(fname[1]))
data = pd.read_csv(fname, engine='python', index_col=False)
# dataname.append(pd.read_excel("data/NT7_InjMol_Monument800/20190724_data2.csv",
#                              engine='python', index_col=False))

print(os.getcwd())
print(os.listdir(os.getcwd()))
print(data.Nominal)
