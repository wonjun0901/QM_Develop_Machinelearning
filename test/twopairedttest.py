import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import stats


data = pd.read_excel('./test/200303_Bradford_ELISA.xlsx',
                     header=0, index=None, sheet_name='Bradford')

sample1 = data['H:T=1:20'][0:5]
print(sample1.mean())
sample2 = data['H:T=1:20'][5:10]
print(sample2.mean())

sample3 = data['H:T=1:20'][0:5]
print(sample3.mean())

sample4 = data['H:T=1:20'][10:15]
print(sample4.mean())

#result = stats.ttest_rel(sample1, sample2)

print(stats.ttest_rel(sample1, sample2))
print(stats.ttest_rel(sample3, sample4))

data = pd.read_excel('./test/200303_Bradford_ELISA.xlsx',
                     header=0, index=None, sheet_name='ELISA')

sample1 = data['H:T=1:20'][0:5]
print(sample1.mean())
sample2 = data['H:T=1:20'][5:10]
print(sample2.mean())
sample3 = data['H:T=1:20'][0:5]
print(sample3.mean())
sample4 = data['H:T=1:20'][10:15]
print(sample4.mean())
#result = stats.ttest_rel(sample1, sample2)

print(stats.ttest_rel(sample1, sample2))
print(stats.ttest_rel(sample3, sample4))
