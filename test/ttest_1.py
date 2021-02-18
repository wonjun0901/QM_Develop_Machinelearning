import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


from scipy import stats

data = pd.read_excel('./test/pthsample1.xlsx', sheet_name="Sheet2")

# 2차 실험 - Bloomage
sample1 = data['Concentration(ug)'][6:31]
#print(sample1.mean())

# 2차 실험 - Bloomage-TS
sample2 = data['Concentration(ug)'][56:62]
#print(sample2.mean())

# 2차 실험 - 일동 
sample3 = data['Concentration(ug)'][31:56]
#print(sample3.mean())

# 2차 실험 - 일동-TS
sample4 = data['Concentration(ug)'][62:69]
#print(sample4.mean())

# 3차 실험 - 1회 채움
sample5 = data['Concentration(ug)'][69:94]

# 3차 실험 - 1회 채움-TS
sample6 = data['Concentration(ug)'][133:140]

# 3차 실험 - 2회 채움 
sample7 = data['Concentration(ug)'][94:119]
# 3차 실험 - 2회 채움-TS
sample8 = data['Concentration(ug)'][140:145]

# 4차 실험 - 2회 채움
sample9 = data['Concentration(ug)'][145:159]

# 4차 실험 - 2회채움-TS
sample10 = data['Concentration(ug)'][159:169]
# 4차 실험 - 2회채움-TS*
sample11 = data['Concentration(ug)'][169:175]

#result = stats.ttest_rel(sample1, sample2)

print(stats.ttest_ind(sample1, sample3))
print(stats.ttest_ind(sample2, sample4))

print(stats.f_oneway(sample1, sample3))
print(stats.f_oneway(sample2, sample4))

# print(stats.f_oneway(sample5, sample6))
# print(stats.f_oneway(sample7, sample8))
# 
# print(stats.f_oneway(sample9, sample10))
# print(stats.f_oneway(sample9, sample11))
print(stats.f_oneway(sample10, sample11))
## data = pd.read_excel('./test/200303_Bradford_ELISA.xlsx',
##                      header=0, sheet_name='ELISA')
## 
## sample1 = data['H:T=1:20'][0:5]
## print(sample1.mean())
## sample2 = data['H:T=1:20'][5:10]
## print(sample2.mean())
## sample3 = data['H:T=1:20'][0:5]
## print(sample3.mean())
## sample4 = data['H:T=1:20'][10:15]
## print(sample4.mean())
## #result = stats.ttest_rel(sample1, sample2)
## 
## print(stats.ttest_rel(sample1, sample2))
## print(stats.ttest_rel(sample3, sample4))
## 