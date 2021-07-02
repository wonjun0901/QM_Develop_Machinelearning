import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./Robot_validation/Standard.xlsx')

print(df.describe)

a0 = df['Force0']
a1 = df['Force1']
a2 = df['Force2']
a3 = df['Force3']
a4 = df['Force4']
#a5 = df['Force5']

time0 = df['Time0']
time1 = df['Time1']
time2 = df['Time2']
time3 = df['Time3']
time4 = df['Time4']
#time5 = df['Time5']

b0 = df['CompressionStrain0']
b1 = df['CompressionStrain1']
b2 = df['CompressionStrain2']
b3 = df['CompressionStrain3']
b4 = df['CompressionStrain4']	
#b5 = df['CompressionStrain5']	

c0 = df['CompressionStrained0']
c1 = df['CompressionStrained1']
c2 = df['CompressionStrained2']
c3 = df['CompressionStrained3']
c4 = df['CompressionStrained4']
#c5 = df['CompressionStrained5']

plt.plot(b0, a0, label='sample0')
plt.plot(b1, a1, label='sample1')
plt.plot(b2, a2, label='sample2')
plt.plot(b3, a3, label='sample3')
plt.plot(b4, a4, label='sample4')
#plt.plot(b5, a5, label='sample5')

plt.xticks(fontsize =30)
plt.yticks(fontsize =30)

plt.legend()
plt.show()