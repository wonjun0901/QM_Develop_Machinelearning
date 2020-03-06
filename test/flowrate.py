import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


df = pd.read_excel('C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/test/data_flowrate2.xlsx', header=0, index=None)
print(df)
time=df['Time']

flowrate = df['flowrate']
rolling_mean = flowrate.rolling(window=20).mean()

#flowrate = flowrate - rolling_mean
flows = np.array(flowrate)

print(np.mean(flows))


plt.plot(time, flowrate)
plt.plot(time, rolling_mean, c='r')
plt.xlabel('time(s)')
plt.ylabel('flowrate(ul/min)')
plt.xticks(np.arange(0, 110, step=10))
plt.yticks(np.arange(0, 151, step=50))
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()