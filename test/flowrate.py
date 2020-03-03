import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


df = pd.read_excel(
    './test/data_flowrate.xlsx', header=0, index=None)
print(df)
time = df['Time']

flowrate = df['flowrate']-170
rolling_mean = flowrate.rolling(window=80, center=True).mean()

#flowrate = flowrate - rolling_mean
#flows = np.array(flowrate)

# print(np.mean(flows))


plt.plot(time, flowrate)
plt.plot(time, rolling_mean, c='r')
plt.xlabel('time(s)')
plt.ylabel('flowrate(ul/min)')
plt.xticks(np.arange(550, 680, step=10))
plt.yticks(np.arange(-50, 300, step=50))
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()
