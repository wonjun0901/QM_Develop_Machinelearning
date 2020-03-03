import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


df = pd.read_excel('C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/test/data_flowrate.xlsx', header=0, index=None)
print(df)
time=df['Time'][550:]

flowrate = df['flowrate'][550:]
flows = np.array(flowrate)

print(np.mean(flows))


plt.plot(time, flowrate)
plt.xlabel('time(s)')
plt.ylabel('flowrate(ul/min)')
plt.grid(c='r')
plt.show()