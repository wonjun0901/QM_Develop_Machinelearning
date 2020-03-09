import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


df = pd.read_excel(
    './test/data_flowrate1.xlsx', header=0, index=None)
# print(df)

time = df['Time']
flowrate = df['flowrate']

rolling_mean = flowrate.rolling(window=20, center=True).mean()

rolling_mean = pd.DataFrame(rolling_mean)

slope = rolling_mean.diff()

pd.set_option('display.max_rows', 150)

print(df)
#slope = pd.DataFrame(slope)
# slope.fillna(value=0)
# print(slope)

#print(slope.loc[slope['flowrate'] > 10])
#print(df.loc[df['flowrate'].rolling(window=20, center=True).mean() > 50 ])
# print(rolling_mean)
#flowrate = flowrate - rolling_mean
#flows = np.array(flowrate)

# print(np.mean(flows))


plt.plot(time, flowrate)
plt.plot(time, rolling_mean, c='r')
plt.xlabel('time(s)')
plt.ylabel('flowrate(ul/min)')
plt.xticks(np.arange(0, 131, step=10))
plt.yticks(np.arange(-50, 201, step=50))
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()
