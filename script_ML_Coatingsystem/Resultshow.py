import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io


df = pd.read_excel('./script_ML_Coatingsystem/Resultcollection.xlsx', sheet_name= 'Sheet1', header=0, index=None)
x = np.linspace(-1500, 1500, 1500)
y = np.linspace(-1500, 1500, 1500)
#plt.scatter(df['Real'], df['LR'], c='blue', alpha=0.1)
#plt.scatter(df['Real'], df['MLP'],c='red', alpha=0.1)
#plt.scatter(df['Real'], df['SVR'],c='black')
#plt.scatter(df['Real'], df['SVR'],c='green')
#plt.scatter(df['Real'], df['RF'],c='orange')
#plt.scatter(df['Real'], df['GB'],c='orange')
plt.scatter(df['Real'], df['voting'], alpha=0.1)
plt.plot(x,y, color='red', linewidth=0.3)
plt.xticks(np.arange(-1500, 1501, step=250))
plt.yticks(np.arange(-1500, 1501, step=250))
plt.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()