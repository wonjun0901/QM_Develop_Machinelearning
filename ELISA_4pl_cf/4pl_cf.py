import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

df = pd.read_excel('./ELISA_4pl_cf/Hb_elisa_test.xlsx', sheet_name=2)
#print(df)

# define 4pl logistic
def logistic4(x, A, B, C, D):
    """4PL logoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D

#df = df.drop_na()

xdata = df['Conc']
xdata = xdata.dropna()
#print(xdata)
ydata = df['Value']
ydata = ydata.dropna()
#print(ydata)
xdata1 = df['Conc1']
xdata1 = xdata1.dropna()
#print(xdata1)
ydata1 = df['Value1']
ydata1 = ydata1.dropna()
#print(ydata1)
#print(ydata1)
popt, pcov = curve_fit(logistic4, xdata, ydata)
popt1, pcov1 = curve_fit(logistic4, xdata1, ydata1)

residuals = ydata - logistic4(xdata, *popt)
ss_res = np.sum(residuals**2)

ss_tot = np.sum((ydata-np.mean(ydata))**2)
r_squared = 1 - (ss_res/ss_tot)

print(r_squared)
#print(popt)
#print(pcov)
x_fit = np.linspace(0.01,1000,1000)
y_fit = logistic4(x_fit, *popt)
#y_fit

x_fit1 = np.linspace(0.01,1000,1000)
y_fit1 = logistic4(x_fit1, *popt1)

plt.plot(xdata, ydata, 'o', label = 'standard')
plt.plot(x_fit, y_fit, label = '4pl_curve')

plt.plot(xdata1, ydata1, 'o', label = 'data')
plt.plot(x_fit1, y_fit1, label = 'fit')


#plt.show()
# 역수를 구해서 OD값에서 conc 값 알아내기
def solvex(y, A, B, C, D):
    """rearranged 4PL logoistic equation."""
    return C*(((A-D)/(y-D)-1.0)**(1.0/B))

a = solvex(ydata1, *popt)
print(a)
plt.plot(a, ydata1, 'o', label='sample')
plt.legend()
plt.xscale('log')
plt.show()