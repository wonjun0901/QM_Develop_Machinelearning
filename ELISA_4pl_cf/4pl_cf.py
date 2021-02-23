import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

df = pd.read_excel('./ELISA_4pl_cf/Hb_elisa_test.xlsx', sheet_name=2)
# print(df)

# define 4pl logistic


def logistic4(x, A, B, C, D):
    """4PL logoistic equation."""
    return ((A-D)/(1.0+((x/C)**B))) + D

# df = df.drop_na()


### standard ####
std_dup = 2
dil_num = 5

std_xdata = df.iloc[:dil_num, 0:std_dup]
std_xdata = np.reshape(np.array(std_xdata), -1)
std_xdata = pd.Series(std_xdata)

std_ydata = df.iloc[:dil_num, 10:10+std_dup]
std_ydata = np.reshape(np.array(std_ydata), -1)
std_ydata = pd.Series(std_ydata)

popt, pcov = curve_fit(logistic4, std_xdata, std_ydata,
                       bounds=(0, [10, 10, 50, 3.4]))

print(popt)

residuals = std_ydata - logistic4(std_xdata, *popt)
ss_res = np.sum(residuals**2)

ss_tot = np.sum((std_ydata-np.mean(std_ydata))**2)
r_squared = 1 - (ss_res/ss_tot)

print(r_squared)

x_fit = np.linspace(0.1, 100, 1000)
y_fit = logistic4(x_fit, *popt)

plt.plot(std_xdata, std_ydata, 'o', label='standard')
plt.plot(x_fit, y_fit, label='4pl_curve_fit')
plt.legend()
plt.xscale('log')
plt.show()

####################################################################
# 샘플 갯수
spl_num = 3

spl_dup = [0, 2, 2, 2]

spl_sum = np.cumsum(spl_dup, axis=0)

proc_spl_ydata = []

for i in range(0, spl_num):

    proc_spl_ydata.append(
        df.iloc[:dil_num, 12+spl_sum[i]:12+spl_sum[i]+spl_dup[i+1]])
    proc_spl_ydata[i] = np.reshape(np.array(proc_spl_ydata[i]), -1)
    proc_spl_ydata[i] = proc_spl_ydata[i][np.logical_not(
        np.isnan(proc_spl_ydata[i]))]

proc_spl_xdata = []

for i in range(0, spl_num):

    proc_spl_xdata.append(
        df.iloc[:dil_num, 2+spl_sum[i]:2+spl_sum[i]+spl_dup[i+1]])
    proc_spl_xdata[i] = np.reshape(np.array(proc_spl_xdata[i]), -1)
    proc_spl_xdata[i] = proc_spl_xdata[i][np.logical_not(
        np.isnan(proc_spl_xdata[i]))]


# #print(ydata1)
# #print(ydata1)

# popt1, pcov1 = curve_fit(logistic4, xdata1, ydata1)
def solvex(y, A, B, C, D):
    """rearranged 4PL logoistic equation."""
    return C*(((A-D)/(y-D)-1.0)**(1.0/B))


regressed_x = []

for i in range(0, spl_num):
    regressed_x.append(solvex(proc_spl_ydata[i], *popt))

for i in range(0, spl_num):
    plt.plot(regressed_x[i], proc_spl_ydata[i], 'o', label=i+1)

plt.plot(x_fit, y_fit, label='4pl_curve_fit')
plt.legend()
plt.xscale('log')
plt.show()

y_fit1 = []
popt1 = []
pcov1 = []

popt1, pcov1 = curve_fit(
    logistic4, proc_spl_xdata[0], proc_spl_ydata[0], bounds=(0, [10, 10, 50, 3.4]))
popt2, pcov2 = curve_fit(
    logistic4, proc_spl_xdata[1], proc_spl_ydata[1], bounds=(0, [10, 10, 50, 3.4]))
popt3, pcov3 = curve_fit(
    logistic4, proc_spl_xdata[2], proc_spl_ydata[2], bounds=(0, [10, 10, 50, 3.4]))

y_fit1 = []
y_fit1.append(logistic4(x_fit, *popt1))
y_fit1.append(logistic4(x_fit, *popt2))
y_fit1.append(logistic4(x_fit, *popt3))

for i in range(0, spl_num):
    plt.plot(regressed_x[i], proc_spl_ydata[i], 'o', label=i+1)
    plt.plot(x_fit, y_fit1[i],  label=i+1)

plt.plot(x_fit, y_fit, label='4pl_curve_fit')

# print(popt)
print(popt1)
print(popt2)
print(popt3)

plt.legend()
plt.xscale('log')
plt.show()


#########################################
residuals1 = []

residuals1.append(proc_spl_ydata[0] - logistic4(proc_spl_xdata[0], *popt1))
residuals1.append(proc_spl_ydata[1] - logistic4(proc_spl_xdata[1], *popt2))
residuals1.append(proc_spl_ydata[2] - logistic4(proc_spl_xdata[2], *popt3))

ss_res1 = []

ss_res1.append(np.sum(residuals1[0]**2))
ss_res1.append(np.sum(residuals1[1]**2))
ss_res1.append(np.sum(residuals1[2]**2))

ss_tot1 = []
ss_tot1.append(np.sum((proc_spl_ydata[0]-np.mean(proc_spl_ydata[0]))**2))
ss_tot1.append(np.sum((proc_spl_ydata[1]-np.mean(proc_spl_ydata[1]))**2))
ss_tot1.append(np.sum((proc_spl_ydata[2]-np.mean(proc_spl_ydata[2]))**2))

r_squared1 = []
r_squared1.append(1 - (ss_res1[0]/ss_tot1[0]))
r_squared1.append(1 - (ss_res1[1]/ss_tot1[1]))
r_squared1.append(1 - (ss_res1[2]/ss_tot1[2]))

print(r_squared1)
