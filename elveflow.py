import pandas as pd
import matplotlib.pyplot as plt
data1 = pd.read_excel('C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/coatingshape/data1.xlsx', sheet_name = 'Sheet2', index=None)

time = data1['Time [s]']
target_flowrate = data1['qm(Target) [탅/min]']
real_flowrate = data1['qm(Read)[탅/min]']



plt.plot(time, target_flowrate,'r--', label='target Flow rate')
plt.plot(time, real_flowrate, label='Real Flow rate')
plt.xlabel('time(s)')
plt.ylabel('flow rate(ul/min)')
plt.title('Eleveflow Flow by Pressure')
plt.grid(True)
plt.legend()
plt.show()