import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

import matplotlib as mpl

#arial_bold = fm.FontProperties(fname = './Library/FontsFree-Net-arial-bold.otf')
font_name = fm.FontProperties(fname='./library/FontsFree-Net-arial-bold.ttf').get_name()

data = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Sheet1")
data_1 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Sheet2")
data_2 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Sheet3")
data_3 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Sheet4")
data_4 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Sheet5")
data_5 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Thigh")
data_6 = pd.read_excel('./PTH_1/pk_result(merged).xlsx', sheet_name="Elbow")
#########################################################################

data1 = data.query('Name == ["BSK(SC)", "CSO(SC)","KJY(SC)","SGH(SC)"]')
data2 = data.query('Name == ["BSK(Shoulder)", "CSO(Shoulder)","KJY(Shoulder)","SGH(Shoulder)"]')
data3 = data.query('Name == ["BSK(Wrist)", "CSO(Wrist)","KJY(Wrist)","SGH(Wrist)"]')
data4 = data.query('Name == ["BSK(손등)", "CSO(손등)"]')
data5 = data.query('Name == ["BSK(Thigh)", "CSO(Thigh)","KJY(Thigh)","SGH(Thigh)"]')
data6 = data.query('Name == ["BSK(Elbow)", "CSO(Elbow)","KJY(Elbow)","SGH(Elbow)"]')

########################################################################

data7 = data.query('Name == ["BSK(SC)", "BSK(Shoulder)","BSK(Wrist)","BSK(손등)","BSK(Thigh)","BSK(Elbow)"]')
data8 = data.query('Name == ["CSO(SC)", "CSO(Shoulder)","CSO(Wrist)","CSO(손등)","CSO(Thigh)","CSO(Elbow)"]')
data9 = data.query('Name == ["KJY(SC)", "KJY(Shoulder)","KJY(Wrist)", "KJY(Thigh)","KJY(Elbow)"]')
data10 = data.query('Name == ["SGH(SC)", "SGH(Shoulder)","SGH(Wrist)", "SGH(Thigh)","SGH(Elbow)"]')


#print(data5)
#plt.plot(data10["Time"][:10], data10["pk"][:10], label="SC")
#plt.scatter(data10["Time"][:10], data10["pk"][:10], s=50, marker="o")
#
#plt.plot(data10["Time"][10:20], data10["pk"][10:20], label="Shoulder")
#plt.scatter(data10["Time"][10:20], data10["pk"][10:20],s=50, marker="^")
#
#plt.scatter(data10["Time"][20:30], data10["pk"][20:30],s=50, marker="s")
#plt.plot(data10["Time"][20:30], data10["pk"][20:30], label="Wrist")
#
#plt.scatter(data10["Time"][30:40], data10["pk"][30:40], s=50, marker="^")
#plt.plot(data10["Time"][30:40], data10["pk"][30:40], label="Thigh")
#
#plt.scatter(data10["Time"][40:50], data10["pk"][40:50], s=50, marker="v")
#plt.plot(data10["Time"][40:50], data10["pk"][40:50], label="Elbow")
#
#plt.plot(data10["Time"][50:60], data10["pk"][50:60], label="Elbow")
#plt.scatter(data10["Time"][50:60], data10["pk"][50:60], s=50, marker="D")



plt.errorbar(data_6["Time"], data_6["pk"], yerr=data_6["std"], label="Average", \
            marker = "o", markersize = 3, elinewidth=2.5, capsize=7.5, capthick=1.5, linewidth=3.5)
plt.scatter(data=data_6, x="Time", y="pk", marker="o", s=55)

xaxis = [0, 5, 10, 15, 30, 45, 60, 120, 180, 240]
plt.xticks(xaxis,fontsize=15)
plt.yticks(fontsize=15)
#plt.tick_params(axis='x', labelsize=16)

plt.ylim(-1, 1100)
#plt.yticks([0, 200,400,600,800,1000,10500])
plt.title('MAP3(Backofhand) Averaged PK profile', fontsize = 25)
#plt.legend(fontsize=35, loc=1)
plt.xlabel("Time(min)",fontsize=25)
plt.ylabel("Concentration(pg/ml)", fontsize=25)
plt.show()