import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as fm

import matplotlib as mpl

#arial_bold = fm.FontProperties(fname = './Library/FontsFree-Net-arial-bold.otf')
font_name = fm.FontProperties(fname='./library/FontsFree-Net-arial-bold.ttf').get_name()

data = pd.read_excel('./PTH_1/PDMS_merged.xlsx', sheet_name="Sheet2")
data_1 = pd.read_excel('./PTH_1/PDMS_merged.xlsx', sheet_name="Sheet2")
#data_2 = pd.read_excel('./PTH_1/PDMS_merged.xlsx', sheet_name="Sheet3")
#data_3 = pd.read_excel('./PTH_1/PDMS_merged.xlsx', sheet_name="Sheet4")
#data_4 = pd.read_excel('./PTH_1/PDMS_merged.xlsx', sheet_name="Sheet5")

print(data)

plt.plot(data["s1"], data["p1"])
plt.scatter(data["s1"], data["p1"], s=50, marker="o")

plt.plot(data["s2"], data["p2"])
plt.scatter(data["s2"], data["p2"], s=50, marker="o")

plt.plot(data["s3"], data["p3"])
plt.scatter(data["s3"], data["p3"], s=50, marker="o")

plt.plot(data["s4"], data["p4"])
plt.scatter(data["s4"], data["p4"], s=50, marker="o")

plt.plot(data["s5"], data["p5"])
plt.scatter(data["s5"], data["p5"], s=50, marker="o")

plt.plot(data["s6"], data["p6"])
plt.scatter(data["s6"], data["p6"], s=50, marker="o")

#plt.plot(data["s7"], data["p7"])
#plt.scatter(data["s7"], data["p7"], s=50, marker="o")
##
#plt.plot(data1["Time"][20:30], data1["pk"][20:30], label="KJY")
#plt.scatter(data1["Time"][20:30], data1["pk"][20:30],s=50, marker="^")
##
#plt.scatter(data1["Time"][30:], data1["pk"][30:],s=50, marker="s")
#plt.plot(data1["Time"][30:], data1["pk"][30:], label="SGH")
##
#plt.scatter(data1["Time"][10:20], data1["pk"][10:20], s=50, marker="^")
#plt.plot(data1["Time"][10:20], data1["pk"][10:20], label="CSO")



#plt.errorbar(data_4["Time"], data_4["pk"], yerr=data_4["std"], label="Average", \
#            marker = "o", markersize = 3, elinewidth=2.5, capsize=7.5, capthick=1.5, linewidth=3.5)
#plt.scatter(data=data_4, x="Time", y="pk", marker="o", s=55)

#xaxis = [0, 15, 30, 60, 120, 180, 240]
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.tick_params(axis='x', labelsize=16)

#plt.ylim(-1, 1100)
#plt.yticks([0, 200,400,600,800,1000,10500])
plt.title('30 celsius, PDMS mold after autoclave', fontsize = 25)
#plt.legend(fontsize=25)
plt.xlabel("Compression Strain(mm)",fontsize=25)
plt.ylabel("Pressre(kPa)", fontsize=25)
plt.show()