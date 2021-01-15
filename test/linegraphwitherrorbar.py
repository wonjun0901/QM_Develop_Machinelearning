import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data = pd.read_excel('./test/pthsample1.xlsx', sheet_name="Sheet1")


data1 = pd.read_excel('./test/pthsample1.xlsx', sheet_name="Sheet2")


#df = pd.DataFrame(data)

#df1 = sns.load_dataset("exercise")

#print(df1)

# cal1= df[:4, 0]
# cal2= df[:3, 1]
# cal3= df[:3, 2]
# cal4= df[:3, 3]
# cal5= df[:3, 4]
# cal6= df[:3, 5]
# 
# 
# BlankMN= df[:5, 6]
# bloomha = df[:,7]
# ildongha = df[:,8]
# bloomha_gamma = df[:6,9]
# ildong_gamma = df[:7, 10]

sns.regplot(data=data, x="Expected Concentration(ug)", y="Concentration(ug)", fit_reg=True)
#sns.barplot(data=data1, x="Type", y="Concentration(ug)", facecolor=(1, 1, 1, 0),
#                 errcolor=".2", edgecolor=".2", capsize=0.15)
#sns.swarmplot(x="Type", y="Concentration(ug)", data=data1, size=11, dodge=True)
plt.xticks(fontsize=20)
plt.yticks(fontsize=25)
#plt.tick_params(axis='x', labelsize=16)
plt.xlabel("Theoretical Concentration(ug)",fontsize=25)
plt.ylabel("Concentration(ug)", fontsize=25)
plt.show()