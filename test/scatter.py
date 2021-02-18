import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import 
from sklearn.preprocessing import PolynomialFeatures



#data = pd.read_excel('./test/IRMN_weight_measurement_bycycle.xlsx', header=0, sheet_name=0, dtype={'Press Count':np.float64, 'Weight(ug)':np.float64})
data = pd.read_excel('./test/IRMN_weight_measurement.xlsx', header=0, sheet_name=0, dtype={'Press Count':np.float64, 'Weight(ug)':np.float64, 'Measurement':str})
#data2 = pd.read_excel('./test/IRMN_weight_measurement_bycycle.xlsx', header=0, sheet_name=2, dtype={'Press Count':np.float64, 'Weight(ug)':np.float64, "Measurement":str})


df = pd.DataFrame(data)

#fig, ax1 = plt.subplots(sharex=True, sharey=True)
#print(df[3:5,:])

#poly_features = PolynomialFeatures(degree=2, include_bias=True)
#x_poly = poly_features.fit_transform(df["Weight(ug)"])

#print(df)

#ax = sns.barplot(data=data, x="Day", y="Weight", ci="sd")
#ax = plt.subplot()
ax = sns.lineplot(x="Press Count", y="Weight(ug)", hue="Measurement",hue_order=["Theoretical value", "Exp1", "Exp2"], data=data,palette="Blues", ci="sd")

ax = sns.catplot(data=data, x="Press Count", y="Weight(ug)", hue="Measurement", hue_order=["Theoretical value", "Exp1", "Exp2"], kind="bar", ci="sd", legend_out=False, palette="Blues" )
plt.show()
#sns.regplot(x="Press Count", y="Weight(ug)", data=data1, order=2, n_boot=1000)
#sns.regplot(x="Press Count", y="Weight(ug)", data=data2, order=2, n_boot=1000)

#plt.xticks(fontsize=20)
#plt.yticks(fontsize=25)
#plt.tick_params(axis='x', labelsize=16)
#plt.xlabel("Day",fontsize=35)
#plt.ylabel("Weight(ug)", fontsize=35)
#plt.tight_layout()