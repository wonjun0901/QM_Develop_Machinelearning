import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_excel('./test/dispenser_weight_measurement.xlsx', header=0)

df = pd.DataFrame(data)

df = np.array(df)

mater1= df[:, 0]
pvp = df[:7,1]
tre = df[:,2]
suc = df[:,3]

mean_pvp = np.mean(pvp)
mean_tre = np.mean(tre)
mean_suc = np.mean(suc)

std_pvp = np.std(pvp) 
std_tre = np.std(tre)
std_pvp = np.std(suc)

materials = ['PVP+PTH', 'Trehalose+PTH', 'Sucrose+PTH']
x_pos = np.arange(len(materials))
CTEs = [mean_pvp, mean_tre, mean_suc]
error = [std_pvp, std_tre, std_pvp]

fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
sns.swarmplot(data=[pvp, tre, suc])
#ax.plot(range(7), pvp)
ax.set_ylabel('amount of PTH(ug)')
ax.set_xticks(x_pos)
ax.set_xticklabels(materials)
ax.set_title('PTH amount in MAP')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()