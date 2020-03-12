import pandas as pd
import numpy as np

data = pd.read_csv('./Mitutoyo/치수평가/Cavity#2_9_1.csv', encoding='euc-kr', header=0)
data1 = pd.read_csv('./Mitutoyo/치수평가/Cavity#2_9_j.csv', encoding='euc-kr', header=0)

df = pd.DataFrame(data)
df1 = pd.DataFrame(data1)

Shortbase_length = df['Actual'][35]
Longbase_length = df['Actual'][36]

Support_Circle_Diameter = df['Actual'][28]
Base_Circle_Diameter = df['Actual'][33]

Support_thickness_1 = df['Actual'][7]
Support_thickness_2 = df['Actual'][8]
Support_thickness_3 = df['Actual'][9]
Support_thickness_4 = df['Actual'][10]
Support_thickness_Avg = df['Actual'][7:11].mean()

BasetoSupport_thickiness_1 = df['Actual'][3]
BasetoSupport_thickiness_2 = df['Actual'][4]
BasetoSupport_thickiness_3 = df['Actual'][5]
BasetoSupport_thickiness_4 = df['Actual'][6]
BasetoSupport_thickiness_Avg = df['Actual'][3:7].mean()

k = df['Actual'][13]
l = df['Actual'][14]
m = df['Actual'][15]
n = df['Actual'][16]
o = df['Actual'][17]
p = df['Actual'][18]
q = df['Actual'][19]
r = df['Actual'][20]
s = df['Actual'][21]
t = df['Actual'][22]

###############################

u = df['Actual'][23]
v = df['Actual'][24]
w = df['Actual'][25]
x = df['Actual'][26]
y = df['Actual'][27]
z = df['Actual'][28]
label1 = df['Actual'][29]
label2 = df['Actual'][30]
label3 = df['Actual'][31]
label4 = df['Actual'][32]

support_plane = df['Deviation'][0]
base_plane = df['Deviation'][2]

#Dictionary = {['a+b' : Longbase_length]}
new_df = {}
new_df['a+b'] = Longbase_length
new_df['c+d'] = Shortbase_length
new_df['e'] = Base_Circle_Diameter
new_df['f'] = Support_Circle_Diameter

new_df['g'] = Support_thickness_Avg
new_df['g_1'] = Support_thickness_1
new_df['g_2'] = Support_thickness_2
new_df['g_3'] = Support_thickness_3
new_df['g_4'] = Support_thickness_4

new_df['h'] = BasetoSupport_thickiness_Avg
new_df['h_1'] = BasetoSupport_thickiness_1
new_df['h_2'] = BasetoSupport_thickiness_2
new_df['h_3'] = BasetoSupport_thickiness_3
new_df['h_4'] = BasetoSupport_thickiness_4

new_df['j'] = df1['Actual'][1:5].mean()
new_df['j_1'] = df1['Actual'][1]
new_df['j_2'] = df1['Actual'][2]
new_df['j_3'] = df1['Actual'][3]
new_df['j_4'] = df1['Actual'][4]

new_df['k'] = k
new_df['l'] = l
new_df['m'] = m
new_df['n'] = n
new_df['o'] = o
new_df['p'] = p
new_df['q'] = q
new_df['r'] = r
new_df['s'] = s
new_df['t'] = t
new_df['u'] = u
new_df['v'] = v
new_df['w'] = w
new_df['x'] = x
new_df['y'] = y
new_df['z'] = z
new_df['1'] = label1
new_df['2'] = label2
new_df['3'] = label3
new_df['4'] = label4

new_df['base_flatness'] = base_plane
new_df['support_flatness'] = support_plane

output = pd.Series(new_df)
output.to_csv('./Mitutoyo/치수평가/output/output_Cavity#2_9.csv', sep=',')