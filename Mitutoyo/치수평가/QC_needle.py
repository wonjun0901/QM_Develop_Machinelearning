import pandas as pd
import numpy as np

data = pd.read_csv('./Mitutoyo/200309_200305사출/sample9.csv', encoding='euc-kr', header=0)

df = pd.DataFrame(data)

#print(df['Actual'])

Shortbase_length = df['Actual'][1]
Longbase_length = df['Actual'][2]

Support_Circle_Diameter = df['Actual'][4]

Support_thickness = df['Actual'][6]

BasetoSupport_thickiness = df['Actual'][7]

Pitch_left5to6 = df['Actual'][12]
Pitch_left4to6 = df['Actual'][11:13].sum()
Pitch_left3to6 = df['Actual'][10:13].sum()
Pitch_left2to6 = df['Actual'][9:13].sum()
Pitch_left1to6 = df['Actual'][8:13].sum()
Pitch_left6to7 = df['Actual'][13]
Pitch_left6to8 = df['Actual'][13:15].sum()
Pitch_left6to9 = df['Actual'][13:16].sum()
Pitch_left6to10 = df['Actual'][13:17].sum()
Pitch_left6to11 = df['Actual'][13:18].sum()

###############################

Pitch_bot6to7 = df['Actual'][22]
Pitch_bot6to8 = df['Actual'][21:23].sum()
Pitch_bot6to9 = df['Actual'][20:23].sum()
Pitch_bot6to10 = df['Actual'][19:23].sum()
Pitch_bot6to11 = df['Actual'][18:23].sum()
Pitch_bot5to6 = df['Actual'][23]
Pitch_bot4to6 = df['Actual'][23:25].sum()
Pitch_bot3to6 = df['Actual'][23:26].sum()
Pitch_bot2to6 = df['Actual'][23:27].sum()
Pitch_bot1to6 = df['Actual'][23:28].sum()

Pitch_left1to11 = df['Actual'][8:18].sum()
Pitch_bot1to11 = df['Actual'][18:28].sum()

#Dictionary = {['a+b' : Longbase_length]}
new_df = {}
new_df['a+b'] = Longbase_length
new_df['c+d'] = Shortbase_length
new_df['f'] = Support_Circle_Diameter
new_df['g'] = Support_thickness
new_df['h'] = BasetoSupport_thickiness
new_df['i'] = df['Deviation'][0]
new_df['k'] = Pitch_left5to6
new_df['l'] = Pitch_left4to6
new_df['m'] = Pitch_left3to6
new_df['n'] = Pitch_left2to6
new_df['o'] = Pitch_left1to6
new_df['p'] = Pitch_left6to7
new_df['q'] = Pitch_left6to8
new_df['r'] = Pitch_left6to9
new_df['s'] = Pitch_left6to10
new_df['t'] = Pitch_left6to11
new_df['u'] = Pitch_bot6to7
new_df['v'] = Pitch_bot6to8
new_df['w'] = Pitch_bot6to9
new_df['x'] = Pitch_bot6to10
new_df['y'] = Pitch_bot6to11
new_df['z'] = Pitch_bot5to6
new_df['1'] = Pitch_bot4to6
new_df['2'] = Pitch_bot3to6
new_df['3'] = Pitch_bot2to6
new_df['4'] = Pitch_bot1to6

new_df['5'] = Pitch_left1to11
new_df['6'] = Pitch_bot1to11

output = pd.Series(new_df)
print(output)
output.to_csv('./Mitutoyo/치수평가/sample9.csv', sep=',')