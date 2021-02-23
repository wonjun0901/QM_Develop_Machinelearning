import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./Robot_validation/robot3_DO0.csv')
#print(df)
#[0,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,89,90][:]
#df = df[:, :]
pd.options.display.max_rows = 150
pd.options.display.max_columns = 27
#pd.options.display.max_rows = 600

column_name = []
# time line
column_name.append('timestamp')

# target TCP pose
column_name.append('target_TCP_pose_0')
column_name.append('target_TCP_pose_1')
column_name.append('target_TCP_pose_2')
column_name.append('target_TCP_pose_3')
column_name.append('target_TCP_pose_4')
column_name.append('target_TCP_pose_5')

# actual TCP pose
column_name.append('actual_TCP_pose_0')
column_name.append('actual_TCP_pose_1')
column_name.append('actual_TCP_pose_2')
column_name.append('actual_TCP_pose_3')
column_name.append('actual_TCP_pose_4')
column_name.append('actual_TCP_pose_5')

# target TCP speed
column_name.append('target_TCP_speed_0')
column_name.append('target_TCP_speed_1')
column_name.append('target_TCP_speed_2')
column_name.append('target_TCP_speed_3')
column_name.append('target_TCP_speed_4')
column_name.append('target_TCP_speed_5')

# actual_TCP_speed
column_name.append('actual_TCP_speed_0')
column_name.append('actual_TCP_speed_1')
column_name.append('actual_TCP_speed_2')
column_name.append('actual_TCP_speed_3')
column_name.append('actual_TCP_speed_4')
column_name.append('actual_TCP_speed_5')

# digital input bits
column_name.append('actual_digital_input_bits')

# digital output bits
column_name.append('actual_digital_output_bits')

 
p1_df = df.loc[:, column_name]
#print(p1_df)

# data processing which actual digital output bit equal to 1
p2_df = p1_df[df['actual_digital_output_bits']==1]

# 데이터 프로세싱 - 1에서 0되는 부분 찾아야됨



# shift to 1 below to all data
p2_df_shift = p2_df.shift(periods=1, fill_value=0)

# substract
p3_df = p2_df - p2_df_shift
#print(p3_df)

df_1st = p1_df.shift(periods=1, fill_value=0)
df_2nd = p1_df - df_1st

#print(p3_df)
df_3rd = df_2nd[df_2nd['actual_digital_output_bits']==-1]

#print(p2_df)
#print(df_3rd)

# data processing - 차이가 0.02 이상인 부근의 데이터 프레임만 찾기
p4_df = p3_df[p3_df['timestamp']>0.02]

# data processing
index_df_4th = df_3rd.index.tolist()
print(index_df_4th)


# 차이가 0.02이상의 부근인 데이터에 대한 인덱스 찾기
index_p4_df = p4_df.index.tolist()
#print(index_p4_df)

#해당 인덱스에 대한 데이터 프레임, 300 equals to 30s
a1 = p1_df.iloc[index_p4_df[0]:index_p4_df[0]+300]
a2 = p1_df.iloc[index_p4_df[1]:index_p4_df[1]+300]
#print(a1)

a3 = df.iloc[index_df_4th[0]:index_df_4th[0]+199]
a4 = df.iloc[index_df_4th[0]:index_df_4th[0]+199]

#########################################################

# 해당 인덱스부터 신호 기간 동안만 데이터를 추출하기
a = {}

for i in range(0, len(index_p4_df)):
    a[i]=p1_df.iloc[index_p4_df[i]:index_p4_df[i]+300]

dis_x_cord = {}
dis_y_cord = {}
dis_z_cord = {}

for i in range(0, len(index_p4_df)):
    dis_x_cord[i]= np.array(a[i]['target_TCP_pose_0']-a[i]['actual_TCP_pose_0'])*1000000
    dis_y_cord[i]= np.array(a[i]['target_TCP_pose_1']-a[i]['actual_TCP_pose_1'])*1000000
    dis_z_cord[i]= np.array(a[i]['target_TCP_pose_2']-a[i]['actual_TCP_pose_2'])*1000000

######################################################################

bc = {}

for i in range(0, len(index_df_4th)):
    bc[i]=df.iloc[index_df_4th[i]:index_df_4th[i]+199]

spd_cord = {}

for i in range(0, len(index_df_4th)):
    spd_cord[i]= np.square(np.array(bc[i]['actual_TCP_speed_1']))+\
    np.square(np.array(bc[i]['actual_TCP_speed_0']))+np.square(np.array(bc[i]['actual_TCP_speed_2']))
    spd_cord[i] = np.sqrt(spd_cord[i]*1000000)

print(spd_cord)

spd_target_cord={}

for i in range(0, len(index_df_4th)):
    spd_target_cord[i]= np.square(np.array(bc[i]['target_TCP_speed_1']))+\
    np.square(np.array(bc[i]['target_TCP_speed_0']))+np.square(np.array(bc[i]['target_TCP_speed_2']))
    spd_target_cord[i] = np.sqrt(spd_target_cord[i]*1000000)



values12 = list(spd_target_cord.values())



#####################################################################


dis_targetbtnreal = {}

for i in range(0, len(index_p4_df)):
    dis_targetbtnreal[i] = np.sqrt(np.square(dis_x_cord[i])+np.square(dis_y_cord[i])+np.square(dis_z_cord[i]))

keys = list(dis_targetbtnreal.keys())
values = list(dis_targetbtnreal.values())

values_std = []
values_std.append(np.std(values[0]))
values_std.append(np.std(values[1]))

values_mean = []
values_mean.append(np.mean(values[0]))
values_mean.append(np.mean(values[1]))

xxdata = []

###############################################################


keys1 = list(spd_cord.keys())
values1 = list(spd_cord.values())

values_std1 = []
values_std1.append(np.std(values1[0]))
values_std1.append(np.std(values1[1]))

values_mean1 = []
values_mean1.append(np.mean(values1[0]))
values_mean1.append(np.mean(values1[1]))

xxdata = []


#############################################################



for i in range(0, len(index_p4_df)):
    xxdata.append(np.full(1, i))
    print(xxdata)
    plt.bar(xxdata[i], values_mean[i], yerr=values_std[i], label='Move'+str(i+1))

plt.legend()
plt.ylim((0, 70))
plt.show()

timestamp_signal1 = np.arange(0, 30, 0.1)
for i in range(0, len(index_p4_df)):

    plt.plot(timestamp_signal1, values[i],label='Move'+str(i+1))

plt.legend()
plt.ylim((0,150))
plt.show()

################################################

timestamp_signal2 = np.arange(0, 19.9, 0.1)

plt.plot(timestamp_signal2, values12[0], label="Target Speed")
for i in range(0, len(index_df_4th)):
    
    plt.plot(timestamp_signal2, values1[i], label='Move'+str(i+1))

plt.legend()
plt.show()



####################################################
# 각도에 대한 부분은 추후에 진행할 것
#################################################

#rx1 = np.array(a2['target_TCP_pose_3']-a2['actual_TCP_pose_3'])
#ry1 = np.array(a2['target_TCP_pose_4']-a2['actual_TCP_pose_4'])
#rz1 = np.array(a2['target_TCP_pose_5']-a2['actual_TCP_pose_5'])

#print(rx1)
#print(ry1)
#print(rz1)

######################################################
hue1 = np.diff(values1[0])
hue1 = pd.Series(hue1)
plt.plot(timestamp_signal2[:-1],hue1.rolling(window=15, center=True).mean()*10)
plt.show()