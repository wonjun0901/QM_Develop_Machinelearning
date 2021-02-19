import pandas as pd
import numpy as np
from math import sqrt

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
#p2_df = p2_df.reset_index(drop=True)
#print(p2_df)
# shift to 1 below to all data
p2_df_shift = p2_df.shift(periods=1, fill_value=0)

# substract
p3_df = p2_df - p2_df_shift

#print(p2_df)
#print(p3_df)

# data processing - 차이가 0.02 이상인 부근의 데이터 프레임만 찾기
p4_df = p3_df[p3_df['timestamp']>0.02]

# 차이가 0.02이상의 부근인 데이터에 대한 인덱스 찾기
index_p4_df = p4_df.index.tolist()
#print(index_p4_df)
#해당 인덱스에 대한 데이터 프레임
a1 = p1_df.iloc[index_p4_df]

a2 = p1_df.loc[4665:4681]


#print(a2)


x_cord = np.array(a2['target_TCP_pose_0']-a2['actual_TCP_pose_0'])*1000000
y_cord = np.array(a2['target_TCP_pose_1']-a2['actual_TCP_pose_1'])*1000000
z_cord = np.array(a2['target_TCP_pose_2']-a2['actual_TCP_pose_2'])*1000000

#print(a2['actual_TCP_pose_3'])
a3 = a2['actual_TCP_pose_3']
print(a3)


if a3> 3.12414 and a3 < 3.15905:
    a2['actual_TCP_pose_3'] = a2['actual_TCP_pose_3'] 

if a2['actual_TCP_pose_3'] < -3.12414 and a2['actual_TCP_pose_3'] > -3.15905:
    a2['actual_TCP_pose_3'] = a2['actual_TCP_pose_3'] + 6.28319

print(a2['actual_TCP_pose_3'])


#a2['actual_TCP_pose_4'] = correction_radian(a2['actual_TCP_pose_4'])
#a2['actual_TCP_pose_5'] = correction_radian(a2['actual_TCP_pose_5'])

rx1 = np.array(a2['target_TCP_pose_3']-a2['actual_TCP_pose_3'])
ry1 = np.array(a2['target_TCP_pose_4']-a2['actual_TCP_pose_4'])
rz1 = np.array(a2['target_TCP_pose_5']-a2['actual_TCP_pose_5'])

print(rx1)
print(ry1)
print(rz1)

for i in range(0, len(rx1)):

    if rx1[i] < 0.001 :
        rx1[i] = 0
    if ry1[i] < 0.001:
        ry1[i] = 0    
    if rz1[i] < 0.001:
        rz1[i] = 0    

#print(rx1)
#print(ry1)
#print(rz1)
deviation = []
for i in range(0, len(x_cord)):
    
    deviation.append(sqrt(x_cord[i]*x_cord[i] + y_cord[i]*y_cord[i]))

#print(deviation)