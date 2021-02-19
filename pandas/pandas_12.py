# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

# 데이터프레임의 병합
# - concat 함수의 사용
# - 단순 데이터 연결을 지원
# - 수평으로 데이터를 연결하는 경우 
#   axis=1로 인수를 설정

df1 = pd.DataFrame(
        np.arange(1, 7).reshape(-1, 2),
        index=['a', 'b', 'c'],
        columns=['data1', 'data2'])
print(df1)

df2 = pd.DataFrame(
        np.arange(7, 11).reshape(-1, 2),
        index=['a', 'c'],
        columns=['data3', 'data4'])
print(df2)

# - 데이터의 병합 시, 인덱스의 값을 기준으로 병합됨
# - 일치하는 인덱스 없는 경우 
#   해당 위치의 컬럼은 NaN으로 채워집니다.
r = pd.concat([df1, df2], axis=1)
print(r)
















