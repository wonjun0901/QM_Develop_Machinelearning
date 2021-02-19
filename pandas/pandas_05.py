# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

data = {'year' : [2017, 2018, 2017],
        'GDP Rate' : [2.8, 3.1, 3.],
        'GDP' : ['1.637M', None, '2.197M']}

df = pd.DataFrame(data)
print(df)

# DataFrame 타입에서 사용할 수 있는 메소드
# 1. info 메소드는 현재 데이터프레임 내부에
# 존재하는 데이터 전체 개수
# 각 컬럼의 존재하는 실제 데이터 개수
# (NULL이 아닌 데이터)
# 각 컬럼의 데이터 타입 정보를 출력
print(df.info())

# 데이터프레임에 저장된 수치데이터에 대한
# 간략한 통계 정보를 출력할 수 있는 
# describe 메소드
# (수치데이터에 대한 정보만 출력)
print(df.describe())

# 각각의 열에 대해서 통계 메소드를 제공
print(df['year'].sum())
print(df['year'].mean())
print(df['year'].max())
print(df['year'].min())

# 특정 열에 저장된 데이터의 개수를 반환
# (중복이 제거된 각 데이터의 개수를 반환)
print(df.year.value_counts())

# value_counts 함수의 결과를
# 전체 데이터의 개수로 나누어서
# 비율을 확인할 수 있음
print(df.year.value_counts() / len(df))




















