# -*- coding: utf-8 -*-

import pandas as pd

# pandas의 데이터 저장 구조
# 1차원 : Series
# 2차원 : DataFrame
# 3차원 : Panel

# 딕셔너리 변수의 선언
# { 키1 : 값1, 키2 : 값2, ... 키N : 값N}
data = {'year' : [2017, 2018, 2019],
        'GDP Rate' : [2.8, 3.1, 3.],
        'GDP' : ['1.637M', '1.859M', '2.197M']}

# 딕셔너리 변수를 사용하여 DataFrame 객체 생성
# - 각 키의 값들은 동일한 개수를 가져야함
# - 해당되는 위치에 데이터가 없는 경우 
#   결측데이터(Nan)가 대입됨
df = pd.DataFrame(data)

print(f'type(df) -> {type(df)}')
print(df)











