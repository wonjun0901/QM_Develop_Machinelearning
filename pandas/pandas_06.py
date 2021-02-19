# -*- coding: utf-8 -*-

import pandas as pd

data = {'year' : [2017, 2018, 2017],
        'GDP Rate' : [2.8, 3.1, 3.],
        'GDP' : ['1.637M', None, '2.197M']}

df = pd.DataFrame(data)
print(df)

# 데이터프레임의 앞 부분의 데이터 추출
# - head 메소드
# - 기본값 = 5개
print(df.head(2))

# 데이터프레임의 뒷 부분의 데이터 추출
# - tail 메소드
# - 기본값 = 5개
print(df.tail(2))














