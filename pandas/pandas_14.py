# -*- coding: utf-8 -*-

import pandas as pd

# 파일의 경로를 포함한 이름을 저장
# csv 파일
# - , 를 기준으로 데이터가 분리되어 있는 파일
# - 각 라인이 하나의 샘플을 의미함
fname = '../../data/iris.csv'

# 특정 파일을 DataFrame으로 로딩
# read_csv 함수
# - csv 포맷으로 저장된 파일의 내용을 DataFrame 타입으로
#   반환하는 함수
# - 첫 번째 라인의 값을 사용하여 헤더(컬럼의 이름)의 
#   정보로 사용
# - 두 번째 라인부터 데이터로 취급함
iris = pd.read_csv(fname)

# iris.csv 파일을 로딩한 DataFrame의 내용을 확인
# - 기본적으로 데이터의 앞과 뒤의 5개씩 확인할 수 있음
print(iris)

# pandas 옵션을 수정
# - Data의 샘플 출력 개수를 수정
pd.options.display.max_rows = 200
print(iris)

# iris 데이터의 일부분을 확인
print(iris.head(3))
print(iris.tail(3))

# iris 데이터의 개수 및 결측데이터 확인
# - 특성(컬럼)의 개수도 확인할 수 있음
print(iris.info())

# iris 데이터의 수치 데이터의 통계 확인
print(iris.describe())

# iris 데이터는 iris 품종을 맞추기 위한 
# 데이터 셋으로 앞의 4가지가 특성 데이터
# 마지막 열의 데이터가 라벨 데이터임

# 특성 데이터와 라벨 데이터를 분할하는 작업
# DataFrame의 iloc 연산
# 인덱스 정보를 기반으로 데이터프레임을 분할
# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 제외하고 분할하는 코드
X_df = iris.iloc[:, :-1]
print(type(X_df))

# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 추출하는 코드
# 아래와 같이 특정 열만 추출하는 경우
# DataFrame 이 아닌 Series 타입으로 반환됩니다.
y_s = iris.iloc[:, -1]
print(type(y_s))

# 라벨 데이터의 분포를 확인
print(y_s.value_counts())
print(y_s.value_counts() / len(y_s))

# 특정 데이터의 분포를 확인
from matplotlib import pyplot as plt
X_df.hist()
plt.show()

X_df.plot.hist()
plt.show()
























