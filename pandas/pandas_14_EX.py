# -*- coding: utf-8 -*-

import pandas as pd

fname = '../../data/winequality-white.csv'
# 데이터의 분류를 다른 문자로 사용하는 경우
# sep 매개변수의 값으로 구분 문자를 지정할 수 있음
wine = pd.read_csv(fname, sep=';')
print(wine)

# 데이터프레임의 출력 컬럼의 개수를 지정
pd.options.display.max_columns = 15
# 데이터프레임의 출력 행의 개수를 지정
pd.options.display.max_rows = 10
print(wine)

# wine 데이터의 일부분을 확인
print(wine.head(3))
print(wine.tail(3))

# wine 데이터의 개수 및 결측데이터 확인
# - 특성(컬럼)의 개수도 확인할 수 있음
print(wine.info())

# wine 데이터의 수치 데이터의 통계 확인
print(wine.describe())

# 특성 데이터와 라벨 데이터를 분할하는 작업
# DataFrame의 iloc 연산
# 인덱스 정보를 기반으로 데이터프레임을 분할
# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 제외하고 분할하는 코드
X_df = wine.iloc[:, :-1]
print(type(X_df))

# 아래의 코드는 전체 행(샘플)에서 마지막 열을
# 추출하는 코드
# 아래와 같이 특정 열만 추출하는 경우
# DataFrame 이 아닌 Series 타입으로 반환됩니다.
y_s = wine.iloc[:, -1]
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