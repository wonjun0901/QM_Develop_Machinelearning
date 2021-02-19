# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.datasets import load_boston

# boston 집 가격 데이터
# 회귀분석을 위한 데이터셋
# 회귀 : 라벨 데이터가 연속된 수치로 
#       구성되어 있는 데이터셋
# 분류 : 라벨 데이터가 범주형으로 
#       구성되어 있는 데이터셋
boston = load_boston()

# 회귀 분석을 위한 데이터 셋의 경우
# target_names 가 존재하지 않습니다.
# - target 자체가 정답이기 때문에
print(boston.keys())

# 학습을 위한 특성 데이터 추출(2차원)
X = pd.DataFrame(boston.data)
# 특성 데이터의 컬럼명을 지정
X.columns = boston.feature_names
print(X.head())

# 학습을 위한 라벨 데이터 추출(1차원)
y = pd.Series(boston.target)
print(y.head())

# 회귀분석을 위한 데이터 셋이므로
# value_counts 메소드를 사용하여 
# 유의미한 정보를 확인할 수 없음
print(y.value_counts())

# 회귀분석을 위한 데이터 셋이므로
# 값의 기본 통계를 확인할 수 있음
print(y.describe())














