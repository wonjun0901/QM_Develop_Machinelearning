# -*- coding: utf-8 -*-

# 머신러닝 예제 코드

# 1. 데이터의 로딩
import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = pd.DataFrame(cancer.data)
y = pd.Series(cancer.target)

# 2. 데이터의 특성 확인
print(X.info())
print(X.describe())

print(y.value_counts())
print(y.value_counts() / len(y))

# 3. 데이터의 학습
# - 학습을 위해서 사용할 클래스의 로딩
from sklearn.neighbors import KNeighborsClassifier

# - 학습을 위한 클래스의 객체 생성
#   (하이퍼 파라메터의 제어)
model = KNeighborsClassifier(n_neighbors=3)

# - 학습
model.fit(X.values, y.values)

# 4. 모델(알고리즘)의 평가
print("모델의 정확도 : ", 
      model.score(X.values, y.values))














