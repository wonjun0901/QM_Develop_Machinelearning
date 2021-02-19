# -*- coding: utf-8 -*-

# pandas를 사용하여 사이킷 런의 학습데이터를 
# 처리하는 방법
import pandas as pd

# 사이킷 런의 sklearn.datasets 패키지 내부의
# 학습 데이터를 로딩하는 코드
# (load_... 이름으로 함수가 정의되어 있음)
# - iris 데이터를 제공할 수 있는 load_iris 함수
from sklearn.datasets import load_iris

# iris 데이터를 로딩하는 코드
iris = load_iris()

# Bunch 클래스 타입의 값이 반환
# 파이썬의 딕셔너리와 유사한 타입으로
# 키값을 사용하여 데이터를 추출할 수 있음
print(type(iris))

# Bunch 클래스 keys 메소드
# 사용할 수 있는 키의 목록을 반환하는 메소드
# data 키 : X 데이터를 반환받을 수 있는 키 값
print(iris.keys())

print(iris.data)
# target 키 : y 데이터를 반환받을 수 있는 키 값
print(iris.target)
# target_names : y 데이터의 실제 정보을 
#               반환받을 수 있는 키 값
print(iris.target_names)
# feature_names : 각 특성의 이름을 제공
print(iris.feature_names)
# DESCR : 데이터 셋에 대한 설명
print(iris.DESCR)

# Bunch 클래스 타입의 값이 반환
# 파이썬의 딕셔너리와 유사한 타입으로
# 키 값을 사용하여 데이터를 추출할 수 있음

# 키 값 'data' 는 특성 데이터를 반환
# (numpy 2차원 배열의 형태)

# pandas 데이터 프레임으로 
# 특성 데이터를 저장
X_df = pd.DataFrame(iris.data)

# Bunch 클래스의 타입의 feature_names 키 값을
# 사용하여 데이터프레임의 헤더를 설정
X_df.columns = iris.feature_names

# iris 데이터의 샘플 개수 및 결측데이터 확인
print(X_df.info())

# iris 데이터의 수치 데이터 통계 확인
print(X_df.describe())

# 라벨 데이터의 Series 생성
# 키 값 'target' 은 라벨 데이터를 반환
# (numpy 1차원 배열의 형태)
y_s = pd.Series(iris.target)

# 데이터의 확인
# 사이킷 런에서 제공되는 데이터들은
# 전처리가 완료된 상태의 데이터이므로
# 문자열이 아닌 수치 데이터가 제공됨
print(y_s)

# 라벨 데이터의 분포 확인
print(y_s.value_counts() / len(y_s))

# 특성 데이터와 라벨 데이터의 결합
iris_all = pd.concat([X_df, y_s], axis=1)
iris_all.columns = iris.feature_names + ['label']
print(iris_all)

# 데이터 프레임 내부의 특성 간 상관관계를
# 분석하여 반환하는 메소드 - corr()
iris_corr = iris_all.corr()

# 결과(라벨) 데이터와 특성 데이터들간의
# 상관관계를 출력 
# 1에 가까울수록 강한 양의 상관관계를 보여줌
# (라벨 데이터의 수치가 커질수록 특성 데이터의 
# 값이 증가)
# 0에 가까울수록 약한 상관관계를 보여줌
# (특성 데이터의 수치 변화가 특성 데이터와 관계없음)
# -1에 가까울수록 강한 음의 상관관계를 보여줌
# (특성 데이터의 수치가 커질수록 특성 데이터의 
# 값이 감소)
print(iris_corr)











