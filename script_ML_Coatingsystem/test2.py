from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import xgboost as xgb
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
import scipy.io
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC

# image = camera()
image_path = "D:/DEV/Python/Practiceforeverything/QuadMedicine/imagecollection"
file_spec = '*.png'
load_pattern = os.path.join(image_path, file_spec)

image_collection = io.imread_collection(load_pattern)
# print(image_collection.files)

numpy_array_0 = np.zeros((25, 1))
numpy_array_1 = np.ones((50, 1))
numpy_array_2 = np.full((40, 1), 2)
result = np.r_[numpy_array_0, numpy_array_1, numpy_array_2].reshape(-1)

print(result)
data1 = image_collection[0]
data1 = rgba2rgb(data1)
print(data1.shape)
data1 = np.array(data1).reshape(-1)
print(data1.shape)
#data2 = image_collection[1]
data = []

for a in range(115):
    converted_data = np.array(rgba2rgb(image_collection[a])).reshape(-1)
    data.append(converted_data)

    #data1 = rgb2gray(data1)
    # plt.imshow(data1)
data2 = np.array(data)
print(data2.shape)

X = pd.DataFrame(data2)
y = pd.Series(result)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=42)

# 다항 분류를 위한 XGBClassifier 객체를 생성
# objective 하이퍼 파라메터의 값을 multi:softmax or multi:softprob 으로 설정
svc = SVC(gamma=0.0005, C=1, random_state=0)

svc.fit(X_train, y_train)

print("훈련 세트 정확도: {:.2f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.2f}".format(svc.score(X_test, y_test)))

pred_train = svc.predict(X_train)
pred_test = svc.predict(X_test)

print("훈련 데이터 - classification_report")
print(classification_report(y_train, pred_train))

print("테스트 데이터 - classification_report")
print(classification_report(y_test, pred_test))


print(confusion_matrix(y_train, pred_train))
print(confusion_matrix(y_test, pred_test))
