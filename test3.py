import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
import scipy.io
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# image = camera()
image_path_0 = "D:/DEV/Python/Practiceforeverything/QuadMedicine/coatingimages/0"
image_path_1 = "D:/DEV/Python/Practiceforeverything/QuadMedicine/coatingimages/1"
image_path_2 = "D:/DEV/Python/Practiceforeverything/QuadMedicine/coatingimages/2"
file_spec = '*.jpg'
load_pattern_0 = os.path.join(image_path_0, file_spec)
load_pattern_1 = os.path.join(image_path_1, file_spec)
load_pattern_2 = os.path.join(image_path_2, file_spec)

image_collection_0 = io.imread_collection(load_pattern_0)
image_collection_1 = io.imread_collection(load_pattern_1)
image_collection_2 = io.imread_collection(load_pattern_2)

data = []

numpy_array_0 = np.zeros((106, 1))
numpy_array_1 = np.ones((190, 1))
numpy_array_2 = np.full((144, 1), 2)
result = np.r_[numpy_array_0, numpy_array_1, numpy_array_2].reshape(-1)

for a in range(106):
    crop_image = image_collection_0[a][440:715, 395:810]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

for a in range(190):
    crop_image = image_collection_1[a][440:715, 395:810]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

for a in range(144):
    crop_image = image_collection_2[a][440:715, 395:810]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

data = np.array(data)
print(data.shape)

X = pd.DataFrame(data)
y = pd.Series(result)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, stratify=y, random_state=0)

# 다항 분류를 위한 XGBClassifier 객체를 생성
# objective 하이퍼 파라메터의 값을 multi:softmax or multi:softprob 으로 설정
#svc = SVC(gamma=0.01, C=0.8, random_state=0)

knn_model = KNeighborsClassifier(n_neighbors=7, n_jobs=-1)
knn_model.fit(X_train, y_train)


# Linear_svc = LinearSVC(
#    C=0.1, max_iter=10000)
##svc.fit(X_train, y_train)
#Linear_svc.fit(X_train, y_train)

kfold = KFold(n_splits=5, shuffle=True, random_state=3)

scores = cross_val_score(knn_model, X, y, cv=kfold)

print("crossvalidation score(Logistic_model) : ", scores)
print("crossvalidation score Average(Logistic_model) : ", scores.mean())

print("train set accuracy(Logistic_model): {:.2f}".format(
    knn_model.score(X_train, y_train)))
print("test set accuracy(Logistic_model): {:.2f}".format(
    knn_model.score(X_test, y_test)))

pred_train = knn_model.predict(X_train)
pred_test = knn_model.predict(X_test)

print("train data - classification_report(Logistic_model)")
print(classification_report(y_train, pred_train))

print("test data - classification_report(Logistic_model)")
print(classification_report(y_test, pred_test))

print("train data - confusion matrix(Logistic_model)")
print(confusion_matrix(y_train, pred_train))
print("test data - confusion matrix(Logistic_model)")
print(confusion_matrix(y_test, pred_test))
