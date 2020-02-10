from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
import scipy.io

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
    train_test_split(X.values, y.values,
                     stratify=y, random_state=1)


lr_model = LogisticRegression(
    solver='lbfgs', multi_class='multinomial',
    max_iter=10000)
knn_model = KNeighborsClassifier(n_neighbors=3)
tree_model = DecisionTreeClassifier()
ensemble_model = VotingClassifier(
    estimators=[('lr', lr_model), ('knn', knn_model), ('tree', tree_model)])

lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
ensemble_model.fit(X_train, y_train)

print("LogisticRegression 평가(train) : ", lr_model.score(X_train, y_train))
print("KNeighborsClassifier 평가(train) : ", knn_model.score(X_train, y_train))
print("DecisionTreeClassifier 평가(train) : ",
      tree_model.score(X_train, y_train))
print("VotingClassifier 평가(train) : ", ensemble_model.score(X_train, y_train))

print("=" * 35)

print("LogisticRegression 평가(test) : ", lr_model.score(X_test, y_test))
print("KNeighborsClassifier 평가(test) : ", knn_model.score(X_test, y_test))
print("DecisionTreeClassifier 평가(test) : ", tree_model.score(X_test, y_test))
print("VotingClassifier 평가(test) : ", ensemble_model.score(X_test, y_test))

# print(image_collection[0][1][-1][-1])
#image1 = image1.color.rgb2gray
# plt.imshow(image1)
# plt.show()
## edge_roberts = roberts(image1)
## edge_sobel = sobel(image1)
##
# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True,
# figsize=(8, 4))
#####
###ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
###ax[0].set_title('Roberts Edge Detection')
#####
###ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
###ax[1].set_title('Sobel Edge Detection')
#####
# for a in ax:
# a.axis('off')

# plt.tight_layout()
# plt.show()
