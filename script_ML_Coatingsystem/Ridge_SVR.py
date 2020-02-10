import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
import scipy.io
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import LinearSVR, SVR


image_path = "D:/DEV/Python/Practiceforeverything/QuadMedicine/coating_linear"
file_spec = '*.jpg'
load_pattern = os.path.join(image_path, file_spec)

image_collection = io.imread_collection(load_pattern)

data = []

for a in range(253):
    crop_image = image_collection[a][210:519, 371:940]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

fname = 'D:/DEV/Python/Practiceforeverything/QuadMedicine/data_coating.xlsx'

data_y = pd.read_excel(fname, header=None)

y = data_y[0]
data = np.array(data)

X = pd.DataFrame(data)
y = pd.Series(y)

X_data = X.values
y_data = y.values

X_train, X_test, y_train, y_test = \
    train_test_split(X_data, y_data, random_state=0)

ridge_model = Ridge(alpha=1.0, random_state=0).fit(X_train, y_train)

svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1).fit(X_train, y_train)
svr_lin = SVR(kernel='linear', C=1, gamma='auto').fit(X_train, y_train)
svr_poly = SVR(kernel='poly', C=1, gamma='auto', degree=3,
               epsilon=.1, coef0=1).fit(X_train, y_train)


print('svr_rbf score : ', svr_rbf.score(X_train, y_train))
print('svr_rbf test score : ', svr_rbf.score(X_test, y_test))

print('svr_lin score : ', svr_lin.score(X_train, y_train))
print('svr_lin test score : ', svr_lin.score(X_test, y_test))

print('svr_poly score : ', svr_poly.score(X_train, y_train))
print('svr_poly test score : ', svr_poly.score(X_test, y_test))

predicted = ridge_model.predict(X_train)
predicted_test = ridge_model.predict(X_test)

predicted_svr_rbf = svr_rbf.predict(X_train)
predicted_test_svr_rbf = svr_rbf.predict(X_test)

predicted_svr_lin = svr_lin.predict(X_train)
predicted_test_svr_lin = svr_lin.predict(X_test)

predicted_svr_poly = svr_poly.predict(X_train)
predicted_test_svr_poly = svr_poly.predict(X_test)


#################################################
print('(MAE) - train - ridge : ',
      mean_absolute_error(y_train, predicted))

print('(MAE) - test - ridge : ',
      mean_absolute_error(y_test, predicted_test))

print('(MSE) - train - ridge : ',
      mean_squared_error(y_train, predicted))

print('(MSE) - test - ridge : ',
      mean_squared_error(y_test, predicted_test))

####################################################
print('(MAE) - train - svr_rbf : ',
      mean_absolute_error(y_train, predicted_svr_rbf))

print('(MAE) - test - svr_rbf : ',
      mean_absolute_error(y_test, predicted_test_svr_rbf))

print('(MSE) - train - svr_rbf : ',
      mean_squared_error(y_train, predicted_svr_rbf))

print('(MSE) - test - svr_rbf : ',
      mean_squared_error(y_test, predicted_test_svr_rbf))


########################################################
print('(MAE) - train - svr_lin : ',
      mean_absolute_error(y_train, predicted_svr_lin))

print('(MAE) - test - svr_lin : ',
      mean_absolute_error(y_test, predicted_test_svr_lin))

print('(MSE) - train - svr_lin : ',
      mean_squared_error(y_train, predicted_svr_lin))

print('(MSE) - test - svr_lin : ',
      mean_squared_error(y_test, predicted_test_svr_lin))

###########################svr_lin##############################
print('(MAE) - train - svr_poly : ',
      mean_absolute_error(y_train, predicted_svr_poly))

print('(MAE) - test - svr_poly : ',
      mean_absolute_error(y_test, predicted_test_svr_poly))

print('(MSE) - train - svr_poly : ',
      mean_squared_error(y_train, predicted_svr_poly))

print('(MSE) - test - svr_poly : ',
      mean_squared_error(y_test, predicted_test_svr_poly))
