import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
import scipy.io
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

RFRegressor_model = RandomForestRegressor(
    n_estimators=500,  max_depth=3, random_state=0, n_jobs=-1)
gradient_model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0, loss='ls')
linear_model = LinearRegression()
svr_lin = SVR(kernel='linear', C=1, gamma='auto')

votingregressor = VotingRegressor(estimators=[('RF', RFRegressor_model), (
    'gb', gradient_model), ('lin', linear_model), ('svr_lin', svr_lin)]).fit(X_train, y_train)


print('votingregressor score : ', votingregressor.score(X_train, y_train))
print('votingregressor test score : ', votingregressor.score(X_test, y_test))

predicted = votingregressor.predict(X_train)
predicted_test = votingregressor.predict(X_test)

print('(MAE) - train : ',
      mean_absolute_error(y_train, predicted))

print('(MAE) - test : ',
      mean_absolute_error(y_test, predicted_test))

print('(MSE) - train : ',
      mean_squared_error(y_train, predicted))

print('(MSE) - test : ',
      mean_squared_error(y_test, predicted_test))

print(y_test)
print(predicted_test)
