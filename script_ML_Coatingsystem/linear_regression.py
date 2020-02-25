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
from skimage.exposure import match_histograms

Data1 = io.imread('coating_linear/1.jpg')
reference = Data1

image_path = "C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/coating_linear"
file_spec = '*.jpg'
load_pattern = os.path.join(image_path, file_spec)

image_collection = io.imread_collection(load_pattern)

image_collection1 = []

for a in range(253):
      
      image_collection1.append(match_histograms(image_collection[a], reference, multichannel=True)) 
      

data = []

for a in range(253):
    crop_image = image_collection1[a][210:519, 371:940]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

fname = 'C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem/data_coating.xlsx'

data_y = pd.read_excel(fname, header=None)

y = data_y[0]
data = np.array(data)

X = pd.DataFrame(data)
y = pd.Series(y)

X_data = X.values
y_data = y.values

X_train, X_test, y_train, y_test = \
    train_test_split(X_data, y_data, random_state=0)

linear_model = LinearRegression().fit(X_train, y_train)

print('Gradient boost regressor score : ',
      linear_model.score(X_train, y_train))
print('Gradient boost regressor test score : ',
      linear_model.score(X_test, y_test))

predicted = linear_model.predict(X_train)
predicted_test = linear_model.predict(X_test)

print('(R2) train : ', r2_score(y_train, predicted))
print('(R2) test : ', r2_score(y_test, predicted_test))

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
