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
from sklearn.neural_network import MLPRegressor

Data1 = io.imread(
    'C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem/coating_ML/232.jpg')
reference = Data1

image_path = "C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem/coating_ML"
file_spec = '*.jpg'
load_pattern = os.path.join(image_path, file_spec)

image_collection = io.imread_collection(load_pattern)

image_collection1 = []

for a in range(316):

    image_collection1.append(match_histograms(
        image_collection[a], reference, multichannel=True))


data = []

for a in range(77, 316, 1):
    crop_image = image_collection1[a][391:638, 310:670]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

fname = 'C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem/200228_coatingdata.xlsx'

data_y = pd.read_excel(fname, header=None)

y = data_y[0][77:]
data = np.array(data)

X = pd.DataFrame(data)
y = pd.Series(y)

X_data = X.values/255.
y_data = y.values

X_train, X_test, y_train, y_test = \
    train_test_split(X_data, y_data, random_state=0)

Mlp = MLPRegressor(hidden_layer_sizes=(11,11,11,11), tol=1e-5, max_iter=50000, random_state=0).fit(X_train, y_train)

print('MLP train - score : ',
      Mlp.score(X_train, y_train))
print('MLP test - score : ',
      Mlp.score(X_test, y_test))

predicted = Mlp.predict(X_train)
predicted_test = Mlp.predict(X_test)

#print('(R2) train : ', r2_score(y_train, predicted))
#print('(R2) test : ', r2_score(y_test, predicted_test))

print('(MAE) - train : ',
      mean_absolute_error(y_train, predicted))

print('(MAE) - test : ',
      mean_absolute_error(y_test, predicted_test))

print('(MSE) - train : ',
      mean_squared_error(y_train, predicted))

print('(MSE) - test : ',
      mean_squared_error(y_test, predicted_test))

base_dir = "C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem"
file_nm = "result2.xlsx"
xlxs_dir = os.path.join(base_dir, file_nm)

df = pd.DataFrame({'real value':y_test, 'predicted value':predicted_test})
df.to_excel(xlxs_dir, sheet_name = 'Sheet1', na_rep = 'NaN', float_format = "%.2f", 
            header = True, index = True, index_label = "id", 
            startrow = 1, startcol = 1, freeze_panes = (2, 0)) 