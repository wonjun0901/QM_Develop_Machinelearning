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
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.exposure import match_histograms

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

for a in range(316):
    crop_image = image_collection1[a][391:638, 310:670]
    converted_data = np.array(crop_image).reshape(-1)
    data.append(converted_data)

fname = 'C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem/200228_coatingdata.xlsx'

data_y = pd.read_excel(fname, header=None)

y = data_y[0]
data = np.array(data)

X = pd.DataFrame(data)
y = pd.Series(y)


X_data = X.values/255.
y_data = y.values

X_train, X_test, y_train, y_test = \
    train_test_split(X_data, y_data, random_state=0)

RFRegressor_model = RandomForestRegressor(
    n_estimators=5000,  max_depth=3, random_state=0, n_jobs=-1)
gradient_model = GradientBoostingRegressor(
    n_estimators=15000, max_features=0.1, subsample=0.5, max_depth=3, random_state=0, loss='ls')
linear_model = LinearRegression()
svr_lin = SVR(kernel='linear', C=0.5, gamma='auto')
Mlp = MLPRegressor(hidden_layer_sizes=(11,11,11,11), tol=1e-5, max_iter=70000, random_state=0)


votingregressor = VotingRegressor(estimators=[('RF', RFRegressor_model), (
    'gb', gradient_model), ('lin', linear_model), ('svr_lin', svr_lin), ('Mlp', Mlp)]).fit(X_train, y_train)


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

base_dir = "C:/Users/wonju/Documents/GitHub/QM_Develop_Machinelearning/script_ML_Coatingsystem"
file_nm = "voting_result.xlsx"
xlxs_dir = os.path.join(base_dir, file_nm)

df = pd.DataFrame({'real value':y_test, 'predicted value':predicted_test})
df.to_excel(xlxs_dir, sheet_name = 'Sheet1', na_rep = 'NaN', float_format = "%.2f", 
            header = True, index = True, index_label = "id", 
            startrow = 1, startcol = 1, freeze_panes = (2, 0)) 