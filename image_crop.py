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


image_path = "D:/DEV/Python/Practiceforeverything/QuadMedicine/coating_linear"
file_spec = '*.jpg'
load_pattern = os.path.join(image_path, file_spec)

image_collection = io.imread_collection(load_pattern)


plt.imshow(image_collection[0][210:519, 371:940])
plt.show()
