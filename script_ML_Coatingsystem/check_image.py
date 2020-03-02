import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters

Data1 = io.imread(
    'D:/DEV/Python/QM_Develop_Machinelearning/script_ML_Coatingsystem/coating_ML/232.jpg')

plt.imshow(Data1)

plt.show()
