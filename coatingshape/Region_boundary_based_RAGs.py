import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import data, io, filters
from skimage.color import rgba2rgb, rgb2gray
from skimage.external import tifffile
from skimage import feature
from skimage.filters import roberts, sobel, sobel_h, sobel_v
from skimage.filters import threshold_otsu
from skimage import morphology
from scipy import ndimage as ndi
from skimage.color import label2rgb

from skimage import exposure
from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries, chan_vese
from skimage.future import graph

from skimage import segmentation, color

import matplotlib
#Data1 = io.imread('coatingshape/x1_side_mid.tif', plugin='tifffile')
Data1 = io.imread('coatingshape/1.tif', as_gray=True)
Data2 = io.imread('coatingshape/1.tif', as_gray=True)
#Data1[Data1<120] = 255
# Data1 = Data1[200:350, :]
image = img_as_float(Data2)


cv = chan_vese(image, mu=0.7, lambda1=1, lambda2=1, tol=1e-3, max_iter=1000,
               dt=0.5, init_level_set="checkerboard", extended_output=True)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].set_title("Original Image", fontsize=12)

ax[1].imshow(cv[0], cmap="gray")
ax[1].set_axis_off()
title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
ax[1].set_title(title, fontsize=12)

ax[2].imshow(cv[1], cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Final Level Set", fontsize=12)

ax[3].plot(cv[2])
ax[3].set_title("Evolution of energy over iterations", fontsize=12)

fig.tight_layout()
plt.show()