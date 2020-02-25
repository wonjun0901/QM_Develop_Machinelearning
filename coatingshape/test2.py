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

import matplotlib
#Data1 = io.imread('coatingshape/x1_side_mid.tif', plugin='tifffile')
Data1 = io.imread('coatingshape/x1.0 S26.tif', as_gray=True)
Data2 = io.imread('coatingshape/x1.0 S26.tif', as_gray=True)
#Data1[Data1<120] = 255
# Data1 = Data1[200:350, :]
Data3 = gaussian_filter(Data2, 1.5, order=0)


#thresh = threshold_otsu(Data1, nbins=50)
#binary = Data1 > thresh

edges1 = roberts(Data2)
edges2 = sobel(Data2)
edges3 = feature.canny(Data2, sigma=1.5)
#fill_edges = ndi.binary_fill_holes(edges2)


markers = np.zeros_like(Data2)

markers[Data2 < 130] = 1
markers[Data2 > 190] = 2


segmentation = morphology.watershed(edges2, markers)
segmentation[segmentation >= 2] = 0

a1 = np.array(segmentation)
#print(np.count_nonzero(a1 == 1))

segmentation = ndi.binary_fill_holes(segmentation)

labeled_coins, num_features = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_coins, image=Data1)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(Data2, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=2, colors='y')
axes[1].imshow(image_label_overlay, cmap=plt.cm.gray)

#hyst = filters.apply_hysteresis_threshold(edges2, low, high)
#a1= np.array(labeled_coins)
#print(np.count_nonzero(a1 == 1))
#print(np.count_nonzero(a1 == 2))
#print(np.count_nonzero(a1 == 3))
#print(np.count_nonzero(a1 == 4))
#print(np.count_nonzero(a1 == 5))
#print(np.count_nonzero(a1 == 6))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
ax1.imshow(Data2, cmap=plt.cm.gray, interpolation='nearest')
ax1.axis('off')
ax2.imshow(markers, vmin=Data3.min(), vmax=Data3.max(), cmap=plt.cm.gray)
ax2.axis('off')
ax3.imshow(segmentation, cmap=plt.cm.gray)
ax3.axis('off')

plt.show()


#fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
#
#ax1.imshow(edges1, cmap=plt.cm.gray)
# ax1.axis('off')
#
#ax2.imshow(edges2, cmap=plt.cm.gray)
# ax2.axis('off')
#
#ax3.hist(Data1.ravel(), bins=512)
#
# fig.tight_layout()
# plt.show()
